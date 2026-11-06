"""Offline analysis of explicitly aligned anonymous human GTR annotations.

This implements metrics for paper section 2.3; it does not recover its original
three-rater/50-clip cohort or infer ground truth from filenames or row order.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

AXES = {"g": (0, 4), "t": (1, 5), "r": (0, 6)}
FIELDS = {"clip_id", "rater_id", *(f"gt_{axis}" for axis in AXES),
          *(f"rating_{axis}" for axis in AXES)}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def identifier(value, field, number):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value):
        raise ValueError(f"Row {number}: {field} requires a 1..128-character ASCII ID token; anonymize before input")
    return value


def level(value, axis, *, missing, number, field):
    if missing and (value is None or value == ""):
        return None
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        value = int(value)
    low, high = AXES[axis]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Row {number}: {field} requires an integer {low}..{high}; no scale conversion is inferred")
    return value


def unique_object_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field names are ambiguous and are not accepted")
        result[key] = value
    return result


def load_annotations(path):
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV requires unique column names")
        if set(reader.fieldnames) != FIELDS:
            raise ValueError("CSV columns must exactly match the documented anonymous annotation schema")
        rows = list(reader)
        format_name = "CSV"
    elif suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line, object_pairs_hook=unique_object_fields) for line in text.splitlines() if line.strip()]
        format_name = "JSONL"
    else:
        rows = json.loads(text, object_pairs_hook=unique_object_fields)
        if isinstance(rows, dict):
            if set(rows) != {"schema_version", "ratings"} or type(rows["schema_version"]) is not int or rows["schema_version"] != 1:
                raise ValueError("JSON object requires schema_version=1 and ratings only")
            rows = rows["ratings"]
        format_name = "JSON"
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must contain a nonempty annotation list")
    seen, truths, annotations = set(), {}, []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"Row {number}: fields must exactly match the documented schema; use null/blank for missing ratings")
        clip = identifier(row["clip_id"], "clip_id", number)
        rater = identifier(row["rater_id"], "rater_id", number)
        key = (clip, rater)
        if key in seen:
            raise ValueError(f"Duplicate (clip_id, rater_id) annotation at row {number}")
        seen.add(key)
        ground_truth = {axis: level(row[f"gt_{axis}"], axis, missing=False, number=number,
                                   field=f"gt_{axis}") for axis in AXES}
        if (ground_truth["g"] == 0) != (ground_truth["r"] == 0):
            raise ValueError(f"Row {number}: GT must belong to the 125 legal combinations (G=0 iff R=0)")
        if clip in truths and truths[clip] != ground_truth:
            raise ValueError(f"Conflicting/ambiguous ground truth for clip_id {clip}")
        truths[clip] = ground_truth
        ratings = {axis: level(row[f"rating_{axis}"], axis, missing=True, number=number,
                              field=f"rating_{axis}") for axis in AXES}
        inconsistent = (ratings["g"] is not None and ratings["r"] is not None and
                        ((ratings["g"] == 0) != (ratings["r"] == 0)))
        annotations.append({"annotation_index": number, "clip_id": clip, "rater_id": rater,
                            "ground_truth": ground_truth, "ratings": ratings,
                            "rater_cross_axis_inconsistency": inconsistent})
    return annotations, {"sha256": sha256(raw), "bytes": len(raw), "format": format_name,
                         "schema_version": 1}


def pearson(pairs):
    n = len(pairs)
    result = {"paired_count": n, "coefficient": None, "undefined_reason": None}
    if n < 2:
        result["undefined_reason"] = "fewer_than_two_pairs"
        return result
    xconstant = all(x == pairs[0][0] for x, _ in pairs)
    yconstant = all(y == pairs[0][1] for _, y in pairs)
    result.update(ground_truth_constant=xconstant, rating_constant=yconstant)
    if xconstant or yconstant:
        result["undefined_reason"] = "constant_ground_truth_or_rating"
        return result
    xmean = math.fsum(x for x, _ in pairs) / n
    ymean = math.fsum(y for _, y in pairs) / n
    centered = [(x - xmean, y - ymean) for x, y in pairs]
    xss = math.fsum(x * x for x, _ in centered)
    yss = math.fsum(y * y for _, y in centered)
    if xss == 0 or yss == 0:
        result["undefined_reason"] = "zero_centered_variance"
        return result
    result["coefficient"] = max(-1.0, min(1.0, math.fsum(x * y for x, y in centered) / math.sqrt(xss * yss)))
    return result


def axis_metrics(rows):
    metrics = {}
    for axis in AXES:
        pairs = [(row["ground_truth"][axis], row["ratings"][axis]) for row in rows
                 if row["ratings"][axis] is not None]
        if axis != "r":
            metric = pearson(pairs)
        else:
            correct = sum(gt == rating for gt, rating in pairs)
            metric = {"paired_count": len(pairs), "correct_count": correct,
                      "accuracy": correct / len(pairs) if pairs else None}
        metric["missing_rating_count"] = len(rows) - len(pairs)
        metrics[axis.upper()] = metric
    return metrics


def analyze(annotations, *, expected_clips=None, expected_raters=None):
    clips = sorted({row["clip_id"] for row in annotations})
    raters = sorted({row["rater_id"] for row in annotations})
    by_rater = {}
    for rater in raters:
        rows = [row for row in annotations if row["rater_id"] == rater]
        present = {row["clip_id"] for row in rows}
        by_rater[rater] = {"annotation_count": len(rows), "clip_count": len(present),
                           "missing_clip_ids_from_observed_set": sorted(set(clips) - present),
                           "metrics": axis_metrics(rows)}
    macro = {}
    for axis in AXES:
        key = "accuracy" if axis == "r" else "coefficient"
        values = {rater: report["metrics"][axis.upper()][key] for rater, report in by_rater.items()}
        defined = [value for value in values.values() if value is not None]
        macro[axis.upper()] = {"mean": math.fsum(defined) / len(defined) if defined else None,
                               "defined_rater_count": len(defined),
                               "undefined_rater_ids": [rater for rater, value in values.items() if value is None]}
    clip_ledger = []
    for clip in clips:
        rows = [row for row in annotations if row["clip_id"] == clip]
        votes = {axis: [row["ratings"][axis] for row in rows if row["ratings"][axis] is not None] for axis in AXES}
        clip_ledger.append({"clip_id": clip, "ground_truth": rows[0]["ground_truth"],
                            "annotation_count": len(rows),
                            "available_ratings_by_axis": {axis.upper(): len(values) for axis, values in votes.items()},
                            "mean_rating_G": math.fsum(votes["g"]) / len(votes["g"]) if votes["g"] else None,
                            "mean_rating_T": math.fsum(votes["t"]) / len(votes["t"]) if votes["t"] else None,
                            "R_category_counts": dict(sorted(Counter(votes["r"]).items()))})
    clip_mean_metrics = {axis.upper(): pearson([(row["ground_truth"][axis], row[f"mean_rating_{axis.upper()}"])
                         for row in clip_ledger if row[f"mean_rating_{axis.upper()}"] is not None]) for axis in ["g", "t"]}
    expected = {"clips": {"expected": expected_clips, "observed": len(clips),
                           "matches": len(clips) == expected_clips if expected_clips is not None else None},
                "raters": {"expected": expected_raters, "observed": len(raters),
                            "matches": len(raters) == expected_raters if expected_raters is not None else None},
                "note": "Counts do not identify completely absent clips/raters; no IDs or annotations are invented"}
    return {"schema_version": 1, "status": "COMPUTED_FROM_EXPLICIT_INPUT_NOT_PAPER_REPRODUCED",
            "paper_results_reproduced": False,
            "ground_truth_authority": "User-provided; schema consistency does not establish historical ground-truth provenance",
            "coverage": {"annotation_count": len(annotations), "observed_clip_count": len(clips),
                         "observed_rater_count": len(raters),
                         "missing_annotations_within_observed_clip_rater_matrix": len(clips) * len(raters) - len(annotations),
                         "missing_ratings_by_axis": {axis.upper(): sum(row["ratings"][axis] is None for row in annotations) for axis in AXES},
                         "rater_cross_axis_inconsistent_annotations_retained": sum(row["rater_cross_axis_inconsistency"] for row in annotations),
                         "expected_counts": expected},
            "per_rater": by_rater,
            "unweighted_mean_of_defined_rater_metrics": macro,
            "annotation_pooled_metrics": axis_metrics(annotations),
            "pearson_after_unweighted_clip_mean_ratings": clip_mean_metrics,
            "aggregation_notes": ["Per-rater metric means, annotation pooling and Pearson after averaging G/T by clip are distinct estimands",
                                  "R is categorical: individual correctness is aggregated; numeric R labels are never averaged",
                                  "Missing ratings are absent only from that axis's paired calculation; every annotation stays in the ledger",
                                  "Anonymous rater IDs denote declared groups; independence and original participant identities are not inferred",
                                  "No p-values or confidence intervals are inferred from repeated annotations"]}, clip_ledger


def write_analysis(input_path, output_dir, *, expected_clips=None, expected_raters=None):
    for value in [expected_clips, expected_raters]:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Expected counts must be positive integers")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; existing results are never overwritten")
    annotations, source = load_annotations(input_path)
    summary, clips = analyze(annotations, expected_clips=expected_clips, expected_raters=expected_raters)
    summary["input"] = source
    summary["code_sha256"] = sha256(Path(__file__).read_bytes())
    ledgers = {"annotations.jsonl": annotations, "clips.jsonl": clips}
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, rows in ledgers.items():
        data = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
        artifacts[name] = {"sha256": sha256(data), "bytes": len(data)}
    summary["artifacts"] = artifacts
    with (output_dir / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for name, receipt in artifacts.items():
        if sha256((output_dir / name).read_bytes()) != receipt["sha256"]:
            raise ValueError("Ledger readback hash differs from written output")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Anonymous JSON, JSONL or CSV with explicit GT and rater/clip IDs")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    parser.add_argument("--expected-clips", type=int, help="Optional count check, not a cohort-selection rule")
    parser.add_argument("--expected-raters", type=int, help="Optional count check, not invented identities")
    args = parser.parse_args(argv)
    try:
        summary = write_analysis(args.input, args.output_dir, expected_clips=args.expected_clips,
                                 expected_raters=args.expected_raters)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"],
                      "summary": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

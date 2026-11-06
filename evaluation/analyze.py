"""Recalculate anonymous listening responses without selecting a paper-like subset.

No participant identities, model assignment, trial categories, or exclusions are
inferred from the desired published values. All choices are explicit inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import statistics

CHOICES = {"correct": "correct", "variant1": "variant1", "variant2": "variant2",
           "variant_probe_1": "variant1", "variant_probe_2": "variant2"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
EVALUATION = re.compile(r"evaluation_([1-9][0-9]*)\Z")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    return digest_bytes(Path(path).read_bytes())


def validate_identifier(value, field):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a nonempty anonymous identifier, without spaces or email punctuation")
    return value


def evaluation_id(value):
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return f"evaluation_{value}"
    if isinstance(value, str) and EVALUATION.fullmatch(value):
        return value
    raise ValueError("evaluation_id must be evaluation_N with positive integer N")


def parse_result(value):
    if not isinstance(value, str):
        raise ValueError("result must be a string")
    pieces = value.strip().split(",")
    page = evaluation_id(pieces[0].strip())
    if len(pieces) < 2:
        raise ValueError("Result has no trials")
    trials = []
    for number, piece in enumerate(pieces[1:], 1):
        fields = [field.strip() for field in piece.split("/")]
        if len(fields) != 3 or fields[0] not in CHOICES:
            raise ValueError(f"Trial {number}: expected a legal choice/quality/naturalness triple")
        if not re.fullmatch("[1-5]", fields[1]) or not re.fullmatch("[1-5]", fields[2]):
            raise ValueError(f"Trial {number}: MOS values must be integer strings from1 through5")
        trials.append({"trial_index": number, "choice": CHOICES[fields[0]],
                       "original_choice_token": fields[0], "correct": int(fields[0] == "correct"),
                       "quality": int(fields[1]), "naturalness": int(fields[2])})
    return page, trials


def read_responses(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        raw = list(csv.DictReader(io.StringIO(text)))
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = raw.get("responses")
        if not isinstance(raw, list):
            raise ValueError("JSON input must be a response list or an object with responses:list")
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    allowed = {"response_id", "participant_id", "row_id", "result", "resultstring", "result_string", "evaluation_id"}
    records, seen = [], set()
    for number, row in enumerate(raw, 1):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"Input record {number}: unexpected fields; anonymize the input before analysis")
        fields = [key for key in ("result", "resultstring", "result_string") if key in row]
        if len(fields) != 1:
            raise ValueError(f"Input record {number}: provide exactly one result field")
        page, trials = parse_result(row[fields[0]])
        if row.get("evaluation_id") and evaluation_id(row["evaluation_id"]) != page:
            raise ValueError(f"Input record {number}: result and evaluation_id disagree")
        response = row.get("response_id") or row.get("row_id") or f"response_{number:06d}"
        validate_identifier(response, "response_id")
        if response in seen:
            raise ValueError("Duplicate response_id; preserve separate rows with separate anonymous response IDs")
        seen.add(response)
        participant = row.get("participant_id") or None
        if participant is not None:
            validate_identifier(participant, "participant_id")
        records.append({"response_id": response, "participant_id": participant,
                        "evaluation_id": page, "input_record_index": number,
                        "result_sha256": digest_bytes(row[fields[0]].encode()), "trials": trials})
    if not records:
        raise ValueError("No response records")
    return records


def read_model_map(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema_version", 1) != 1:
        raise ValueError("Model map schema_version must be1")
    mapping = {}
    for row in config.get("ranges", []):
        if not isinstance(row, dict):
            raise ValueError("Each model range must be an object")
        start, end, model = row.get("start"), row.get("end"), row.get("model")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= 100000:
            raise ValueError("Model ranges require positive integer start/end")
        validate_identifier(model, "model name")
        for number in range(start, end + 1):
            page = evaluation_id(number)
            if page in mapping:
                raise ValueError("Overlapping model map ranges")
            mapping[page] = model
    for key, model in config.get("evaluations", {}).items():
        page = evaluation_id(key)
        validate_identifier(model, "model name")
        if page in mapping:
            raise ValueError("Duplicate or overlapping explicit model assignment")
        mapping[page] = model
    if not mapping:
        raise ValueError("Model map is empty; no ID ranges are assumed by this program")
    confirmed = config.get("paper_mapping_confirmed", False)
    if not isinstance(confirmed, bool):
        raise ValueError("paper_mapping_confirmed must be a boolean")
    if confirmed:
        validate_source(config.get("source"))
    return mapping, {"name": config.get("name", "explicit_model_map"),
                     "sha256": file_digest(path), "paper_mapping_confirmed": confirmed,
                     "assignments": mapping}


def validate_source(source):
    if not isinstance(source, dict) or not isinstance(source.get("description"), str) or not source["description"].strip():
        raise ValueError("Mapping source needs a nonempty description")
    if not isinstance(source.get("sha256"), str) or not re.fullmatch("[a-f0-9]{64}", source["sha256"]):
        raise ValueError("Mapping source needs a SHA256 provenance digest")


def read_trial_map(path):
    if path is None:
        return {}, {}, {"provided": False, "figure_categories": "NOT_COMPUTED_NO_TRIAL_MAP"}
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema_version", 1) != 1:
        raise ValueError("Trial map schema_version must be1")
    validate_source(config.get("source"))
    counts = {}
    for key, count in config.get("evaluation_trial_counts", {}).items():
        if type(count) is not int or count < 1:
            raise ValueError("Trial-map page trial counts must be positive integers")
        counts[evaluation_id(key)] = count
    mapping = {}
    for row in config.get("entries", []):
        if not isinstance(row, dict):
            raise ValueError("Each trial-map entry must be an object")
        page = evaluation_id(row.get("evaluation_id"))
        number, dimension, level = row.get("trial_index"), row.get("dimension"), row.get("level")
        if page not in counts:
            raise ValueError("Every mapped page needs an explicit evaluation_trial_counts entry")
        if type(number) is not int or not 1 <= number <= counts[page]:
            raise ValueError("Mapped trial_index falls outside its source page count")
        valid = {"G": range(5), "T": range(1, 6), "R": range(7)}
        if dimension not in valid or type(level) is not int or level not in valid[dimension]:
            raise ValueError("Trial map needs a valid G/T/R dimension and reference category level")
        include = row.get("include_in_figure", True)
        if not isinstance(include, bool) or (not include and not row.get("figure_exclusion_reason")):
            raise ValueError("Explicit figure exclusion requires boolean include_in_figure and a reason")
        key = page, number
        if key in mapping:
            raise ValueError("Duplicate trial map entry")
        mapping[key] = {"dimension": dimension, "level": level, "include_in_figure": include,
                        "figure_exclusion_reason": row.get("figure_exclusion_reason")}
    return mapping, counts, {"provided": True, "sha256": file_digest(path),
                            "source_sha256": config["source"]["sha256"],
                            "mapped_page_count": len(counts), "mapped_trial_definitions": len(mapping),
                            "version_gate": "Response trial count must match mapped source page count"}


def apply_exclusions(records, path):
    raw = [] if path is None else json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("exclusions")
    if not isinstance(raw, list):
        raise ValueError("Exclusions must be a list or an object with exclusions:list")
    selectors = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("Each exclusion must be an object")
        keys = [key for key in ("response_id", "participant_id") if row.get(key)]
        if len(keys) != 1 or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("Every exclusion needs exactly one response_id/participant_id and a reason")
        key = keys[0]
        validate_identifier(row[key], key)
        if not any(record.get(key) == row[key] for record in records):
            raise ValueError("An explicit exclusion matches no input record")
        selectors.append((key, row[key], row["reason"]))
    for record in records:
        reasons = [reason for key, value, reason in selectors if record.get(key) == value]
        record["included"] = not reasons
        record["exclusion_reasons"] = reasons
    return {"provided": path is not None, "sha256": file_digest(path) if path else None,
            "selector_count": len(selectors), "implicit_exclusions": False}


def annotate_trial_mapping(records, mapping, counts, provided):
    """Keep every trial in the ledger, including why it is absent from a figure."""
    for row in records:
        page = row["evaluation_id"]
        for trial in row["trials"]:
            definition = mapping.get((page, trial["trial_index"]))
            if not provided:
                status = "not_provided"
            elif page not in counts:
                status = "unmapped_page"
            elif len(row["trials"]) != counts[page]:
                status = "source_page_trial_count_mismatch"
            elif definition is None:
                status = "unmapped_trial"
            elif not definition["include_in_figure"]:
                status = "explicit_figure_exclusion"
            else:
                status = "mapped"
            trial["figure_mapping_status"] = status
            trial["figure_included"] = row["included"] and status == "mapped"
            if status in {"mapped", "explicit_figure_exclusion"}:
                trial["dimension"] = definition["dimension"]
                trial["level"] = definition["level"]
                trial["figure_exclusion_reason"] = definition["figure_exclusion_reason"]


def describe(values):
    values = list(values)
    n = len(values)
    return {"n": n, "mean": statistics.fmean(values) if n else None,
            "sample_sd_ddof1": statistics.stdev(values) if n > 1 else None,
            "population_sd_ddof0": statistics.pstdev(values) if n else None,
            "sem_from_sample_sd": statistics.stdev(values) / math.sqrt(n) if n > 1 else None}


def summarize(records):
    trials = [trial for row in records for trial in row["trials"]]
    per_response = {field: [statistics.fmean(trial[field] for trial in row["trials"]) for row in records]
                    for field in ("correct", "quality", "naturalness")}
    known = defaultdict(list)
    for row in records:
        if row["participant_id"] is not None:
            known[row["participant_id"]].extend(row["trials"])
    unknown = sum(row["participant_id"] is None for row in records)
    participant_statistics = None
    if known:
        participant_statistics = {field: describe(statistics.fmean(trial[field] for trial in items) for items in known.values())
                                  for field in ("correct", "quality", "naturalness")}
    correct = sum(trial["correct"] for trial in trials)
    return {"response_count": len(records), "trial_count": len(trials),
            "participant_count": len(known) if unknown == 0 else None,
            "participant_identity_status": "DECLARED_COMPLETE" if unknown == 0 else "NOT_ESTABLISHED_FOR_ALL_RESPONSES",
            "responses_without_participant_id": unknown,
            "known_unique_participant_ids": len(known),
            "trial_count_histogram": dict(sorted(Counter(len(row["trials"]) for row in records).items())),
            "accuracy": {"correct": correct, "total": len(trials),
                         "proportion": correct / len(trials) if trials else None,
                         "percent": 100 * correct / len(trials) if trials else None},
            "trial_pooled_statistics": {field: describe(trial[field] for trial in trials)
                                         for field in ("correct", "quality", "naturalness")},
            "per_response_mean_statistics": {field: describe(values) for field, values in per_response.items()},
            "per_declared_participant_mean_statistics": participant_statistics,
            "participant_statistics_scope": "all_declared_participants" if unknown == 0 else "known_ID_subset_only" if known else "unavailable"}


def pearson_table(table):
    if len(table) < 2:
        return {"status": "NOT_COMPUTABLE", "reason": "Need at least two observed categories"}
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return {"status": "NOT_COMPUTED", "reason": "scipy is not installed"}
    try:
        statistic, pvalue, dof, expected = chi2_contingency(table, correction=False)
    except ValueError:
        return {"status": "NOT_COMPUTABLE", "reason": "Contingency table has zero expected frequencies"}
    return {"status": "COMPUTED", "statistic": float(statistic), "p_value": float(pvalue),
            "degrees_of_freedom": int(dof), "yates_correction": False,
            "minimum_expected_frequency": float(expected.min()),
            "expected_cells_below5": int((expected < 5).sum()),
            "expected_cell_count": int(expected.size),
            "method": "Trial-pooled Pearson chi-square, correction=False",
            "assumption_warning": "Within-response/participant dependence is not modelled; this is not a new confirmatory study."}


def figure_statistics(records, trial_map, expected_counts, *, chi_square=True, pairwise=False):
    groups = defaultdict(list)
    coverage, definition_exclusions = Counter(), Counter()
    for row in records:
        page = row["evaluation_id"]
        if page not in expected_counts:
            coverage["unmapped_page_responses"] += 1
            coverage["unmapped_trials"] += len(row["trials"])
            continue
        if len(row["trials"]) != expected_counts[page]:
            coverage["source_page_trial_count_mismatch_responses"] += 1
            coverage["unmapped_trials"] += len(row["trials"])
            continue
        for trial in row["trials"]:
            definition = trial_map.get((page, trial["trial_index"]))
            if definition is None:
                coverage["unmapped_trials"] += 1
                continue
            coverage["mapped_trials"] += 1
            if not definition["include_in_figure"]:
                coverage["explicitly_excluded_from_figure_trials"] += 1
                definition_exclusions[definition["figure_exclusion_reason"]] += 1
                continue
            coverage["figure_trials"] += 1
            groups[row["model"], definition["dimension"], definition["level"]].append((row, trial))
    categories = []
    for (model, dimension, level), items in sorted(groups.items()):
        correct = sum(trial["correct"] for _, trial in items)
        response_values = defaultdict(list)
        for row, trial in items:
            response_values[row["response_id"]].append(trial["correct"])
        categories.append({"model": model, "dimension": dimension, "level": level,
                           "trial_count": len(items), "correct": correct,
                           "accuracy": correct / len(items), "response_count": len(response_values),
                           "per_response_accuracy_statistics": describe(statistics.fmean(values) for values in response_values.values())})
    tests = []
    for model, dimension in sorted({(key[0], key[1]) for key in groups}):
        rows = [row for row in categories if row["model"] == model and row["dimension"] == dimension]
        counts = [[row["correct"], row["trial_count"] - row["correct"]] for row in rows]
        test = {"model": model, "dimension": dimension, "level_order": [row["level"] for row in rows],
                "correct_incorrect_counts": counts,
                "pearson": pearson_table(counts) if chi_square else {"status": "NOT_REQUESTED"}}
        if pairwise and chi_square:
            pairs = list(itertools.combinations(range(len(rows)), 2))
            test["pairwise_bonferroni_family_size"] = len(pairs)
            test["pairwise"] = []
            for left, right in pairs:
                result = pearson_table([counts[left], counts[right]])
                if result["status"] == "COMPUTED":
                    result["bonferroni_adjusted_p_value"] = min(1.0, result["p_value"] * len(pairs))
                test["pairwise"].append({"levels": [rows[left]["level"], rows[right]["level"]], **result})
        tests.append(test)
    total = sum(len(row["trials"]) for row in records)
    return {"status": "MAPPED_DESCRIPTIVE_SUBSET" if categories else "NO_VALIDLY_MAPPED_CATEGORIES",
            "coverage": {**dict(coverage), "included_global_trials": total,
                         "all_included_trials_have_source_mapping": coverage["mapped_trials"] == total},
            "explicit_figure_exclusion_reasons": dict(definition_exclusions),
            "categories": categories, "chi_square_tests": tests,
            "paper_figure1_reproduced": False,
            "note": "Global summaries retain all included responses, even when category mapping is absent or page versions mismatch."}


def analyze(input_path, model_map_path, *, exclusions=None, trial_map=None, chi_square=True, pairwise=False):
    records = read_responses(input_path)
    model_mapping, model_info = read_model_map(model_map_path)
    exclusion_info = apply_exclusions(records, exclusions)
    mapping, counts, trial_info = read_trial_map(trial_map)
    for row in records:
        row["model"] = model_mapping.get(row["evaluation_id"])
        if row["included"] and row["model"] is None:
            raise ValueError("Included evaluation ID has no explicit model assignment; no row was silently discarded")
    annotate_trial_mapping(records, mapping, counts, trial_map is not None)
    included = [row for row in records if row["included"]]
    models = {name: summarize([row for row in included if row["model"] == name])
              for name in sorted({row["model"] for row in included})}
    figure = figure_statistics(included, mapping, counts, chi_square=chi_square, pairwise=pairwise) if trial_map else {
        "status": "NOT_COMPUTED_NO_TRIAL_MAP", "paper_figure1_reproduced": False,
        "categories": [], "chi_square_tests": [], "reason": "Trial identity, dimension and category may not be inferred from response scores or evaluation numbers."}
    summary = {"schema_version": 1, "status": "ANONYMOUS_RESPONSES_RECALCULATED",
               "input_sha256": file_digest(input_path), "analyzer_source_sha256": file_digest(__file__),
               "input_response_count": len(records), "included_response_count": len(included),
               "excluded_response_count": len(records) - len(included),
               "input_trial_count": sum(len(row["trials"]) for row in records),
               "excluded_trial_count": sum(len(row["trials"]) for row in records if not row["included"]),
               "all_rows_accounted_for": True, "global": summarize(included), "models": models,
               "model_mapping": model_info, "exclusions": exclusion_info, "trial_mapping": trial_info,
               "figure1": figure, "accepted_choice_aliases": CHOICES,
               "statistical_units": {"trial_pooled_statistics": "Every retained trial is one value; SD is reported with both ddof0 and ddof1.",
                   "per_response_mean_statistics": "Unweighted statistics of the mean within each retained response.",
                   "per_declared_participant_mean_statistics": "Only caller-supplied participant IDs establish grouping; response IDs are not participant identities.",
                   "sem_from_sample_sd": "SampleSD/sqrt(n) for the named unit, not a confidence interval; independence is not established from anonymous response strings.",
                   "paper_plus_minus": "Paper labels plus/minus as SD but does not report the exact aggregation algorithm; no automatic SD/SEM/CI substitution."},
               "paper_results_reproduced": False,
               "scientific_boundary": "Recovered response records alone do not establish the original participant population, exclusions, full trial versions or the paper's statistical aggregation."}
    return summary, records


def write_outputs(directory, summary, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ledgers = {"included-responses.jsonl": [], "excluded-responses.jsonl": [], "trials.jsonl": []}
    for row in records:
        response = {key: value for key, value in row.items() if key != "trials"}
        response["trial_count"] = len(row["trials"])
        ledgers["included-responses.jsonl" if row["included"] else "excluded-responses.jsonl"].append(response)
        for trial in row["trials"]:
            ledgers["trials.jsonl"].append({"response_id": row["response_id"], "participant_id": row["participant_id"],
                "evaluation_id": row["evaluation_id"], "model": row["model"],
                "included": row["included"], "exclusion_reasons": row["exclusion_reasons"], **trial})
    summary["ledgers"] = {}
    for name, rows in ledgers.items():
        data = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
        path = directory / name
        path.write_bytes(data)
        summary["ledgers"][name] = {"rows": len(rows), "bytes": len(data), "sha256": digest_bytes(data)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Anonymous JSONL, JSON response list, or CSV")
    parser.add_argument("--model-map", required=True, type=Path, help="Explicit evaluation-ID/model assignments")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing results are never overwritten")
    parser.add_argument("--exclusions", type=Path, help="Explicit response/participant exclusions, each with a reason")
    parser.add_argument("--trial-map", type=Path, help="Source-bound trial/category map, including page trial counts")
    parser.add_argument("--chi-square", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairwise", action="store_true", help="Explicitly request all pairwise tests with Bonferroni adjustment")
    args = parser.parse_args(argv)
    try:
        if args.pairwise and not args.chi_square:
            raise ValueError("--pairwise requires --chi-square")
        summary, records = analyze(args.input, args.model_map, exclusions=args.exclusions,
                                  trial_map=args.trial_map, chi_square=args.chi_square, pairwise=args.pairwise)
        write_outputs(args.output_dir, summary, records)
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": summary["status"], "input_response_count": summary["input_response_count"],
                      "included_response_count": summary["included_response_count"],
                      "excluded_response_count": summary["excluded_response_count"],
                      "trial_count": summary["global"]["trial_count"], "paper_results_reproduced": False,
                      "summary": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

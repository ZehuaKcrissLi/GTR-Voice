"""Reproducible extraction, three-axis SVM training, save/reload and evaluation.

This is an executable implementation of section 2.3's feature families. The
paper's exact split, aggregation and SVM hyperparameters were not published;
new measured metrics are never presented as recovered paper results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import sys

from gtr_data import LABEL_VALUES, file_sha256, load_records, make_split, parse_label

PAPER_ACCURACIES = {"G": 0.85, "T": 0.83, "R": 0.94}


def _extract_one(args):
    record, config, aggregation = args
    from feature_extractor import FeatureExtractor
    try:
        path = Path(record["source_path"])
        actual = file_sha256(path)
        if record.get("source_sha256", actual) != actual:
            raise ValueError("Source SHA256 does not match manifest")
        feature = FeatureExtractor(**config).extract_file(path, aggregation)
        return {**record, "source_sha256": actual}, feature, None
    except Exception as error:
        return record, None, {"id": record["id"], "path": record["source_path"],
                              "error": f"{type(error).__name__}: {error}"}


def extract(args):
    import numpy as np
    from feature_extractor import FeatureExtractor

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {output}")
    records = load_records(data_root=args.data_root, manifest=args.manifest)
    constructor = {"sr": args.sample_rate, "n_fft": args.n_fft, "hop_length": args.hop_length,
                   "n_mfcc": args.n_mfcc, "n_mels": args.n_mels}
    tasks = [(record, constructor, args.aggregation) for record in records]
    valid, features, errors = [], [], []
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = pool.map(_extract_one, tasks) if pool else map(_extract_one, tasks)
    try:
        for count, (record, feature, error) in enumerate(results, 1):
            if error:
                errors.append(error)
            else:
                valid.append(record)
                features.append(feature)
            if count % 25 == 0 or count == len(records):
                print(f"Extracted {count}/{len(records)}; rejected {len(errors)}", file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    config = {**FeatureExtractor(**constructor).config, "aggregation": args.aggregation,
              "families": ["MFCC", "mel", "contrast", "centroid", "bandwidth", "rolloff",
                           "minimum_rolloff", "f0", "voiced_flag", "voiced_probability", "zcr", "flatness"]}
    metadata = {"schema_version": 1, "config": config, "input_count": len(records),
                "retained_count": len(valid), "rejected": errors, "on_error": args.on_error,
                "source_audio_modified": False, "records": valid}
    output.parent.mkdir(parents=True, exist_ok=True)
    error_path = output.with_suffix(output.suffix + ".extraction.json")
    error_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if errors and args.on_error == "fail":
        raise ValueError(f"{len(errors)} input errors; see {error_path}. No feature cache written. "
                         "Use an explicit clean manifest or --on-error skip to retain documented exclusions.")
    if not valid:
        raise ValueError("No successfully extracted samples")
    with output.open("xb") as stream:
        np.savez_compressed(stream, features=np.stack(features),
                            metadata=np.asarray(json.dumps(metadata)))
    return {"features": str(output), "sha256": file_sha256(output),
            "shape": [len(valid), len(features[0])], "rejected": errors,
            "source_audio_modified": False}


def load_cache(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].copy()
        metadata = json.loads(str(archive["metadata"].item()))
    records = metadata["records"]
    if features.ndim != 2 or features.shape[0] != len(records) or not np.isfinite(features).all():
        raise ValueError("Invalid/nonfinite feature matrix or manifest length")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate feature-cache record IDs")
    labels = np.asarray([parse_label(row["label"]) for row in records], dtype=int)
    return features, labels, metadata


def metrics(models, x, labels):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    result = {}
    for axis, column in zip("GTR", range(3)):
        prediction = models[axis].predict(x)
        classes = list(LABEL_VALUES[axis])
        result[axis] = {"accuracy": float(accuracy_score(labels[:, column], prediction)),
                        "macro_report": classification_report(labels[:, column], prediction,
                                                               labels=classes, output_dict=True, zero_division=0),
                        "confusion_matrix": confusion_matrix(labels[:, column], prediction, labels=classes).tolist(),
                        "label_order": classes, "predictions": prediction.tolist()}
    return result


def train(args):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    x, labels, metadata = load_cache(args.features)
    train_indices, test_indices, split = make_split(metadata["records"], method=args.split,
        test_size=args.test_size, seed=args.seed, test_sentences=args.test_sentences)
    models = {}
    gamma = args.gamma if args.gamma in {"scale", "auto"} else float(args.gamma)
    for axis, column in zip("GTR", range(3)):
        classifier = SVC(C=args.C, gamma=gamma, kernel=args.kernel)
        models[axis] = make_pipeline(StandardScaler(), classifier) if args.scale else make_pipeline(classifier)
        models[axis].fit(x[train_indices], labels[train_indices, column])
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    bundle = {"schema_version": 1, "models": models, "feature_config": metadata["config"],
              "source_records": [{key: row[key] for key in ("id", "label", "utterance_id", "source_sha256")}
                                 for row in metadata["records"]],
              "extraction_rejected": metadata.get("rejected", []),
              "feature_cache_sha256": file_sha256(args.features), "split": split,
              "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
              "svm_config": {"C": args.C, "gamma": gamma, "kernel": args.kernel, "scale": args.scale},
              "versions": {name: importlib.metadata.version(name) for name in ["numpy", "librosa", "scikit-learn", "joblib"]}}
    model_path = directory / "models.joblib"
    joblib.dump(bundle, model_path)
    reloaded = joblib.load(model_path)
    before = metrics(models, x[test_indices], labels[test_indices])
    after = metrics(reloaded["models"], x[test_indices], labels[test_indices])
    if before != after:
        raise ValueError("Predictions changed after saved-model reload")
    report = {"status": "TRAIN_EVALUATE_SAVE_RELOAD_VERIFIED", "evaluation_role": "held_out_test",
              "metrics": after, "train_samples": len(train_indices), "test_samples": len(test_indices),
              "split": split, "feature_config": metadata["config"], "svm_config": bundle["svm_config"],
              "feature_cache_sha256": bundle["feature_cache_sha256"], "model_sha256": file_sha256(model_path),
              "extraction_rejected": bundle["extraction_rejected"],
              "paper_reported_accuracies": PAPER_ACCURACIES,
              "paper_exact_reproduction": False,
              "paper_comparability": "Original paper split, aggregation and SVM hyperparameters are not reported; these are new measured results.",
              "reload_predictions_equal": True, "source_audio_modified": False}
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate(args):
    import joblib
    import numpy as np

    bundle = joblib.load(args.model)  # Only load trusted, locally generated artifacts.
    x, labels, metadata = load_cache(args.features)
    if metadata["config"] != bundle["feature_config"]:
        raise ValueError("Feature configuration differs from saved model")
    records = metadata["records"]
    if args.all_samples:
        indices = np.arange(len(records))
        role = "all_samples_including_training_not_held_out"
    else:
        if file_sha256(args.features) != bundle["feature_cache_sha256"]:
            raise ValueError("Feature cache bytes differ from the cache bound to the saved split")
        expected = {row["id"]: row["source_sha256"] for row in bundle["source_records"]}
        current = {row["id"]: row["source_sha256"] for row in records}
        if current != expected:
            raise ValueError("Cached source IDs/hashes differ from saved split; refusing a misleading held-out score")
        by_id = {row["id"]: i for i, row in enumerate(records)}
        indices = np.asarray([by_id[sample_id] for sample_id in bundle["split"]["test_ids"]])
        role = "saved_held_out_test"
    return {"status": "EVALUATED_SAVED_MODEL", "evaluation_role": role,
            "samples": len(indices), "metrics": metrics(bundle["models"], x[indices], labels[indices]),
            "model_sha256": file_sha256(args.model), "features_sha256": file_sha256(args.features),
            "paper_exact_reproduction": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="Extract all twelve paper feature families")
    source = ex.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--manifest", type=Path)
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--sample-rate", type=int, default=22050)
    ex.add_argument("--n-fft", type=int, default=2048)
    ex.add_argument("--hop-length", type=int, default=512)
    ex.add_argument("--n-mfcc", type=int, default=40)
    ex.add_argument("--n-mels", type=int, default=128)
    ex.add_argument("--aggregation", choices=["mean", "mean_std"], default="mean")
    ex.add_argument("--workers", type=int, default=1)
    ex.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    ex.set_defaults(func=extract)
    tr = sub.add_parser("train", help="Train and independently reload G/T/R SVMs")
    tr.add_argument("--features", type=Path, required=True)
    tr.add_argument("--output-dir", type=Path, required=True)
    tr.add_argument("--split", choices=["stratified", "sentence", "manifest"], default="stratified")
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--test-sentences", type=int, nargs="+")
    tr.add_argument("--C", type=float, default=0.1)
    tr.add_argument("--gamma", default="0.001")
    tr.add_argument("--kernel", choices=["linear", "poly", "rbf", "sigmoid"], default="rbf")
    tr.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    tr.set_defaults(func=train)
    ev = sub.add_parser("evaluate", help="Evaluate a trusted saved model without retraining")
    ev.add_argument("--features", type=Path, required=True)
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--all-samples", action="store_true")
    ev.set_defaults(func=evaluate)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(args.func(args), indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

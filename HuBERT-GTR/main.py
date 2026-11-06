"""Optional HuBERT GTR experiments; not the paper's reported SVM method."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys

_ACOUSTIC = str(Path(__file__).resolve().parents[1] / "Acoustic-GTR")
if _ACOUSTIC not in sys.path:
    sys.path.append(_ACOUSTIC)
from gtr_data import LABEL_VALUES, check_wav_frames, file_sha256, load_records


def extract_features(model_path, data_path, feature_path, feature_extractor=None, *, manifest=None,
                     on_error="fail", local_files_only=True):
    import librosa
    import numpy as np
    import soundfile as sf
    if feature_extractor is None:
        from transformers import Wav2Vec2FeatureExtractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=local_files_only)
    output = Path(feature_path)
    output.mkdir(parents=True, exist_ok=False)
    records = load_records(data_root=data_path, manifest=manifest)
    saved, errors = [], []
    for row in records:
        try:
            source = Path(row["source_path"])
            actual = file_sha256(source)
            if row.get("source_sha256", actual) != actual:
                raise ValueError("Source SHA256 does not match manifest")
            check_wav_frames(source)
            wave, sr = sf.read(source, dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if sr != 16000:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            values = feature_extractor(wave, sampling_rate=16000, return_tensors="np").input_values[0]
            values = np.asarray(values, dtype=np.float32)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Nonfinite or invalid input_values")
            relative = Path(row["label"]) / (source.stem + ".npy")
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            saved.append({**row, "source_sha256": actual, "feature_path": relative.as_posix(),
                          "feature_sha256": file_sha256(target), "samples_16000": len(values)})
        except Exception as error:
            errors.append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
    report = {"model_path": model_path, "input_count": len(records), "saved": len(saved),
              "rejected": errors, "on_error": on_error, "original_audio_modified": False,
              "purpose": "Optional HuBERT experiment, not paper SVM reproduction"}
    (output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors and on_error == "fail":
        raise ValueError(f"{len(errors)} extraction errors; see extraction.json. No usable index written.")
    if not saved:
        raise ValueError("No extracted examples")
    (output / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in saved))
    return report


def evaluate_model(model, loader, device, task):
    import torch
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, target, mask in loader:
            values = model(inputs.to(device), mask.to(device))
            prediction = values.argmax(-1) if task == "classification" else values.squeeze(-1).round().clamp(0, len(loader.dataset.label_order)-1)
            predictions.extend(prediction.cpu().to(torch.long).tolist())
            labels.extend(target.tolist())
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(loader.dataset.label_order)))).tolist(),
            "predictions": [loader.dataset.label_order[i] for i in predictions],
            "labels": [loader.dataset.label_order[i] for i in labels],
            "label_order": loader.dataset.label_order}


def run_train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dataset import AudioDataset
    from model import GTRClassifier, GTRRegressor
    if args.task == "regression" and args.axis == "R":
        raise ValueError("R is categorical; do not regress an ordering onto resonance")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    common = dict(file_path=args.features, GTR=args.axis, seed=args.seed,
                  split=args.split, test_size=args.test_size, test_sentences=args.test_sentences)
    train_data, test_data = AudioDataset(mode="train", **common), AudioDataset(mode="test", **common)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=AudioDataset.collation_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=AudioDataset.collation_fn)
    kind = GTRClassifier if args.task == "classification" else GTRRegressor
    extra = {"num_classes": len(LABEL_VALUES[args.axis])} if args.task == "classification" else {}
    model = kind(args.model_path, freeze_backbone=args.freeze_backbone,
                 local_files_only=args.local_files_only, **extra).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss() if args.task == "classification" else torch.nn.MSELoss()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    history = []
    for epoch in range(args.epochs):
        model.train(); total_loss = 0.0
        for inputs, labels, mask in train_loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            values = model(inputs, mask)
            loss = criterion(values, labels) if args.task == "classification" else criterion(values.squeeze(-1), labels.float())
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_data)})
        print(json.dumps(history[-1]), file=sys.stderr, flush=True)
    result = evaluate_model(model, test_loader, device, args.task)
    directory = Path(args.output_dir); directory.mkdir(parents=True, exist_ok=False)
    checkpoint = {"schema_version": 1, "axis": args.axis, "task": args.task,
                  "model_path": args.model_path, "freeze_backbone": args.freeze_backbone,
                  "label_order": train_data.label_order, "feature_index_sha256": train_data.index_sha256,
                  "split": train_data.split_record, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
                  "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "epochs": args.epochs, "paper_reported_method": False}
    torch.save(checkpoint, directory / "model.pt")
    loaded = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(loaded["model_state"], strict=True)
    replay = evaluate_model(model, test_loader, device, args.task)
    if replay != result:
        raise ValueError("Saved-model reload changed held-out predictions")
    report = {"status": "TRAIN_SAVE_RELOAD_EVALUATE_VERIFIED", "task": args.task, "axis": args.axis,
              "train_samples": len(train_data), "test_samples": len(test_data), "history": history,
              "metrics": result, "split": train_data.split_record, "checkpoint_sha256": file_sha256(directory / "model.pt"),
              "reload_predictions_equal": True, "paper_reported_method": False,
              "note": "HuBERT is an additional repository experiment, not the 2024 paper SVM method."}
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_checkpoint_split(current, saved):
    """Replay old metadata only after independently verifying source separation."""
    stable_fields = {"method", "seed", "test_size", "test_sentences", "train_ids",
                     "test_ids", "unused_validation_ids", "paper_original_split"}
    allowed = stable_fields | {"source_separation"}
    if not isinstance(saved, dict) or not stable_fields <= saved.keys() or set(saved) - allowed:
        raise ValueError("Saved checkpoint split metadata is incomplete or unrecognized")
    if {key: current[key] for key in stable_fields} != {key: saved[key] for key in stable_fields}:
        raise ValueError("Held-out split differs from saved model")
    guard = current.get("source_separation", {})
    if guard.get("status") != "VERIFIED_NO_SHARED_SOURCE_SHA256":
        raise ValueError("Held-out source separation is UNKNOWN; re-extract an audited cache")
    if "source_separation" in saved and saved["source_separation"] != guard:
        raise ValueError("Source separation differs from saved checkpoint")
    return {"source_separation": guard,
            "checkpoint_split_metadata": "current" if "source_separation" in saved else "legacy_stable_fields_with_recomputed_guard"}


def run_evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    from dataset import AudioDataset
    from model import GTRClassifier, GTRRegressor
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = checkpoint["args"]
    dataset = AudioDataset(args.features, GTR=checkpoint["axis"], mode="test",
        seed=original["seed"], test_size=original["test_size"], split=original["split"],
        test_sentences=original["test_sentences"])
    if dataset.index_sha256 != checkpoint["feature_index_sha256"]:
        raise ValueError("Feature index differs from saved model")
    split_validation = verify_checkpoint_split(dataset.split_record, checkpoint["split"])
    model_path = args.model_path or checkpoint["model_path"]
    kind = GTRClassifier if checkpoint["task"] == "classification" else GTRRegressor
    extra = {"num_classes": len(checkpoint["label_order"])} if checkpoint["task"] == "classification" else {}
    model = kind(model_path, freeze_backbone=checkpoint["freeze_backbone"],
                 local_files_only=args.local_files_only, **extra)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=AudioDataset.collation_fn)
    return {"status": "SAVED_MODEL_EVALUATED", "metrics": evaluate_model(model, loader, device, checkpoint["task"]),
            "checkpoint_sha256": file_sha256(args.checkpoint), "paper_reported_method": False,
            **split_validation}


def add_model_access_args(parser):
    access = parser.add_mutually_exclusive_group()
    access.add_argument("--local-files-only", dest="local_files_only", action="store_true",
                        help="Use local/cached model files only (default)")
    access.add_argument("--allow-download", dest="local_files_only", action="store_false",
                        help="Explicitly allow Hugging Face model downloads")
    parser.set_defaults(local_files_only=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    source = ex.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path);source.add_argument("--manifest", type=Path)
    ex.add_argument("--output-dir", required=True, type=Path);ex.add_argument("--model-path", required=True)
    ex.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    add_model_access_args(ex)
    ex.set_defaults(func=lambda a: extract_features(a.model_path, a.data_root, a.output_dir, manifest=a.manifest,
        on_error=a.on_error, local_files_only=a.local_files_only))
    tr = sub.add_parser("train")
    tr.add_argument("--features", required=True, type=Path);tr.add_argument("--model-path", required=True)
    tr.add_argument("--output-dir", required=True, type=Path);tr.add_argument("--axis", choices=list("GTR"), default="G")
    tr.add_argument("--task", choices=["classification", "regression"], default="classification")
    tr.add_argument("--epochs", type=int, default=20);tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--learning-rate", type=float, default=1e-4);tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--test-size", type=float, default=0.2);tr.add_argument("--split", choices=["stratified", "sentence", "manifest"], default="stratified")
    tr.add_argument("--test-sentences", nargs="+", type=int);tr.add_argument("--device")
    tr.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    add_model_access_args(tr);tr.set_defaults(func=run_train)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--features", type=Path, required=True);ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--model-path");ev.add_argument("--device");ev.add_argument("--batch-size", type=int, default=8)
    add_model_access_args(ev);ev.set_defaults(func=run_evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try: print(json.dumps(args.func(args), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as error:raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

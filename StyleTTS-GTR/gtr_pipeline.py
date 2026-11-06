"""Explicit extract/train/evaluate/infer entry points for StyleTTS-GTR.

No asset downloads, credential prompts, device selection or training at import.
The legacy released embedder and a paper-described derived architecture are
distinct; neither a short run nor automatic metrics reproduce the human study.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def torch_load(path):
    import torch
    # Tensor/primitive-only checkpoints. Never fall back to unrestricted pickle.
    return torch.load(path, map_location="cpu", weights_only=True)


def device_for(name):
    import torch
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable; no fallback was applied")
    return device


def read_manifest(path):
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ("id", "source_path", "source_sha256", "label"):
            if key not in item:
                raise ValueError(f"manifest line {line_number} lacks {key}")
        if item["id"] in seen:
            raise ValueError(f"duplicate sample id: {item['id']}")
        label = str(item["label"])
        if len(label) != 3 or not label.isdecimal():
            raise ValueError("label must preserve three decimal digits, including leading zero")
        g, t, r = map(int, label)
        if not (0 <= g <= 4 and 1 <= t <= 5 and 0 <= r <= 6 and ((g == 0) == (r == 0))):
            raise ValueError(f"illegal GTR label: {label}")
        if item.get("decode_ok") is False or item.get("eligible") is False:
            raise ValueError(f"sample explicitly marked ineligible: {item['id']}")
        seen.add(item["id"])
        records.append(item)
    if not records:
        raise ValueError("manifest is empty")
    return records



def check_source_wav(path):
    # Reuse the repository's audited RIFF/frame validator without loading model
    # packages or making another implementation of anomaly handling.
    import importlib.util
    validator = Path(__file__).resolve().parents[1] / "Acoustic-GTR" / "gtr_data.py"
    spec = importlib.util.spec_from_file_location("_gtr_audio_integrity", validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_wav_frames(Path(path))


def extract(args):
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    import torchaudio
    from models import StyleEncoder
    device = device_for(args.device)
    records = read_manifest(args.manifest)
    teacher = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    state = torch_load(args.checkpoint)
    teacher.load_state_dict(state["net"]["style_encoder"], strict=True)
    teacher.eval().requires_grad_(False).to(device)
    # The official published checkpoint code processes 24kHz waveforms but used
    # torchaudio's implicit 16kHz mel-filter sample_rate. Preserve it explicitly;
    # silently replacing it by 24kHz would change the pretrained feature space.
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=2048,
        win_length=1200, hop_length=300)
    vectors = []
    for item in records:
        source = Path(item["source_path"])
        check_source_wav(source)
        if digest(source) != item["source_sha256"]:
            raise ValueError(f"source SHA256 mismatch: {item['id']}")
        wave, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError(f"expected finite mono source audio: {item['id']}")
        wave = wave[:, 0]
        if sample_rate != 24000:
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=24000)
        wave, _ = librosa.effects.trim(wave, top_db=30)
        if len(wave) < 1025:
            raise ValueError(f"source is too short after trim: {item['id']}")
        with torch.inference_mode():
            # Keep exact CPU waveform/STFT/mel arithmetic for every device.
            # Only the frozen convolutional encoder receives a device tensor.
            mel = transform(torch.from_numpy(wave))
            mel = (torch.log(1e-5 + mel) + 4) / 4
            vector = teacher(mel[None, None].to(device)).squeeze(0).cpu().numpy()
        if vector.shape != (128,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid teacher vector: {item['id']}")
        vectors.append(vector)
        print(json.dumps({"extracted": len(vectors), "total": len(records), "id": item["id"]}), flush=True)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"manifest_sha256": digest(args.manifest),
                "teacher_checkpoint_sha256": digest(args.checkpoint),
                "audio_sample_rate": 24000, "mel_filter_sample_rate": 16000,
                "n_fft": 2048, "win_length": 1200, "hop_length": 300,
                "n_mels": 80, "trim_top_db": 30, "source_bytes_modified": False,
                "mel_device": "cpu", "style_encoder_device": str(device),
                "precision": "float32", "torch_version": str(torch.__version__)}
    with output.open("xb") as stream:
        np.savez_compressed(stream, embeddings=np.stack(vectors).astype("float32"),
                            labels=np.array([int(x["label"]) for x in records], dtype="int64"),
                            ids=np.array([x["id"] for x in records], dtype=str),
                            source_sha256=np.array([x["source_sha256"] for x in records], dtype=str),
                            metadata=np.array(json.dumps(metadata)))
    write_json(str(output) + ".json", dict(metadata, samples=len(records), output_sha256=digest(output)))


def embeddings(path):
    import numpy as np
    import torch
    from gtr_embedder import validate_labels
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("embeddings", "labels", "ids", "source_sha256")}
        result["metadata"] = json.loads(str(data["metadata"]))
    if (result["embeddings"].shape != (len(result["labels"]), 128)
            or not np.isfinite(result["embeddings"]).all()
            or len(set(result["ids"].tolist())) != len(result["ids"])):
        raise ValueError("invalid or duplicate embedding rows")
    validate_labels(torch.as_tensor(result["labels"]))
    if not len(result["labels"]):
        raise ValueError("embedding file is empty")
    return result


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from gtr_embedder import make_embedder
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    device = device_for(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = embeddings(args.embeddings)
    valid = embeddings(args.validation_embeddings) if args.validation_embeddings else None
    if valid:
        if set(data["source_sha256"]) & set(valid["source_sha256"]):
            raise ValueError("training and validation sources overlap")
        if data["metadata"]["teacher_checkpoint_sha256"] != valid["metadata"]["teacher_checkpoint_sha256"]:
            raise ValueError("train/validation teacher checkpoint differs")
    config = {"architecture": args.architecture, "style_dim": args.style_dim, "out_dim": 128}
    if args.architecture == "paper-described":
        config["hidden_dim"] = args.hidden_dim
    model = make_embedder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("output already has last.pt; use a new directory or explicit --resume")
    start_epoch = 0
    if args.resume:
        state = torch_load(args.resume)
        if state["model_config"] != config or state["training_embeddings_sha256"] != digest(args.embeddings):
            raise ValueError("resume configuration or training data differs")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
    tensor_data = TensorDataset(torch.as_tensor(data["labels"]), torch.as_tensor(data["embeddings"]))
    history = []
    for epoch in range(start_epoch, args.epochs):
        loader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=0)
        total, count = 0., 0
        model.train()
        for labels, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(labels.to(device))
            loss = torch.nn.functional.mse_loss(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        row = {"epoch": epoch + 1, "train_mse": total / count}
        model.eval()
        if valid:
            with torch.inference_mode():
                row["validation_mse"] = torch.nn.functional.mse_loss(
                    model(torch.as_tensor(valid["labels"], device=device)),
                    torch.as_tensor(valid["embeddings"], device=device)).item()
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {"format": "gtr-style-embedder-v1", "epoch": epoch + 1,
                 "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "model_config": config,
                 "training_embeddings_sha256": digest(args.embeddings),
                 "training_source_sha256": data["source_sha256"].tolist(),
                 "teacher_feature_metadata": data["metadata"],
                 "teacher_checkpoint_sha256": data["metadata"]["teacher_checkpoint_sha256"],
                 "seed": args.seed, "torch_version": str(torch.__version__)}
        atomic_checkpoint(output / "last.pt", state)
    checkpoint = torch_load(output / "last.pt")
    fresh = make_embedder(checkpoint["model_config"]).to(device)
    fresh.load_state_dict(checkpoint["state_dict"], strict=True)
    fresh.eval()
    with torch.inference_mode():
        labels = torch.as_tensor(data["labels"][:8], device=device)
        if not torch.equal(fresh(labels), model(labels)):
            raise RuntimeError("strict checkpoint replay differs")
    write_json(output / "training.json", {"state": "TRAIN_SAVE_STRICT_RELOAD_VERIFIED", "history": history,
               "epochs_completed": checkpoint["epoch"], "samples": len(data["labels"]),
               "model_config": config, "checkpoint_sha256": digest(output / "last.pt"),
               "teacher_checkpoint_sha256": checkpoint["teacher_checkpoint_sha256"],
               "paper_human_metrics_reproduced": False,
               "architecture_note": "legacy matches released architecture; paper-described widths/joint coding are derived"})


def evaluate(args):
    import torch
    from gtr_embedder import make_embedder
    device = device_for(args.device)
    data = embeddings(args.embeddings)
    checkpoint = torch_load(args.checkpoint)
    if checkpoint["teacher_checkpoint_sha256"] != data["metadata"]["teacher_checkpoint_sha256"]:
        raise ValueError("evaluation teacher differs from training teacher")
    training_sources = checkpoint.get("training_source_sha256")
    overlap = None if training_sources is None else sorted(set(training_sources) & set(data["source_sha256"].tolist()))
    scope = "UNKNOWN_TRAINING_SOURCES" if overlap is None else "CONTAINS_TRAINING_SOURCES" if overlap else "HELD_OUT_SOURCE_HASHES"
    model = make_embedder(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.as_tensor(data["labels"], device=device)).cpu()
        mse = torch.nn.functional.mse_loss(prediction, torch.as_tensor(data["embeddings"])).item()
    write_json(args.output, {"state": "FRESH_PROCESS_STRICT_LOAD_EVALUATED", "samples": len(prediction),
               "training_teacher_feature_metadata": checkpoint.get("teacher_feature_metadata"),
               "evaluation_teacher_feature_metadata": data["metadata"],
               "evaluation_scope": scope, "contains_training_sources": None if overlap is None else bool(overlap),
               "overlapping_source_hash_count": None if overlap is None else len(overlap),
               "mse": mse, "checkpoint_sha256": digest(args.checkpoint),
               "embeddings_sha256": digest(args.embeddings),
               "prediction_float32_sha256": hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
               "paper_human_metrics_reproduced": False})


def infer(args):
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from munch import Munch
    from models import build_model_old
    from gtr_embedder import make_embedder
    from text_utils import symbols
    device = device_for(args.device)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(args.seed)
    config = yaml.safe_load(Path(args.model_config).read_text())
    nets = build_model_old(Munch(config["model_params"]), None, None)
    base = torch_load(args.checkpoint)["net"]
    for key in ("text_encoder", "predictor", "decoder"):
        nets[key].load_state_dict(base[key], strict=True)
        nets[key].eval().requires_grad_(False).to(device)
    embedding_checkpoint = torch_load(args.gtr_checkpoint)
    if embedding_checkpoint["teacher_checkpoint_sha256"] != digest(args.checkpoint):
        raise ValueError("GTR embedder was trained against a different StyleTTS checkpoint")
    embedder = make_embedder(embedding_checkpoint["model_config"]).to(device)
    embedder.load_state_dict(embedding_checkpoint["state_dict"], strict=True)
    embedder.eval()
    phonemes = args.text
    if not args.phonemes:
        from phonemizer.backend import EspeakBackend
        if args.espeakng_loader:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            if not hasattr(EspeakWrapper, 'set_data_path'):
                raise RuntimeError('espeakng-loader requires a phonemizer version with EspeakWrapper.set_data_path; install the pinned root requirements')
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        phonemes = EspeakBackend(language="en-us", preserve_punctuation=True,
                                  with_stress=True).phonemize([args.text])[0]
    symbol_to_id = {char: index for index, char in enumerate(symbols)}
    unknown = sorted(set(phonemes) - set(symbol_to_id))
    if unknown:
        raise ValueError(f"phonemes contain unknown symbols: {unknown}")
    tokens = torch.tensor([[0] + [symbol_to_id[c] for c in phonemes] + [0]], device=device)
    if tokens.max() >= config["model_params"]["n_token"]:
        raise ValueError("text symbol inventory differs from base checkpoint configuration")
    from vocoder.model import Generator
    vocoder_config = json.loads(Path(args.vocoder_config).read_text())
    if vocoder_config.get("num_mels", 80) != 80 or vocoder_config.get("sampling_rate", 24000) != 24000:
        raise ValueError("StyleTTS requires its compatible 80-mel 24kHz vocoder")
    generator = Generator(Munch(vocoder_config)).to(device)
    generator.load_state_dict(torch_load(args.vocoder_checkpoint)["generator"], strict=True)
    generator.eval().requires_grad_(False)
    generator.remove_weight_norm()
    with torch.inference_mode():
        lengths = torch.tensor([tokens.shape[1]], device=device)
        mask = torch.arange(tokens.shape[1], device=device)[None] >= lengths[:, None]
        encoded = nets.text_encoder(tokens, lengths, mask)
        style = embedder(torch.tensor([int(args.gtr)], device=device))
        duration_features = nets.predictor.text_encoder(encoded, style, lengths, mask)
        recurrent, _ = nets.predictor.lstm(duration_features)
        duration = nets.predictor.duration_proj(recurrent).squeeze(0).squeeze(-1).round().clamp(min=1).long()
        total_frames = int(duration.sum())
        if total_frames > args.max_mel_frames:
            raise ValueError("predicted utterance exceeds --max-mel-frames; no truncation applied")
        alignment = torch.repeat_interleave(torch.eye(tokens.shape[1], device=device), duration, dim=1)[None]
        prosody = duration_features.transpose(-1, -2) @ alignment
        f0, noise = nets.predictor.F0Ntrain(prosody, style)
        mel = nets.decoder(encoded @ alignment, f0, noise, style)
        wave = generator(mel).squeeze().cpu().numpy()
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or np.abs(wave).max() <= 0:
        raise RuntimeError("synthesis produced invalid waveform")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wave, 24000, subtype="FLOAT")
    write_json(str(output) + ".json", {"state": "FULL_TEXT_GTR_TO_WAVEFORM_INFERENCE_COMPLETED",
               "output_sha256": digest(output), "pcm_float32_sha256": hashlib.sha256(wave.astype("float32").tobytes()).hexdigest(),
               "samples": len(wave), "sample_rate": 24000, "gtr": args.gtr,
               "text": args.text, "phonemes": phonemes, "base_checkpoint_sha256": digest(args.checkpoint),
               "gtr_checkpoint_sha256": digest(args.gtr_checkpoint),
               "vocoder_checkpoint_sha256": digest(args.vocoder_checkpoint),
               "seed": args.seed, "waveform_peak": float(np.abs(wave).max()),
               "paper_human_metrics_reproduced": False})


def main(default_command=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("extract", help="extract frozen teacher style targets from verified source manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(function=extract)
    p = subs.add_parser("train", help="train only GTR embedder against recorded teacher targets")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--validation-embeddings")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--architecture", choices=["legacy", "paper-described"], default="legacy")
    p.add_argument("--style-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume")
    p.set_defaults(function=train)
    p = subs.add_parser("evaluate", help="strictly load saved embedder in a fresh process")
    for flag in ("--embeddings", "--checkpoint", "--output"):
        p.add_argument(flag, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(function=evaluate)
    p = subs.add_parser("infer", help="generate a real waveform from text and target GTR")
    for flag in ("--checkpoint", "--model-config", "--gtr-checkpoint", "--vocoder-checkpoint", "--vocoder-config", "--text", "--gtr", "--output"):
        p.add_argument(flag, required=True)
    p.add_argument("--phonemes", action="store_true", help="text is already phonemized; do not run espeak")
    p.add_argument("--espeakng-loader", action="store_true", help="explicitly use the separately installed espeakng-loader library/data in this process")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-mel-frames", type=int, default=4000)
    p.set_defaults(function=infer)
    argv = sys.argv[1:]
    if default_command:
        argv = [default_command] + argv
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()

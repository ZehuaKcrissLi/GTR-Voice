"""Explicit optional encoder probe; importing this file never downloads a model."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    access = parser.add_mutually_exclusive_group()
    access.add_argument("--local-files-only", dest="local_files_only", action="store_true",
                        help="Use local/cached model files only (default)")
    access.add_argument("--allow-download", dest="local_files_only", action="store_false",
                        help="Explicitly allow Hugging Face model downloads")
    parser.set_defaults(local_files_only=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import librosa
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    audio, _ = librosa.load(args.wav, sr=16000, mono=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model = HubertModel.from_pretrained(args.model_path, local_files_only=args.local_files_only).to(args.device).eval()
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(args.device)
    with torch.no_grad():
        values = model(inputs).last_hidden_state
    print({"shape": list(values.shape), "finite": bool(torch.isfinite(values).all()), "device": str(values.device)})


if __name__ == "__main__":
    main()

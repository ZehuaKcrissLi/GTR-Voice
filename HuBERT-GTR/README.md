# Optional HuBERT experiments

The Interspeech 2024 paper reports **librosa features + three SVMs**, not HuBERT.
Use `Acoustic-GTR` for that published method. This directory is an additional
experimental classifier/regressor and must not be described as reproducing the
paper's 0.85/0.83/0.94 results.

Use a separate Python 3.10 environment. From the repository root:

```sh
python3.10 -m venv /path/to/gtr-hubert-venv
/path/to/gtr-hubert-venv/bin/python -m pip install -r HuBERT-GTR/requirements.txt
```

This installs the tested acoustic direct dependencies plus PyTorch 2.1.2 and
transformers 4.36.2 (validated with Python 3.10.19). It is not a complete lockfile.
Use that environment's Python for the commands below.

Every CLI and model-loading API uses local/cached files **by default**. Supply a
local model directory; an uncached Hugging Face ID fails without contacting the
Hub. `--local-files-only` remains a compatible explicit spelling of this default.
Only `--allow-download` permits a CLI to download model files; use it deliberately
after choosing the model and storage location. The two flags are mutually
exclusive. No script changes CUDA_VISIBLE_DEVICES or selects GPU2.

```sh
python HuBERT-GTR/main.py extract --manifest /path/to/clean.jsonl \
  --model-path /path/to/chinese-hubert --local-files-only \
  --output-dir /path/to/run/hubert-inputs
python HuBERT-GTR/main.py train --features /path/to/run/hubert-inputs \
  --model-path /path/to/chinese-hubert --local-files-only \
  --axis G --task classification --device cpu --epochs 20 \
  --output-dir /path/to/run/hubert-G
python HuBERT-GTR/main.py evaluate --features /path/to/run/hubert-inputs \
  --checkpoint /path/to/run/hubert-G/model.pt --local-files-only --device cpu
```

Extraction resamples an analysis copy in memory to16kHz, preserves every label
directory and filename, verifies source hashes, rejects incomplete original PCM
frames, and writes an indexed cache. Existing output directories are not
overwritten. Explicit `--on-error skip` retains a rejection report; it never
repairs or modifies original audio.

Classification is the default for all three axes, with global label orders
G0..4, T1..5 and R0..6. Regression is optional for G/T and forbidden for
categorical R. Padded examples carry sample masks through the encoder and masked
time pooling; label offsets never depend on which classes appear in a split.
The backbone is actually frozen and kept in evaluation mode by default.
Hidden width comes from model configuration. Singleton final batches are valid.

Every run stores exact split IDs, feature-index hashes, label order, settings,
model/optimizer state, metrics and a strict reload prediction check. Evaluation
requires the same indexed feature bytes and held-out split. Checkpoints use
`torch.load(weights_only=True)` and should still come from a trusted run.
These engineering checks do not establish useful GTR accuracy or perceptual
control; a real pretrained model and full-data evaluation are separate evidence.

Checkpoints created before the source-hash separation receipt was added can be
evaluated with the same indexed feature bytes and all original stable split
fields unchanged. Evaluation independently recalculates the current SHA256
separation guard and requires verified coverage; missing source hashes or shared
train/test source bytes remain errors. The result labels this metadata-only
compatibility path. Changed test IDs, split settings, feature-index bytes or an
existing guard receipt are never ignored to make an old checkpoint load.

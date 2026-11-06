# Reproducing GTR-Voice research paths

This guide concerns the original Interspeech 2024 paper. It separates executable
code checks, new experiments with stated settings, and recalculation of original
observations. A working optimizer or finite waveform does not establish the
paper's perceptual results.

## Dataset preparation

Download the [original Release ZIP](https://github.com/ZehuaKcrissLi/GTR-Voice/releases/download/gtr-voice-dataset-v1/GTR-Voice.zip),
verify its SHA256 from the root README, and unpack it into your own data directory.
Set `--data-root` to the directory containing label subdirectories such as
`010/01.wav`; keep generated outputs separate from original audio.

```sh
python tools/prepare_dataset.py \
  --data-root /path/to/original/gtr_125 \
  --output-dir /path/to/new-audit \
  --expect-release
```

The tool hashes every source, checks RIFF/PCM structure, decodes all complete
eligible audio, checks the source again, and writes `all.jsonl`, `clean.jsonl`,
`excluded.jsonl` and `audit.json`. The original release gives 2,500 total,
2,486 eligible and 14 excluded recordings; original audio bytes are unchanged.
It assigns no experimental split. Transcripts come from the checked-in twenty
original utterances. These manifests include local source paths and belong in
your private run directory, not in the public repository.

The original collection is mono 48 kHz, with 2,467 PCM16 and 33 PCM24 files. This
is not uniform 24-bit audio as described in the paper. Retain this discrepancy
and every excluded ID in your run record. Resampling/features for a model are
separate derivatives; do not overwrite the distributed originals.

## Acoustic classifiers

See [Acoustic-GTR](../Acoustic-GTR/README.md) for the twelve feature families and
184-dimensional mean-pooled implementation. For a new, explicitly configured
experiment:

```sh
python Acoustic-GTR/run_svm.py extract \
  --manifest /path/to/new-audit/clean.jsonl \
  --output /path/to/new-run/features.npz --workers 2 --on-error fail
python Acoustic-GTR/run_svm.py train \
  --features /path/to/new-run/features.npz \
  --output-dir /path/to/new-run/stratified-seed42 \
  --split stratified --test-size 0.2 --seed 42 \
  --C 0.1 --gamma 0.001 --kernel rbf --scale
python Acoustic-GTR/run_svm.py evaluate \
  --features /path/to/new-run/features.npz \
  --model /path/to/new-run/stratified-seed42/models.joblib
```

Scaling fits only the training fold. Persisted IDs, source hashes, features,
configuration and independently replayed metrics bind the result to its inputs.
The paper reports G/T/R accuracies 0.85/0.83/0.94 but does not specify the original
split, feature pooling or SVM hyperparameters. The settings above define a new
experiment, not a claim to recover those hidden settings. An utterance holdout
is a different experiment and should have its own output directory.

## FastPitch

Use [FastPitch-GTR](../FastPitch-GTR/README.md) for an independent 24 kHz,
100-mel cache, train-only pitch statistics, GTR fine-tuning, resume and inference.
The paper describes 80 epochs of AISHELL3 pretraining and 3,000 epochs of full
GTR fine-tuning (batch 32, Adam, initial LR 0.01, inverse-square-root schedule,
weight decay 1e-6). An English base or randomly initialized engineering fixture
cannot stand in for the Mandarin AISHELL3 base.

A checkpoint must match vocabulary, speaker count, model dimensions and feature
grid. The released 100-mel path requires a matching 24 kHz, 100-band vocoder.
An 80-mel StyleTTS HiFi-GAN is incompatible. Public
[NVIDIA BigVGAN](https://huggingface.co/nvidia/bigvgan_24khz_100band) provides a
compatible feature configuration; it is an explicitly identified upstream
vocoder, not a recovered GTR model. Model files are supplied separately and are
never silently fetched on import.

## StyleTTS

Use [StyleTTS-GTR](../StyleTTS-GTR/README.md) for explicit teacher extraction,
embedder-only training, held-out evaluation and text-to-WAV inference. The base
text/style encoder, predictor and decoder remain frozen. Legacy released
three-embedding/three-FC conditioning and the derived paper-described
one-embedding/two-FC variant have distinct checkpoint formats; they must not be
silently substituted.

The paper says the LibriTTS base was pretrained for 200 epochs across two stages
without stating the allocation, followed by 500 GTR-embedder epochs. The public
upstream `yl4579/StyleTTS` LibriTTS archive at revision
`ad253ae1fa39031d8d2d847c9c5d0fd14dd7d155` instead includes
`epoch_2nd_00050.pth` with config `epochs_1st:80`, `epochs_2nd:50` and no stored
epoch metadata. It is a usable, identified public base, not verified as the
paper's original 200-epoch base. The upstream mel-filter convention is preserved
for checkpoint compatibility and recorded explicitly in extraction metadata.

## Listening-response analysis

See [evaluation](../evaluation/README.md) for strict anonymous input formats,
explicit model/exclusion maps, separate statistical units and trial-map version
gates. Original individual responses, private spreadsheet links and participant
information are not included in this public repository.

Recalculation of all available valid records, with zero implicit exclusions:

| Model assignment | Responses | Correct / trials | Accuracy | Quality mean | Naturalness mean |
|---|---:|---:|---:|---:|---:|
| FastPitch | 32 | 391 / 583 | 67.0669% | 3.04974 | 3.14408 |
| StyleTTS | 30 | 317 / 557 | 56.9120% | 4.19928 | 4.14004 |

These are 62 response records, not 62 verified independent participants. The
explicit legacy compatibility map assigns evaluation IDs 1–20 to FastPitch and
21–50 to StyleTTS; the latter range is not proof of the paper's 20-page cohort.
FastPitch rounds to the reported 67.07%, 3.05 and 3.14. StyleTTS differs from the
paper's 57.14%, 4.21 and 4.16. The paper's original 60-person selection and SD
aggregation are unknown; no score-based filtering is used to force a match.

Table 1 labels its ± values as SD. Trial-pooled SD, response-mean SD and SEM are
separate quantities; this implementation does not relabel one to fit the paper.
Two historical 18-trial pages provide source-bound category mappings covering
only 4 responses / 72 trials. The remaining 58 responses / 1,068 trials lack
verified version-matched category maps. Full Figure 1 and its significance tests
are not reproduced from this incomplete mapping.

The human-label statistics CLI is available for an independently aligned,
anonymized source table:

```sh
python evaluation/analyze_labels.py \
  --input /path/to/anonymous-labels.csv \
  --output-dir /path/to/new-label-analysis \
  --expected-clips 50 --expected-raters 3
```

See the [input schema](../evaluation/README.md) for the eight explicit columns.
Expected counts report coverage; they do not filter or manufacture observations.
Pearson correlation and the separate pooled, rater-mean and clip-mean estimates
are stated implementation choices because the original correlation algorithm
and aggregation details were not specified.

The three-human-rater label verification also remains unverified: the available
label tables have incompatible versions, missing clip/ground-truth alignment and
one out-of-range R=7 value. The tool does not guess a conversion or treat a table
named final as ground truth.

## Run records and limits

For every real run, retain the code revision, environment lock, source hashes,
eligibility/exclusion counts, explicit train/validation/test IDs, model/vocoder
hashes, complete command, terminal status and output hashes. Publish reviewed
aggregates only; keep source-level paths and response/trial ledgers private.

Historical notebooks, absolute-path filelists and upstream utilities are retained
for provenance. The supported paths are the explicit CLIs linked above; this
maintenance does not assert that every historical notebook or optional upstream
training/export/deployment utility is portable or tested.

## macOS packaging note

The official PyPI `torch-2.1.2-cp310-none-macosx_11_0_arm64.whl` has an
inconsistent internal `WHEEL` tag naming x86_64. On the tested ARM64 Mac, the
extension itself is ARM64 and imports/tensor execution work, but current
`pip check`/`uv pip check` report a platform warning. The wheel metadata is not
patched here. Installation, regression tests and actual inference are recorded
separately from that packaging warning; Linux CPU CI checks its own wheel.

## Measured runs on the original released audio (2026-11-06)

All 2,500 WAV members of the public ZIP were byte-matched against the experiment
originals. The 2,486 eligible originals were used without modifying source bytes.
Machine-readable aggregates and artifact hashes are in
[reproduction-results.json](reproduction-results.json).

| New SVM protocol | Train / test | G accuracy | T accuracy | R accuracy |
|---|---:|---:|---:|---:|
| Stratified 80/20, seed 42 | 1988 / 498 | 61.0442% | 53.0120% | 76.7068% |
| Hold out utterances 17–20 | 1990 / 496 | 55.2419% | 52.6210% | 78.6290% |

Both runs use all twelve feature families, 184 mean-pooled features, RBF SVC
C=0.1/gamma=0.001 and a scaler fit only on training data. Independent new-process
evaluation exactly matched saved-model predictions; source hashes were disjoint
across folds. No hyperparameters were tuned to the paper's reported numbers.
These measured accuracies differ from the paper's 85%/83%/94%; they are new
explicit protocols, not a recovered original benchmark configuration.

The identified public StyleTTS base produced targets for 2,239 training and 247
validation recordings, holding out utterances 19–20 across all 125 labels. Both
embedders completed 500 epochs (batch 64, Adam LR 0.001, seed 1234), followed by strict
new-process evaluation and three full text/GTR-to-WAV exports at 24 kHz.

| Derived StyleTTS embedder | Train embedding MSE | Held-out embedding MSE |
|---|---:|---:|
| legacy | 0.00464641 | 0.00555047 |
| paper-described | 0.00474630 | 0.00560335 |

The paper-described variant uses a derived 125×32 joint embedding and 32→64→128
Linear/ReLU/Linear mapping; the paper did not report these dimensions. The
legacy variant preserves released checkpoint keys and architecture. Both use
the public base identified above, which differs from the paper's base-training
description. Embedding MSE and finite, non-silent decoded audio are engineering
observations, not human MOS or proof of perceptual G/T/R control.

FastPitch's complete optimizer/checkpoint/resume/inference engineering path
passed. Three actual original Mandarin recordings also passed independent 24 kHz
100-mel/pitch preparation, source-hash preservation, Mandarin phoneme encoding,
real dataset loading and collation. A real-source BigVGAN reconstruction passed.
The original compatible AISHELL3/GTR checkpoint has not been acquired and
validated, so 80+3000 epoch paper training and paper-quality conditioned FastPitch
synthesis remain unverified. No different TTS architecture or random model is
substituted for that missing asset.

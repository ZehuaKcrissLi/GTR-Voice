# Acoustic GTR label verification

This path implements the twelve librosa feature families named in section 2.3 of
the Interspeech 2024 paper. It trains **three classifiers**, for G, T and R, and
saves a single artifact containing the models, feature configuration, source
hashes and exact split. `main.py` is an alias for `run_svm.py`.

The paper reports G/T/R accuracy **0.85 / 0.83 / 0.94**, but does not specify its
split, temporal aggregation or SVM hyperparameters. The commands below define a
new reproducible run; its measured metrics are not the recovered paper results.
HuBERT is a separate exploratory directory and is not part of this paper's
reported classification method.

## Requirements

Python >=3.10 with `numpy`, `scipy`, `librosa` (>=0.10), `soundfile`,
`scikit-learn` and `joblib`. No PyTorch, GPU or model download is needed here.
Use an isolated environment; keep feature caches outside the checkout.

From the repository root, create a Python 3.10 environment and install the tested
direct dependencies:

```sh
python3.10 -m venv /path/to/gtr-venv
/path/to/gtr-venv/bin/python -m pip install -r Acoustic-GTR/requirements-tested.txt
```

The tested environment uses Python 3.10.19, NumPy 1.26.4, SciPy 1.14.1,
librosa 0.10.2.post1, soundfile 0.12.1, scikit-learn 1.5.2 and joblib 1.4.2.
`requirements.txt` gives broader compatibility ranges; `requirements-tested.txt`
pins these direct dependencies, without claiming a full transitive lockfile.
Use that environment's Python for the commands below.

## Run

```sh
python Acoustic-GTR/run_svm.py extract \
  --data-root /path/to/gtr_125 --output /path/to/run/features.npz \
  --workers 4 --on-error fail
python Acoustic-GTR/run_svm.py train \
  --features /path/to/run/features.npz --output-dir /path/to/run/svm \
  --split stratified --test-size 0.2 --seed 42
python Acoustic-GTR/run_svm.py evaluate \
  --features /path/to/run/features.npz --model /path/to/run/svm/models.joblib
```

Instead of `--data-root`, use `--manifest /path/to/clips.jsonl`. Each line needs
`source_path` and a string `label` such as `"010"`; `id`, `utterance_id` and
`source_sha256` are supported. A relative source path is relative to the manifest.
Labels follow G=0..4, T=1..5, R=0..6; G=0 iff R=0. Leading zeros are preserved.
Source hashes, where supplied, must match before extraction.

The original archive contains fourteen malformed trailing PCM-frame cases.
Extraction rejects those files without changing them. A pre-audited clean
manifest is recommended. To explicitly retain a partial run, choose
`--on-error skip`; every rejected sample and reason is written to
`features.npz.extraction.json` and carried into the training results. No original
is repaired, transcoded or deleted.

Defaults are explicit implementation choices: 22,050Hz analysis copy in memory,
40 MFCCs, 128 mel bins, frame2048/hop512, pYIN C2–C7 (unvoiced F0=0), mean pooling,
train-only StandardScaler, RBF SVC C=0.1/gamma=0.001. `--aggregation mean_std`
adds per-feature standard deviation; `--no-scale`, `--C` and `--gamma` are
available. These options must be reported, not tuned against the final test set.

The default deterministic 80/20 split is stratified by complete GTR label and
shared by all three axes. `--split sentence --test-sentences 17 18 19 20` holds
out those sentence IDs across labels. `--split manifest` uses explicit train/test
records and reports unused validation IDs. The program never silently adopts a
manifest's split. The paper's original split remains unknown.

Training writes `models.joblib` and `results.json` to a **new** output directory.
It reloads the saved artifact and requires identical held-out predictions.
`evaluate` checks the cached IDs/hashes and configuration against the saved
split. Only load trusted joblib files. `--all-samples` is explicitly labeled as
including training samples and must not be described as held-out performance.

Train/test partitions are checked for shared original-source SHA256 values, even
when copied WAVs have different paths or IDs. A cross-partition match is an
error. The split receipt records source-hash coverage; missing hashes are
`UNKNOWN`, and SVM training refuses such a cache before fitting. Re-extract an
audited cache rather than claiming verified separation from paths alone. The
check uses hashes bound into the extraction cache; it does not prove an
unpublished paper split or reread every original during training.


## Measured full-data validation (2026-11-06)

A run on the 2,486 valid original clips completed all twelve feature families
with 184 mean-pooled features per clip and no further extraction rejections.
The fourteen previously identified malformed originals were explicitly excluded
by the clean manifest and left unchanged. Both protocols below reuse the exact
same feature bytes and C=0.1/gamma=0.001/RBF/train-only StandardScaler settings;
neither protocol was tuned to the paper's results.

| New measured protocol | Train / test clips | G accuracy | T accuracy | R accuracy |
| --- | --- | --- | --- | --- |
| GTR-label-stratified 80/20, seed42 | 1988 / 498 | 304/498 (61.04%) | 264/498 (53.01%) | 382/498 (76.71%) |
| Sentence IDs17–20 held out | 1990 / 496 | 274/496 (55.24%) | 261/496 (52.62%) | 390/496 (78.63%) |

For both runs, saved-model reload and a separate-process evaluation produced
identical predictions, scalers matched training rows only, and source SHA256
coverage was complete with no shared train/test hash. The feature-cache SHA256
was `403eab5816248950aa272b0bd30436328d86db645023c1d0a19e0f316319269e`.
These are new, single-speaker validation results and differ from the paper's
reported accuracies. The unpublished original split, aggregation and complete
SVM setup remain unknown; this is not an exact accuracy reproduction claim.

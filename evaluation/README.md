# Anonymous listening-response statistics

This CLI recalculates original-format response strings. It does **not** create
new human listeners, infer a 60-person cohort, select a subset that matches the
paper, or turn a recovered-data calculation into a reproduced user study.

The core CLI needs Python>=3.10 and only the standard library. Optional category
chi-square calculations need SciPy (`evaluation/requirements.txt`).

For optional chi-square calculations, install into your isolated environment
from the repository root with
`python -m pip install -r evaluation/requirements.txt`. Parsing, MOS and ledgers
work without installing that file. The validated environment used SciPy 1.14.1;
this directory does not need PyTorch, model weights or a GPU.

```sh
python evaluation/analyze.py \
  --input /path/to/responses-anonymous.jsonl \
  --model-map evaluation/examples/model-map-legacy.json \
  --output-dir /path/to/new-analysis
python -m unittest discover -s evaluation/tests -p 'test_*.py' -v
```

## Inputs and identity

JSONL, a JSON list (or `{"responses":[...]}`), and CSV are supported. Example
**synthetic** row:

```json
{"response_id":"r001","result":"evaluation_1,correct/4/5,variant1/3/4"}
```

`participant_id` is optional. `response_id` represents a response record, not a
known independent person. If absent, a stable local response ID is generated;
participant identity is never invented. With no identity grouping, the report's
`participant_count` is null. Repeated explicit participant IDs are grouped for
per-participant means while every response and trial remains in global counts.
With incomplete identity coverage, participant statistics are clearly marked as
covering the known-ID subset only.

`resultstring` / `result_string` aliases and optional `row_id` are accepted.
Extra identity/demographic columns are rejected: anonymize the export first.
Never commit raw survey files, names, emails, original spreadsheet links or
response-level ledgers to this public repository.

Strings use `evaluation_N,choice/quality/naturalness,...`. Legal original choices
are `correct`, `variant1`, `variant2`; explicit compatible aliases
`variant_probe_1` / `variant_probe_2` are recorded and normalized. Both MOS fields
must be integer strings1–5. Variable response lengths, including18,19 or20trials,
are preserved. There is no ID40 cutoff, outlier filter or automatic deduplication.
Duplicate response IDs or malformed records cause a clear error, not silent loss.

## Explicit model mapping and exclusions

`--model-map` is required. It accepts either/both nonoverlapping explicit
assignments and ranges:

```json
{"schema_version":1,"name":"my_source_bound_map","paper_mapping_confirmed":false,
 "evaluations":{"evaluation_1":"FastPitch","evaluation_22":"StyleTTS"}}
```

The checked-in legacy example explicitly assigns IDs1–20 toFastPitch and21–50
toStyleTTS. This is compatibility configuration, **not confirmation** that later
IDs were part of the paper's20-pages-per-model study. An included unmapped ID is
an error. Report the map SHA256 with the results.

Optional `--exclusions exclusions.json` requires a reason for every selector:

```json
[{"response_id":"r002","reason":"Documented exclusion from an independent source"}]
```

Selectors may instead use an explicit `participant_id`. Every selector must match
an input record. All rows are parsed first; no excluded record disappears from
the ledger. Do not invent exclusions to make accuracy or MOS match a paper value.

## Statistical units

Each model reports response/trial counts, established participant counts (if
available), correct/total accuracy and the following **separate** statistics:

- Trial-pooled quality, naturalness and correctness: mean, populationSD(ddof0),
  sampleSD(ddof1), and sampleSD/sqrt(n).
- Unweighted statistics over each response's mean, with their ownSD andSEM.
- Unweighted statistics over declared participants' means, when identities exist.

SEM is not a confidence interval. Anonymous records do not establish independence.
The paper labels its ± values as standard deviations but does not report the
aggregation algorithm. The tool never relabels SEM/CI as the paper'sSD, chooses a
formula because it matches, or marks `paper_results_reproduced` true.

## Source-bound trial maps and Figure1

Without `--trial-map`, no G/T/R category graph or category chi-square is computed.
Do not infer categories from evaluation ID, response scores or desired p-values.
A map must have provenance and source-page trial counts:

```json
{"schema_version":1,
 "source":{"description":"Synthetic example; replace with verified provenance","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
 "evaluation_trial_counts":{"evaluation_1":2},
 "entries":[
  {"evaluation_id":"evaluation_1","trial_index":1,"dimension":"G","level":1},
  {"evaluation_id":"evaluation_1","trial_index":2,"dimension":"G","level":4}]}
```

Trial indices are one-based. A response must match the source page's trial count;
an old20-trial response cannot silently use an18-trial version's first18labels.
Unmapped/mismatched records stay in global statistics. Category outputs identify
their exact coverage; a partial map is never called the complete Figure1.

An explicitly sourced figure-only exclusion uses `include_in_figure:false` and
`figure_exclusion_reason`; it does not remove the response from global statistics.
For example, a known whisper exclusion must be specified from its source, not
guessed by this tool.

Mapped groups optionally receive trial-pooled Pearson chi-square tests of
correct/incorrect counts across category levels, with `correction=False` and
small-expected-cell diagnostics. `--pairwise` explicitly requests every pair in
the family with Bonferroni-adjusted p-values. These descriptive tests do not model
within-response/participant dependence and do not establish new confirmatory
evidence or recover the paper's unknown test options. `--no-chi-square` disables
tests while retaining group counts.

## Outputs

The output directory must be new. It contains `summary.json`,
`included-responses.jsonl`, `excluded-responses.jsonl`, and `trials.jsonl`.
All input rows/trials are accounted for; result strings are bound by SHA256.
Trial ledgers also record mapping/mismatch/figure-exclusion status. Summary binds
input, model-map, exclusion and trial-map hashes plus ledger hashes and analyzer
code SHA. Public publication should use reviewed aggregates, not individual rows.

Tests use synthetic strings and a local CLI subprocess to verify formulas,
input rejection, no silent filtering, explicit exclusions, identity units,
mapping-version gates and chi-square counts. They are not scientific study data.


## Human label verification (paper section 2.3)

`analyze_labels.py` computes G/T Pearson correlation and categorical R accuracy
from **explicitly aligned**, user-provided ground truth and anonymous rater/clip
IDs. It uses only Python >=3.10's standard library. The original three-rater,
50-clip input has not been established; runnable code does not reproduce that
cohort or its reported 0.75 / 0.80 / 0.78 results. The paper does not specify
the correlation type or full aggregation formula: Pearson and the distinct
estimands below are explicit implementation choices, not a recovered setup.

```sh
python evaluation/analyze_labels.py --input /path/to/anonymous-labels.csv \
  --output-dir /path/to/new-label-analysis --expected-clips 50 --expected-raters 3
```

The expected counts are optional diagnostics. They never select, generate or
pad a cohort. All actual rows remain in the annotation ledger.

JSON lists, JSONL and CSV use exactly these eight fields:

| Field | Meaning |
| --- | --- |
| `clip_id` | Explicit anonymous clip identifier, e.g. `010/01.wav` |
| `rater_id` | Declared anonymous rater group, e.g. `r01` |
| `gt_g`, `gt_t`, `gt_r` | Complete user-provided ground-truth integers |
| `rating_g`, `rating_t`, `rating_r` | Independent per-axis ratings; JSON null / blank CSV cell means missing |

A JSON object wrapper must be `{"schema_version":1,"ratings":[...]}`. Identifiers
use ASCII letters/numbers and `._/-`, beginning with a letter/number, at most128
characters. The caller must anonymize IDs before export: this syntax check is
not an anonymizer and cannot recognize a real name that resembles a valid ID.
Use pseudonyms, not names or contact details. Extra demographic or identity
columns are rejected. Example **synthetic schema row** (one row alone
cannot define Pearson correlation):

```json
{"clip_id":"c001","rater_id":"r01","gt_g":1,"gt_t":2,"gt_r":3,"rating_g":1,"rating_t":2,"rating_r":3}
```

G is0..4, T is1..5 and R is0..6. Ground truth must belong to the125 legal GTR
combinations (G=0 iff R=0) and be identical for a clip across all raters. Rater
axes are independent judgments: a valid-range G=0/R=3 observation is retained
with a diagnostic. R=7, noninteger/mixed scales, missing or conflicting GT, and
duplicate `(clip_id,rater_id)` pairs or duplicate JSON field names cause errors. No filename/row-order GT,
scale conversion, imputation, outlier removal or automatic exclusions occur.

`summary.json` distinguishes:

- Each rater's G/T Pearson, R correct/paired accuracy, paired counts and missing
  ratings, including clip IDs absent from the observed clip/rater matrix.
- The unweighted mean of **defined** per-rater metrics, with undefined raters
  listed. This is separate from annotation-pooled metrics.
- Pearson after first taking each clip's unweighted mean G/T rating. Each clip
  contributes once; variable available-rater counts appear in `clips.jsonl`.
  R labels are categorical and are never numerically averaged.

Pearson is null with an explicit reason for fewer than two pairs or a constant
GT/rating vector. Missing ratings affect only that axis's paired calculation.
Expected counts cannot identify completely absent IDs; the report says so.
Anonymous rater groups do not independently prove participant identity or
statistical independence. No p-values or confidence intervals are inferred.

Outputs require a new directory and include source/code SHA256, a full
`annotations.jsonl` ledger and per-clip `clips.jsonl`; both ledger hashes are
read back and recorded. Keep original label tables and response-level ledgers
outside this public repository. The tests use synthetic aligned annotations
solely to check formulas, missingness and input boundaries.

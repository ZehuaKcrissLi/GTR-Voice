#!/usr/bin/env bash
# Full GTR fine tuning; paths and pitch statistics are explicit inputs.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON:=python}"
: "${INIT_CHECKPOINT:?set INIT_CHECKPOINT to the compatible AISHELL3 base checkpoint}"
: "${DATASET_PATH:?set DATASET_PATH to an independent prepared derivative directory}"
: "${TRAIN_FILELIST:?set TRAIN_FILELIST}"
: "${VAL_FILELIST:?set VAL_FILELIST}"
: "${OUTPUT_DIR:?set a dedicated OUTPUT_DIR}"
: "${PITCH_MEAN:?use pitch_training_voiced_mean from preparation.json}"
: "${PITCH_STD:?use pitch_training_voiced_std from preparation.json}"
: "${DEVICE:=cuda}"
args=(--init-from-checkpoint "$INIT_CHECKPOINT" --gtr-emb
 --dataset-path "$DATASET_PATH" --training-files "$TRAIN_FILELIST"
 --validation-files "$VAL_FILELIST" --output "$OUTPUT_DIR"
 --symbol-set english_mandarin_basic --text-cleaners english_cleaners_v2
 --p-arpabet 0 --n-speakers "${NSPEAKERS:-200}"
 --epochs "${EPOCHS:-3000}" --batch-size "${BATCH_SIZE:-32}"
 --grad-accumulation "${GRAD_ACCUMULATION:-1}" --optimizer adam
 --learning-rate "${LEARNING_RATE:-0.01}" --weight-decay 1e-6
 --warmup-steps "${WARMUP_STEPS:-1000}" --trainloader-repeats 1
 --epochs-per-checkpoint "${EPOCHS_PER_CHECKPOINT:-100}" --validation-freq 1
 --num-workers "${NUM_WORKERS:-2}" --seed "${SEED:-1234}"
 --sampling-rate 24000 --filter-length 1024 --win-length 1024 --hop-length 256
 --n-mel-channels 100 --mel-fmin 0 --mel-fmax 12000
 --pitch-mean "$PITCH_MEAN" --pitch-std "$PITCH_STD"
 --load-mel-from-disk --load-pitch-from-disk --energy-conditioning)
case "$DEVICE" in cuda) args+=(--cuda);; cpu) ;; *) echo 'DEVICE must be cpu or cuda' >&2; exit 2;; esac
exec "$PYTHON" train.py "${args[@]}" "$@"

#!/usr/bin/env bash

set -a

PYTHONIOENCODING=utf-8

# Mandarin & English bilingual
ARGS+=" --symbol-set english_mandarin_basic --no-gtr-emb"

# Initialize weights with a pre-trained English model
#bash scripts/download_models.sh fastpitch
#ARGS+=" --init-from-checkpoint pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"
#ARGS+=" --sampling_rate=24000"
#ARGS+=" --hop_length=300"
#ARGS+=" --win_length=1200"
#ARGS+=" --filter_length=2048"
ARGS+=" --pitch-mean 196.3935374639679"
ARGS+=" --pitch-std 54.26165097828614"

AMP=false  # FP32 training for better stability

: ${DATASET_PATH:=data/AISHELL3}
: ${TRAIN_FILELIST:=filelists/aishell3_train.txt}
: ${VAL_FILELIST:=filelists/aishell3_val.txt}
: ${OUTPUT_DIR:=./output_aishell3_bigv}
: ${NSPEAKERS:=200}
: ${EPOCHS:=80}
NUM_GPUS=1 GRAD_ACCUMULATION=2 BATCH_SIZE=16 bash scripts/train.sh $ARGS "$@"

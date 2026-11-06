#!/usr/bin/env bash
# HiFi-GAN synthesis requires a checkpoint/config compatible with the FP mel grid.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON:=python}"
: "${FASTPITCH:?set FASTPITCH to the GTR checkpoint}"
: "${HIFIGAN:?set HIFIGAN to the compatible vocoder checkpoint}"
: "${HIFIGAN_CONFIG:?set HIFIGAN_CONFIG}"
: "${FILELIST:?set FILELIST to a text or TSV input}"
: "${OUTPUT_DIR:?set a dedicated OUTPUT_DIR}"
: "${DEVICE:=cuda}"
args=(--input "$FILELIST" --output "$OUTPUT_DIR" --fastpitch "$FASTPITCH"
 --hifigan "$HIFIGAN" --hifigan-config "$HIFIGAN_CONFIG"
 --batch-size "${BATCH_SIZE:-1}" --affinity disabled
 --sampling-rate "${SAMPLING_RATE:-24000}" --hop-length "${HOP_LENGTH:-256}"
 --symbol-set english_mandarin_basic --text-cleaners english_cleaners_v2 --p-arpabet 0
 --speaker "${SPEAKER:-199}" --g "${G:-2}" --t "${T:-3}" --r "${R:-1}")
case "$DEVICE" in cuda) args+=(--cuda);; cpu) ;; *) echo 'DEVICE must be cpu or cuda' >&2; exit 2;; esac
exec "$PYTHON" inference.py "${args[@]}" "$@"

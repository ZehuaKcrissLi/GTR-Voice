#!/usr/bin/env bash
# Local FastPitch checkpoint -> exported mel -> explicit local BigVGAN weights.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON:=python}"
: "${FASTPITCH:?set FASTPITCH to the GTR checkpoint}"
: "${FILELIST:?set FILELIST}"
: "${OUTPUT_DIR:?set a dedicated OUTPUT_DIR}"
: "${BIGVGAN_ROOT:?set BIGVGAN_ROOT to a verified compatible local source directory}"
: "${BIGVGAN_CHECKPOINT:?set BIGVGAN_CHECKPOINT to the compatible 24kHz/100-mel model}"
: "${BIGVGAN_CONFIG:=$BIGVGAN_ROOT/config.json}"
: "${DEVICE:=cuda}"
[ -f "$BIGVGAN_ROOT/bigvgan.py" ] || { echo 'Missing BigVGAN source; no code or weights are downloaded automatically' >&2; exit 2; }
args=(--input "$FILELIST" --output "$OUTPUT_DIR/mels" --fastpitch "$FASTPITCH"
 --save-mels --affinity disabled --batch-size "${BATCH_SIZE:-1}"
 --symbol-set english_mandarin_basic --text-cleaners english_cleaners_v2 --p-arpabet 0
 --sampling-rate 24000 --hop-length 256 --speaker "${SPEAKER:-199}"
 --g "${G:-2}" --t "${T:-3}" --r "${R:-1}")
case "$DEVICE" in cuda) args+=(--cuda);; cpu) ;; *) echo 'DEVICE must be cpu or cuda' >&2; exit 2;; esac
"$PYTHON" inference_bigv.py "${args[@]}" "$@"
shopt -s nullglob
mels=("$OUTPUT_DIR/mels/"*.npy)
[ "${#mels[@]}" -gt 0 ] || { echo 'No FastPitch mels were exported' >&2; exit 2; }
for mel in "${mels[@]}"; do
  name="$(basename "${mel%.npy}")"
  "$PYTHON" vocode_bigvgan.py --source-dir "$BIGVGAN_ROOT" --config "$BIGVGAN_CONFIG" \
   --checkpoint "$BIGVGAN_CHECKPOINT" --mel "$mel" --output "$OUTPUT_DIR/audio/$name.wav" --device "$DEVICE"
done

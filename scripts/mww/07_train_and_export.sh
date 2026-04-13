#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

cd "${MWW_DIR}"

mkdir -p "${OUTPUT_DIR}"

# Link data directories into micro-wake-word working tree so config paths stay simple.
ln -sfn "${DATA_DIR}/generated_augmented_features" "${MWW_DIR}/generated_augmented_features"
ln -sfn "${DATA_DIR}/negative_datasets" "${MWW_DIR}/negative_datasets"

python3 ${SCRIPT_DIR}/06_write_training_config.py \
  --steps "${TRAIN_STEPS}" \
  --output "${MWW_DIR}/training_parameters.yaml" \
  --train-dir "trained_models/${KEYWORD_ID}"

python3 -m microwakeword.model_train_eval \
  --training_config='training_parameters.yaml' \
  --train 1 \
  --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 \
  --test_tflite_nonstreaming 0 \
  --test_tflite_nonstreaming_quantized 0 \
  --test_tflite_streaming 0 \
  --test_tflite_streaming_quantized 1 \
  --use_weights "best_weights" \
  mixednet \
  --pointwise_filters "64,64,64,64" \
  --repeat_in_block "1,1,1,1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 32 \
  --first_conv_kernel_size 5 \
  --stride 3

TFLITE_PATH="${MWW_DIR}/trained_models/${KEYWORD_ID}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
cp -f "${TFLITE_PATH}" "${OUTPUT_DIR}/${KEYWORD_ID}.tflite"

cat > "${OUTPUT_DIR}/${KEYWORD_ID}.json" <<EOF
{
  "type": "micro",
  "wake_word": "${KEYWORD_PHRASE}",
  "author": "local-training",
  "website": "https://github.com/OHF-Voice/micro-wake-word",
  "version": 1,
  "model": "${KEYWORD_ID}.tflite",
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "minimum_esphome_version": "2024.7.0"
  }
}
EOF

log "Exported:"
log " - ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
log " - ${OUTPUT_DIR}/${KEYWORD_ID}.json"

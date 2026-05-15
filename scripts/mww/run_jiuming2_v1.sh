#!/usr/bin/env bash
# 救命救命 v1 MWW 训练（wakeword-mww Docker 内）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs"
MWW_DIR="/workspace/work/micro-wake-word"
KEYWORD_ID="jiuming2_v1"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 救命救命 v1 MWW 训练 ==="

${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh
${SCRIPT_DIR}/02_download_datasets.sh
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# 对抗性负样本 mmap
ADV_MMAP="${DATA_DIR}/negative_datasets/zh_jiuming2_adv/zh_jiuming2_adv/training/zh_jiuming2_adv_mmap"
if [ -d "$ADV_MMAP" ]; then
    log "Adversarial mmap exists, skip"
else
    log "Generating adversarial mmap..."
    python3 ${SCRIPT_DIR}/generate_adversarial_features.py \
        --input-dir /workspace/outputs/jiuming2_adversarial_merged \
        --output-dir ${DATA_DIR}/negative_datasets/zh_jiuming2_adv/zh_jiuming2_adv \
        --data-dir ${DATA_DIR}
fi

# 正样本特征
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
log "Generating positive features..."
rm -rf "${FEATURES_DIR}"
python3 ${SCRIPT_DIR}/03_generate_features_real.py \
    --positive-dir "${DATA_DIR}/positive_augmented/jiuming2_v1" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURES_DIR}"

# 训练配置
log "Writing training config..."
python3 ${SCRIPT_DIR}/06_write_training_config_jiuming2.py \
    --steps 30000 \
    --train-dir "trained_models/${KEYWORD_ID}" \
    --output "${MWW_DIR}/training_parameters.yaml"

# 训练
log "Training..."
cd "${MWW_DIR}"
ln -sfn "${DATA_DIR}/generated_augmented_features" "${MWW_DIR}/generated_augmented_features"
ln -sfn "${DATA_DIR}/negative_datasets" "${MWW_DIR}/negative_datasets"

python3 -u -m microwakeword.model_train_eval \
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

# 导出
log "Exporting..."
TFLITE_SRC="${MWW_DIR}/trained_models/${KEYWORD_ID}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
if [ -f "$TFLITE_SRC" ]; then
    cp -f "${TFLITE_SRC}" "${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
    cat > "${OUTPUT_DIR}/${KEYWORD_ID}.json" <<EOF
{
  "type": "micro",
  "wake_word": "救命救命",
  "author": "local-training",
  "website": "https://github.com/OHF-Voice/micro-wake-word",
  "version": 1,
  "model": "${KEYWORD_ID}.tflite",
  "micro": {
    "probability_cutoff": 0.80,
    "sliding_window_size": 3,
    "minimum_esphome_version": "2024.7.0"
  }
}
EOF
    log "Exported: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
else
    log "WARNING: TFLite not found"
    find "${MWW_DIR}/trained_models/${KEYWORD_ID}" -name "*.tflite" 2>/dev/null
fi

# 验证
TEST_DIR="/workspace/outputs/jiuming2_edgetts/test"
if [ -d "$TEST_DIR" ] && [ -f "$TEST_DIR/manifest.json" ]; then
    log "Verifying on test set..."
    python3 /workspace/scripts/mww/infer_verify_v3.py \
        --test-dir "$TEST_DIR" \
        --model "${OUTPUT_DIR}/${KEYWORD_ID}.tflite" \
        --cutoff 0.80 --window 3 2>&1 || true
fi

log "=== 救命救命 v1 complete ==="

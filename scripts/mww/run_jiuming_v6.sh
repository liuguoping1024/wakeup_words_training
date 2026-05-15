#!/usr/bin/env bash
# jiuming v6 训练流水线
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs"
MWW_DIR="/workspace/work/micro-wake-word"
KEYWORD_ID="jiuming_v6"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== jiuming v6 训练 ==="

# Step 1: 准备 repos
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh

# Step 2: 下载增强数据集
${SCRIPT_DIR}/02_download_datasets.sh

# Step 3: 准备增强音频
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# Step 4: 对抗性负样本 mmap（复用 v5 的）
ADVERSARIAL_MMAP="${DATA_DIR}/negative_datasets/zh_adversarial/zh_adversarial/training/zh_adversarial_mmap"
if [ -d "$ADVERSARIAL_MMAP" ]; then
    log "Step 4: 对抗性负样本 mmap 已存在，跳过"
else
    log "Step 4: 生成对抗性负样本 mmap..."
    python3 ${SCRIPT_DIR}/generate_adversarial_features.py \
        --input-dir /workspace/outputs/adversarial_merged \
        --output-dir ${DATA_DIR}/negative_datasets/zh_adversarial/zh_adversarial \
        --data-dir ${DATA_DIR}
fi

# Step 5: 生成正样本特征（v6 正样本：真实录音 ×10 + TTS）
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
log "Step 5: 清理旧特征并重新生成..."
rm -rf "${FEATURES_DIR}"

AUGMENTED_DIR="${DATA_DIR}/positive_augmented/jiuming_v6"
python3 ${SCRIPT_DIR}/03_generate_features_real.py \
    --positive-dir "${AUGMENTED_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURES_DIR}"

# Step 6: 写训练配置
log "Step 6: 写训练配置..."
python3 ${SCRIPT_DIR}/06_write_training_config_v6.py \
    --steps 25000 \
    --train-dir "trained_models/${KEYWORD_ID}" \
    --output "${MWW_DIR}/training_parameters.yaml"

# Step 7: 训练
log "Step 7: 开始训练..."
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

# Step 8: 导出
log "Step 8: 导出模型..."
TFLITE_SRC="${MWW_DIR}/trained_models/${KEYWORD_ID}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
if [ -f "$TFLITE_SRC" ]; then
    cp -f "${TFLITE_SRC}" "${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
    cat > "${OUTPUT_DIR}/${KEYWORD_ID}.json" <<EOF
{
  "type": "micro",
  "wake_word": "救命",
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
    log "WARNING: TFLite 未找到: ${TFLITE_SRC}"
    find "${MWW_DIR}/trained_models/${KEYWORD_ID}" -name "*.tflite" 2>/dev/null
fi

log "=== v6 Pipeline finished ==="

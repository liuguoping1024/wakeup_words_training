#!/usr/bin/env bash
# jiuming v5 完整训练流水线
# 在 wakeword-mww Docker 内运行
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs"
KEYWORD_ID="jiuming_v5"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== jiuming v5 训练流水线 ==="

# Step 1: 准备 repos 和 patch
log "Step 1: 准备 MWW 源码..."
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh

# Step 2: 下载增强数据集（如果没有）
log "Step 2: 下载增强数据集..."
${SCRIPT_DIR}/02_download_datasets.sh

# Step 3: 准备增强音频
log "Step 3: 准备增强音频..."
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# Step 4: 生成对抗性负样本的 mmap 特征
ADVERSARIAL_MMAP="${DATA_DIR}/negative_datasets/zh_adversarial/zh_adversarial/training/zh_adversarial_mmap"
if [ -d "$ADVERSARIAL_MMAP" ]; then
    log "Step 4: 对抗性负样本 mmap 已存在，跳过"
else
    log "Step 4: 生成对抗性负样本 mmap 特征..."
    python3 ${SCRIPT_DIR}/generate_adversarial_features.py \
        --input-dir /workspace/outputs/adversarial_merged \
        --output-dir ${DATA_DIR}/negative_datasets/zh_adversarial/zh_adversarial \
        --data-dir ${DATA_DIR}
fi

# Step 5: 生成正样本特征（复用 v4 的正样本）
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
if [ -d "${FEATURES_DIR}" ]; then
    log "Step 5: 清理旧特征..."
    rm -rf "${FEATURES_DIR}"
fi

log "Step 5: 生成正样本特征..."
AUGMENTED_DIR="${DATA_DIR}/positive_augmented/jiuming_v4"
python3 ${SCRIPT_DIR}/03_generate_features_real.py \
    --positive-dir "${AUGMENTED_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURES_DIR}"

# Step 6: 写训练配置
log "Step 6: 写训练配置..."
python3 ${SCRIPT_DIR}/06_write_training_config_v5.py \
    --steps 20000 \
    --train-dir "trained_models/${KEYWORD_ID}" \
    --output "${DATA_DIR}/training_config_${KEYWORD_ID}.yaml"

# Step 7: 训练
log "Step 7: 开始训练..."
cd /workspace/work/micro-wake-word

# 创建符号链接
ln -sfn "${DATA_DIR}/generated_augmented_features" /workspace/work/micro-wake-word/generated_augmented_features
ln -sfn "${DATA_DIR}/negative_datasets" /workspace/work/micro-wake-word/negative_datasets

# 复制训练配置到 MWW 目录
cp "${DATA_DIR}/training_config_${KEYWORD_ID}.yaml" /workspace/work/micro-wake-word/training_parameters.yaml

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

# Step 8: 导出 TFLite
log "Step 8: 导出模型..."
TRAIN_DIR="${DATA_DIR}/trained_models/${KEYWORD_ID}"
TFLITE_PATH="${TRAIN_DIR}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
if [ -f "$TFLITE_PATH" ]; then
    cp -f "${TFLITE_PATH}" "${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
    log "Exported: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
else
    log "WARNING: TFLite 文件未找到: ${TFLITE_PATH}"
    # 尝试查找
    find "${TRAIN_DIR}" -name "*.tflite" 2>/dev/null | head -5
fi

log "=== Pipeline finished ==="
log "Model: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"

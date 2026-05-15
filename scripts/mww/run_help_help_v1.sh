#!/usr/bin/env bash
# help help v1 MWW 训练（在 wakeword-mww Docker 内运行）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs"
MWW_DIR="/workspace/work/micro-wake-word"
KEYWORD_ID="help_help_v1"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help v1 MWW 训练 ==="

# Step 1: 准备 repos
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh

# Step 2: 下载增强数据集
${SCRIPT_DIR}/02_download_datasets.sh

# Step 3: 准备增强音频
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# Step 4: 生成对抗性负样本 mmap
EN_ADV_MMAP="${DATA_DIR}/negative_datasets/en_adversarial/en_adversarial/training/en_adversarial_mmap"
if [ -d "$EN_ADV_MMAP" ]; then
    log "Step 4: 对抗性负样本 mmap 已存在，跳过"
else
    log "Step 4: 生成对抗性负样本 mmap..."
    python3 ${SCRIPT_DIR}/generate_adversarial_features.py \
        --input-dir /workspace/outputs/help_help_adversarial_merged \
        --output-dir ${DATA_DIR}/negative_datasets/en_adversarial/en_adversarial \
        --data-dir ${DATA_DIR}
fi

# Step 5: 生成正样本特征
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
log "Step 5: 生成正样本特征..."
rm -rf "${FEATURES_DIR}"

AUGMENTED_DIR="${DATA_DIR}/positive_augmented/help_help_v1"
python3 ${SCRIPT_DIR}/03_generate_features_real.py \
    --positive-dir "${AUGMENTED_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURES_DIR}"

# Step 6: 写训练配置
log "Step 6: 写训练配置..."
python3 ${SCRIPT_DIR}/06_write_training_config_help_help.py \
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
  "wake_word": "help help",
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
    log "WARNING: TFLite 未找到"
    find "${MWW_DIR}/trained_models/${KEYWORD_ID}" -name "*.tflite" 2>/dev/null
fi

# Step 9: 验证
log "Step 9: 验证..."
TEST_DIR="/workspace/outputs/help_help_edgetts/test"
if [ -d "$TEST_DIR" ] && [ -f "$TEST_DIR/manifest.json" ]; then
    python3 /workspace/scripts/mww/infer_verify_v3.py \
        --test-dir "$TEST_DIR" \
        --model "${OUTPUT_DIR}/${KEYWORD_ID}.tflite" \
        --cutoff 0.80 --window 3 2>&1 || true
fi

# 也测试真实录音
REAL_DIR="${DATA_DIR}/real_voices_help_help"
if [ -d "$REAL_DIR" ]; then
    # 建 manifest
    python3 -c "
import json, os
d = '${REAL_DIR}'
m = [{'file': f, 'text': 'help help(real)', 'category': 'positive', 'voice': 'real'}
     for f in sorted(os.listdir(d)) if f.endswith('.wav')]
with open(os.path.join(d, 'manifest.json'), 'w') as fp:
    json.dump(m, fp, indent=2)
print(f'{len(m)} real test files')
"
    log "验证真实录音..."
    python3 /workspace/scripts/mww/infer_verify_v3.py \
        --test-dir "$REAL_DIR" \
        --model "${OUTPUT_DIR}/${KEYWORD_ID}.tflite" \
        --cutoff 0.80 --window 3 2>&1 || true
fi

log "=== help help v1 完成 ==="

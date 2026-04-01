#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

log "=== 真实语音训练流水线 ==="
log "唤醒词: ${KEYWORD_PHRASE} (${KEYWORD_ID})"
log "WORK_DIR=${WORK_DIR}"
log "DATA_DIR=${DATA_DIR}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "SOUNDS_DIR=${SOUNDS_DIR}"

# ── 第 0 步：准备仓库和环境（复用已有脚本）──
/workspace/scripts/01_prepare_repos.sh
/workspace/scripts/00_patch_mww.sh

# ── 第 1 步：下载增强数据和负样本（复用已有脚本）──
/workspace/scripts/02_download_datasets.sh

source "${MWW_DIR}/.venv/bin/activate"

# ── 第 2 步：准备增强音频（MIT RIR / AudioSet / FMA → 16kHz）──
python /workspace/scripts/03_prepare_audio.py --data-dir "${DATA_DIR}"

# ── 第 3 步：切分真实语音 ──
POSITIVE_DIR="${DATA_DIR}/positive_raw/${KEYWORD_ID}"
EXISTING_COUNT=$(find "${POSITIVE_DIR}" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l)
if [[ "${EXISTING_COUNT}" -gt 0 ]]; then
  log "已有 ${EXISTING_COUNT} 条切分样本，跳过切分步骤"
else
  log "切分真实语音到 ${POSITIVE_DIR} ..."
  python /workspace/scripts-real/01_split_real_voices.py \
    --sounds-dir "${SOUNDS_DIR}" \
    --output-dir "${POSITIVE_DIR}"
fi

SPLIT_COUNT=$(find "${POSITIVE_DIR}" -maxdepth 1 -name "*.wav" | wc -l)
log "切分完成: ${SPLIT_COUNT} 条原始样本"

# ── 第 4 步：数据增强扩充 ──
AUGMENTED_DIR="${DATA_DIR}/positive_augmented/${KEYWORD_ID}"
TARGET_POSITIVE="${TARGET_POSITIVE:-5000}"
EXISTING_AUG=$(find "${AUGMENTED_DIR}" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l)
if [[ "${EXISTING_AUG}" -ge "${TARGET_POSITIVE}" ]]; then
  log "已有 ${EXISTING_AUG} 条增强样本 >= 目标 ${TARGET_POSITIVE}，跳过增强步骤"
else
  log "数据增强，目标 ${TARGET_POSITIVE} 条 ..."
  python /workspace/scripts-real/02_augment_positives.py \
    --input "${POSITIVE_DIR}" \
    --output "${AUGMENTED_DIR}" \
    --target "${TARGET_POSITIVE}"
fi

# ── 第 5 步：生成频谱特征 ──
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
# 清理旧特征（如果存在）
if [[ -d "${FEATURES_DIR}" ]]; then
  log "清理旧特征目录..."
  rm -rf "${FEATURES_DIR}"
fi

log "生成频谱特征..."
python /workspace/scripts-real/03_generate_features.py \
  --positive-dir "${AUGMENTED_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${FEATURES_DIR}"

# ── 第 6 步：训练和导出 ──
/workspace/scripts-real/05_train_and_export.sh

log "=== 流水线完成 ==="
log "模型: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
log "配置: ${OUTPUT_DIR}/${KEYWORD_ID}.json"

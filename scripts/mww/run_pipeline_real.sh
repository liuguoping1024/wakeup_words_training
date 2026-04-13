#!/usr/bin/env bash
# MWW Real Voice pipeline: for wake words trained with real voice recordings
# Used for "你好树实" and similar
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

log "=== MWW Real Voice Pipeline ==="
log "Keyword: ${KEYWORD_PHRASE} (${KEYWORD_ID})"

# Step 0-1: Prepare repos and patch
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh

# Step 2: Download augmentation and negative datasets
${SCRIPT_DIR}/02_download_datasets.sh

# Step 3: Prepare augmentation audio
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# Step 4: Split real voices
POSITIVE_DIR="${DATA_DIR}/positive_raw/${KEYWORD_ID}"
EXISTING_COUNT=$(find "${POSITIVE_DIR}" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l)
if [[ "${EXISTING_COUNT}" -gt 0 ]]; then
  log "Already have ${EXISTING_COUNT} split samples, skipping"
else
  log "Splitting real voices..."
  python3 ${SCRIPT_DIR}/01_split_real_voices.py \
    --sounds-dir "${SOUNDS_DIR}" \
    --output-dir "${POSITIVE_DIR}"
fi

SPLIT_COUNT=$(find "${POSITIVE_DIR}" -maxdepth 1 -name "*.wav" | wc -l)
log "Split samples: ${SPLIT_COUNT}"

# Step 5: Augment if needed
AUGMENTED_DIR="${DATA_DIR}/positive_augmented/${KEYWORD_ID}"
EXISTING_AUG=$(find "${AUGMENTED_DIR}" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l)
if [[ "${EXISTING_AUG}" -ge "${TARGET_POSITIVE}" ]]; then
  log "Already have ${EXISTING_AUG} augmented samples >= target ${TARGET_POSITIVE}, skipping"
else
  log "Augmenting to ${TARGET_POSITIVE} samples..."
  python3 ${SCRIPT_DIR}/02_augment_positives.py \
    --input "${POSITIVE_DIR}" \
    --output "${AUGMENTED_DIR}" \
    --target "${TARGET_POSITIVE}"
fi

# Step 6: Generate features
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
if [[ -d "${FEATURES_DIR}" ]]; then
  log "Cleaning old features..."
  rm -rf "${FEATURES_DIR}"
fi

python3 ${SCRIPT_DIR}/03_generate_features_real.py \
  --positive-dir "${AUGMENTED_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${FEATURES_DIR}"

# Step 7: Train and export
${SCRIPT_DIR}/07_train_and_export_real.sh

log "=== Pipeline finished ==="
log "Model: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"

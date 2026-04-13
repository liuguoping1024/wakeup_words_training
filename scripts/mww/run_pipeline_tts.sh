#!/usr/bin/env bash
# MWW TTS pipeline: for "help me" style wake words using TTS-generated positive samples
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

log "=== MWW TTS Pipeline ==="
log "Keyword: ${KEYWORD_PHRASE} (${KEYWORD_ID})"

# Step 0-1: Prepare repos and patch
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh

# Step 2: Download augmentation and negative datasets
${SCRIPT_DIR}/02_download_datasets.sh

# Step 3: Prepare augmentation audio (resample to 16kHz)
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# Step 4: Generate TTS positive samples
${SCRIPT_DIR}/04_generate_positive_samples.sh

# Step 5: Generate spectrogram features
REAL_VOICE_ARG=""
if [[ -d "${DATA_DIR}/real_voices" ]] && ls "${DATA_DIR}/real_voices/"*.wav > /dev/null 2>&1; then
    REAL_VOICE_ARG="--real-voice-dir ${DATA_DIR}/real_voices"
    log "Found real voice samples: $(ls ${DATA_DIR}/real_voices/*.wav | wc -l), will merge"
fi

python3 ${SCRIPT_DIR}/05_generate_features.py \
  --positive-dir "${DATA_DIR}/positive_raw/${KEYWORD_ID}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${DATA_DIR}/generated_augmented_features" \
  ${REAL_VOICE_ARG}

# Step 6-7: Train and export
${SCRIPT_DIR}/07_train_and_export.sh

log "=== Pipeline finished ==="

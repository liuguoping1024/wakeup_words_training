#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

log "Pipeline started for keyword: ${KEYWORD_PHRASE}"
log "WORK_DIR=${WORK_DIR}"
log "DATA_DIR=${DATA_DIR}"
log "OUTPUT_DIR=${OUTPUT_DIR}"

/workspace/scripts/01_prepare_repos.sh
/workspace/scripts/00_patch_mww.sh
/workspace/scripts/02_download_datasets.sh

source "${MWW_DIR}/.venv/bin/activate"
python /workspace/scripts/03_prepare_audio.py --data-dir "${DATA_DIR}"
deactivate

/workspace/scripts/04_generate_positive_samples.sh

source "${MWW_DIR}/.venv/bin/activate"

# 如果存在真实人声目录，混入正样本
REAL_VOICE_ARG=""
if [[ -d "${DATA_DIR}/real_voices" ]] && ls "${DATA_DIR}/real_voices/"*.wav > /dev/null 2>&1; then
    REAL_VOICE_ARG="--real-voice-dir ${DATA_DIR}/real_voices"
    log "发现真实人声样本：$(ls ${DATA_DIR}/real_voices/*.wav | wc -l) 条，将混入训练"
fi

python /workspace/scripts/05_generate_features.py \
  --positive-dir "${DATA_DIR}/positive_raw/${KEYWORD_ID}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${DATA_DIR}/generated_augmented_features" \
  ${REAL_VOICE_ARG}

/workspace/scripts/07_train_and_export.sh

log "Pipeline finished."

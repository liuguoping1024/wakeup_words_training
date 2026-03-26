#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_dir "${DATA_DIR}/positive_raw/${KEYWORD_ID}"

# If user already prepared data, keep it.
if compgen -G "${DATA_DIR}/positive_raw/${KEYWORD_ID}/*.wav" > /dev/null; then
  log "Positive wav samples already exist, skipping generation."
  exit 0
fi

if [[ -d "${PSG_DIR}" ]]; then
  log "Trying piper-sample-generator first..."
  cd "${PSG_DIR}"
  set +e
  python3 generate_samples.py "${KEYWORD_PHRASE}" \
    --max-samples "${POSITIVE_SAMPLES}" \
    --batch-size 100 \
    --output-dir "${DATA_DIR}/positive_raw/${KEYWORD_ID}"
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    log "Generated positive samples with piper-sample-generator."
    exit 0
  fi
  log "piper-sample-generator failed, fallback to espeak-ng synthetic generation."
fi

require_cmd espeak-ng
require_cmd ffmpeg

tmp_dir="${DATA_DIR}/positive_raw/${KEYWORD_ID}/_tmp_espeak"
ensure_dir "${tmp_dir}"

for i in $(seq 1 "${POSITIVE_SAMPLES}"); do
  speed=$((120 + (i % 80)))
  pitch=$((35 + (i % 45)))
  amp=$((80 + (i % 20)))
  tmp_wav="${tmp_dir}/${i}.wav"
  out_wav="${DATA_DIR}/positive_raw/${KEYWORD_ID}/${i}.wav"

  espeak-ng -s "${speed}" -p "${pitch}" -a "${amp}" -w "${tmp_wav}" "${KEYWORD_PHRASE}"
  ffmpeg -y -loglevel error -i "${tmp_wav}" -ac 1 -ar 22050 -sample_fmt s16 "${out_wav}"
done

rm -rf "${tmp_dir}"
log "Generated ${POSITIVE_SAMPLES} positive samples via espeak-ng fallback."

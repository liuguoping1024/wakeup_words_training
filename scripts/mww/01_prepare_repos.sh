#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_dir "${WORK_DIR}"

if [[ ! -d "${MWW_DIR}" ]]; then
  log "Cloning micro-wake-word..."
  git clone https://github.com/OHF-Voice/micro-wake-word.git "${MWW_DIR}"
else
  log "micro-wake-word already exists, skipping clone."
fi

if [[ ! -d "${PSG_DIR}" ]]; then
  log "Cloning piper-sample-generator..."
  git clone https://github.com/rhasspy/piper-sample-generator.git "${PSG_DIR}"
else
  log "piper-sample-generator already exists, skipping clone."
fi

# Install micro-wake-word as editable if not already
if ! python3 -c "import microwakeword" 2>/dev/null; then
  log "Installing micro-wake-word..."
  pip install -e "${MWW_DIR}" --no-deps
fi

# Install piper-sample-generator requirements if present
if [[ -f "${PSG_DIR}/requirements.txt" ]]; then
  log "Installing piper-sample-generator requirements..."
  pip install -r "${PSG_DIR}/requirements.txt" 2>/dev/null || true
fi

log "Repository and environment preparation completed."

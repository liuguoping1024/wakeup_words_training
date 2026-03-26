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

log "Preparing Python virtual environment in micro-wake-word..."
cd "${MWW_DIR}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pip install "datasets==2.19.2" soundfile librosa numpy scipy tqdm resampy huggingface_hub
# TF 2.16+ ships nvidia-cudnn-cu12==8.9.x via pip; no system cuDNN needed.
# Pin TF to 2.16.x: last release using cuDNN 8 and CUDA 12.3 via pip wheels.
pip install "tensorflow==2.16.2" "nvidia-cudnn-cu12==8.9.7.29"
# numpy-minmax and numpy-rms declare numpy>=2 but work fine with 1.26 at runtime;
# install with --no-deps to skip the version check.
pip install "numpy==1.26.4"
pip install "numpy-minmax==0.4.0" "numpy-rms==0.5.0" --no-deps
pip install torch torchaudio piper-phonemize-cross==1.2.1
if [[ -f "${PSG_DIR}/requirements.txt" ]]; then
  log "Installing piper-sample-generator requirements..."
  pip install -r "${PSG_DIR}/requirements.txt"
fi

log "Repository and environment preparation completed."

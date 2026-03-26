#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd wget
require_cmd tar
require_cmd unzip

ensure_dir "${DATA_DIR}/raw"
ensure_dir "${DATA_DIR}/negative_zips"
ensure_dir "${DATA_DIR}/negative_datasets"

cd "${DATA_DIR}/raw"

download_if_missing() {
  local url="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    log "Already downloaded: ${file}"
  else
    log "Downloading ${file}..."
    wget --tries=3 --timeout=30 --retry-connrefused -O "${file}" "${url}"
  fi
}

try_download_if_missing() {
  local url="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    log "Already downloaded: ${file}"
    return 0
  fi
  log "Trying download ${file}..."
  if wget --tries=2 --timeout=30 --retry-connrefused -O "${file}" "${url}"; then
    return 0
  fi
  rm -f "${file}"
  return 1
}

try_download_any() {
  local file="$1"
  shift
  local urls=("$@")
  if [[ -f "${file}" ]]; then
    log "Already downloaded: ${file}"
    return 0
  fi
  for url in "${urls[@]}"; do
    log "Trying URL for ${file}: ${url}"
    if wget --tries=2 --timeout=30 --retry-connrefused -O "${file}" "${url}"; then
      return 0
    fi
    rm -f "${file}"
  done
  return 1
}

redownload() {
  local url="$1"
  local file="$2"
  log "Re-downloading ${file}..."
  rm -f "${file}"
  wget --tries=3 --timeout=30 --retry-connrefused -O "${file}" "${url}"
}

ensure_valid_tar() {
  local file="$1"
  shift
  local urls=("$@")
  if tar -tf "${file}" >/dev/null 2>&1; then
    return 0
  fi
  log "Corrupted tar detected: ${file}"
  rm -f "${file}"
  if ! try_download_any "${file}" "${urls[@]}"; then
    log "Tar re-download failed for all URLs: ${file}"
    return 1
  fi
  if ! tar -tf "${file}" >/dev/null 2>&1; then
    log "Invalid tar after re-download: ${file}"
    rm -f "${file}"
    return 1
  fi
  return 0
}

ensure_valid_zip() {
  local file="$1"
  local url="$2"
  if unzip -tq "${file}" >/dev/null 2>&1; then
    return 0
  fi
  log "Corrupted zip detected: ${file}"
  redownload "${url}" "${file}"
  unzip -tq "${file}" >/dev/null 2>&1 || {
    log "Invalid zip after re-download: ${file}"
    exit 1
  }
}

# Positive augmentation sources
# AudioSet was converted to parquet and old tar links may 404.
AUDSET_URL_MAIN="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"
AUDSET_URL_COMMIT="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/196c0900867eff791b8f4d4be57db277e9a5b131/bal_train09.tar?download=true"
audio_ok=0
if try_download_any "bal_train09.tar" "${AUDSET_URL_MAIN}" "${AUDSET_URL_COMMIT}"; then
  audio_ok=1
fi
download_if_missing "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip" "fma_xs.zip"
ensure_valid_zip "fma_xs.zip" "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"

if [[ "${audio_ok}" -eq 1 ]]; then
  if ensure_valid_tar "bal_train09.tar" "${AUDSET_URL_MAIN}" "${AUDSET_URL_COMMIT}"; then
    audio_ok=1
  else
    audio_ok=0
  fi
else
  log "AudioSet tar unavailable; will use datasets API fallback in prepare_audio."
fi

ensure_dir "${DATA_DIR}/augmentation/audioset"
ensure_dir "${DATA_DIR}/augmentation/fma"
if [[ "${audio_ok}" -eq 1 && ! -d "${DATA_DIR}/augmentation/audioset/audio" ]]; then
  log "Extracting AudioSet tar..."
  tar -xf "bal_train09.tar" -C "${DATA_DIR}/augmentation/audioset"
fi
if [[ ! -d "${DATA_DIR}/augmentation/fma/fma_small" ]]; then
  log "Extracting FMA zip..."
  unzip -o "fma_xs.zip" -d "${DATA_DIR}/augmentation/fma"
fi

# Negative feature zips (pregenerated)
cd "${DATA_DIR}/negative_zips"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/speech.zip" "speech.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/speech_background.zip" "speech_background.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech.zip" "no_speech.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech_background.zip" "no_speech_background.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party.zip" "dinner_party.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party_eval.zip" "dinner_party_eval.zip"
download_if_missing "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party_background.zip" "dinner_party_background.zip"

ensure_valid_zip "speech.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/speech.zip"
ensure_valid_zip "speech_background.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/speech_background.zip"
ensure_valid_zip "no_speech.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech.zip"
ensure_valid_zip "no_speech_background.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech_background.zip"
ensure_valid_zip "dinner_party.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party.zip"
ensure_valid_zip "dinner_party_eval.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party_eval.zip"
ensure_valid_zip "dinner_party_background.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party_background.zip"

cd "${DATA_DIR}/negative_datasets"
for z in "${DATA_DIR}"/negative_zips/*.zip; do
  base="$(basename "${z}" .zip)"
  ensure_dir "${base}"
  if [[ -f "${base}/manifest.json" || -f "${base}/data_specs.json" ]]; then
    log "Already extracted: ${base}"
  else
    log "Extracting ${base}.zip..."
    unzip -o "${z}" -d "${base}"
  fi
done

log "Datasets download and extraction completed."

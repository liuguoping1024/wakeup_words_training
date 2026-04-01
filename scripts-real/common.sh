#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
MWW_DIR="${MWW_DIR:-${WORK_DIR}/micro-wake-word}"
PSG_DIR="${PSG_DIR:-${WORK_DIR}/piper-sample-generator}"

KEYWORD_PHRASE="${KEYWORD_PHRASE:-你好树实}"
KEYWORD_ID="${KEYWORD_ID:-nihao_shushi}"
TRAIN_STEPS="${TRAIN_STEPS:-15000}"

# 真实语音源目录
SOUNDS_DIR="${SOUNDS_DIR:-${ROOT_DIR}/data/sounds}"

log() {
  echo "[$(date +'%F %T')] $*"
}

ensure_dir() {
  mkdir -p "$1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

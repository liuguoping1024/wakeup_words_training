#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
MWW_DIR="${MWW_DIR:-${WORK_DIR}/micro-wake-word}"
PSG_DIR="${PSG_DIR:-${WORK_DIR}/piper-sample-generator}"

KEYWORD_PHRASE="${KEYWORD_PHRASE:-help me}"
KEYWORD_ID="${KEYWORD_ID:-help_me}"
PIPER_VOICES="${PIPER_VOICES:-en_US-lessac-medium,en_US-amy-medium}"
POSITIVE_SAMPLES="${POSITIVE_SAMPLES:-800}"
TRAIN_STEPS="${TRAIN_STEPS:-12000}"

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

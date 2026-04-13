#!/usr/bin/env bash
set -euo pipefail

# 容器内路径
WORK_DIR="${WORK_DIR:-/workspace/work}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
SCRIPT_DIR="${SCRIPT_DIR:-/workspace/scripts/mww}"
MWW_DIR="${MWW_DIR:-${WORK_DIR}/micro-wake-word}"
PSG_DIR="${PSG_DIR:-${WORK_DIR}/piper-sample-generator}"

KEYWORD_PHRASE="${KEYWORD_PHRASE:-help me}"
KEYWORD_ID="${KEYWORD_ID:-help_me}"
PIPER_VOICES="${PIPER_VOICES:-en_US-lessac-medium,en_US-amy-medium}"
POSITIVE_SAMPLES="${POSITIVE_SAMPLES:-800}"
TRAIN_STEPS="${TRAIN_STEPS:-12000}"
TARGET_POSITIVE="${TARGET_POSITIVE:-5000}"

# 真实语音源目录
SOUNDS_DIR="${SOUNDS_DIR:-${DATA_DIR}/sounds}"

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

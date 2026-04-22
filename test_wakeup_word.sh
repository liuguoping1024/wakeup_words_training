#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 连续唤醒词检测（无需按 Enter，自动循环录音+检测）
#
# 用法：
#   ./test_wakeup_word.sh
#   MODEL=inference/help_me.tflite CUTOFF=0.10 WINDOW=5 ./test_wakeup_word.sh
#   DURATION=3 ./test_wakeup_word.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

usage() {
  cat <<EOF
用法: ./test_wakeup_word.sh [选项]

连续自动唤醒词检测（无需按 Enter，自动循环录音+检测）

选项（通过环境变量设置）:
  MODEL=PATH       tflite 模型路径（默认 inference/jiuming_v3.tflite）
  CUTOFF=FLOAT     触发阈值 0~1（默认 0.30）
  WINDOW=N         连续帧数（默认 3）
  DURATION=N       每轮录音秒数（默认 3）
  PULSE_SOCKET=PATH  PulseAudio socket 路径

示例:
  ./test_wakeup_word.sh
  MODEL=inference/help_me.tflite CUTOFF=0.10 WINDOW=5 ./test_wakeup_word.sh
  DURATION=5 ./test_wakeup_word.sh
EOF
  exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INF_DIR="${ROOT_DIR}/inference"
VENV_DIR="${ROOT_DIR}/work/.venv"

MODEL="${MODEL:-${INF_DIR}/jiuming_v3.tflite}"
CUTOFF="${CUTOFF:-0.5}"
WINDOW="${WINDOW:-3}"
DURATION="${DURATION:-3}"
PULSE_SOCKET="${PULSE_SOCKET:-/var/lib/homeassistant/audio/external/pulse.sock}"

# ── 模型检查 ─────────────────────────────────────────────────────────
if [[ ! -f "${MODEL}" ]]; then
  echo "[error] 模型不存在: ${MODEL}"
  exit 1
fi

# ── PulseAudio 检查 ──────────────────────────────────────────────────
if [[ ! -S "${PULSE_SOCKET}" ]]; then
  echo "[error] PulseAudio socket 不存在: ${PULSE_SOCKET}"
  exit 1
fi

# ── Python 环境（统一使用 work/.venv）────────────────────────────────
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
elif [[ -x "${INF_DIR}/.venv/bin/python" ]]; then
  PYTHON="${INF_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

if ! command -v parec >/dev/null 2>&1; then
  echo "[error] 缺少 parec，请安装：apt-get install -y pulseaudio-utils"
  exit 1
fi

export PULSE_SERVER="unix:${PULSE_SOCKET}"

MIC_SOURCE="$(pactl info 2>/dev/null | awk -F': ' '/Default Source:/{print $2}')"
if [[ -n "${MIC_SOURCE}" ]]; then
  pactl set-source-volume "${MIC_SOURCE}" 100% >/dev/null 2>&1 || true
  echo "[info] 默认输入源: ${MIC_SOURCE}"
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         连续唤醒词检测                       ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Python : ${PYTHON}"
echo "║  Model  : ${MODEL}"
echo "║  Params : cutoff=${CUTOFF}, window=${WINDOW}"
echo "║  Chunk  : ${DURATION}s / 轮"
echo "║  Pulse  : ${PULSE_SOCKET}"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "[info] 自动连续检测，无需按键（Ctrl+C 退出）"
echo ""

round=0
hits=0

cleanup() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  共检测 ${round} 轮，触发 ${hits} 次"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  rm -f /tmp/wakeup_test_$$.wav
  exit 0
}
trap cleanup INT TERM

while true; do
  round=$((round + 1))
  TMPWAV="/tmp/wakeup_test_$$.wav"

  echo "──── 第 ${round} 轮 ── 录音中 (${DURATION}s)... 请说唤醒词 ────"

  # 录音
  "${PYTHON}" - "${TMPWAV}" "${DURATION}" <<'PYEOF'
import os, sys, subprocess, wave

tmpwav = sys.argv[1]
duration = float(sys.argv[2])
rate = 16000
n_bytes = int(rate * 1 * 2 * duration)

proc = subprocess.Popen(
    ["parec", f"--rate={rate}", "--format=s16le", "--channels=1", "--latency-msec=50"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=os.environ
)
raw = b""
while len(raw) < n_bytes:
    chunk = proc.stdout.read(min(4096, n_bytes - len(raw)))
    if not chunk:
        break
    raw += chunk
proc.terminate()
proc.wait()
raw = raw.ljust(n_bytes, b"\x00")[:n_bytes]

with wave.open(tmpwav, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(raw)
PYEOF

  # 检测
  result=$("${PYTHON}" "${INF_DIR}/detect.py" \
    --model "${MODEL}" \
    --wav "${TMPWAV}" \
    --cutoff "${CUTOFF}" \
    --window "${WINDOW}" \
    --verbose 2>&1) || true

  echo "${result}"

  if echo "${result}" | grep -q "检测到唤醒词"; then
    hits=$((hits + 1))
    echo "  🔔 触发！(累计 ${hits}/${round})"
  else
    echo "  ── 未触发 (累计 ${hits}/${round})"
  fi

  rm -f "${TMPWAV}"
  echo ""

  # 短暂间隔，避免录音重叠
  sleep 0.3
done

#!/usr/bin/env bash
# 使用当前模型做 USB 耳机唤醒词测试（parec 录音 -> detect.py 离线判定）
# 默认参数：cutoff=0.30, window=3
#
# 用法：
#   ./test_usb_audio.sh
#   MODEL=/path/to/model.tflite CUTOFF=0.25 WINDOW=4 DURATION=3 ./test_usb_audio.sh

set -euo pipefail

usage() {
  cat <<EOF
用法: ./test_usb_audio.sh [选项]

USB 耳机唤醒词测试（按 Enter 触发单次录音+检测）

选项（通过环境变量设置）:
  MODEL=PATH       tflite 模型路径（默认 inference/jiuming_v5.tflite）
  CUTOFF=FLOAT     触发阈值 0~1（默认 0.30）
  WINDOW=N         连续帧数（默认 3）
  DURATION=N       每次录音秒数（默认 3）
  PULSE_SOCKET=PATH  PulseAudio socket 路径

示例:
  ./test_usb_audio.sh
  MODEL=inference/help_me.tflite CUTOFF=0.25 WINDOW=4 ./test_usb_audio.sh
  DURATION=5 ./test_usb_audio.sh
EOF
  exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INF_DIR="${ROOT_DIR}/inference"

MODEL="${MODEL:-${INF_DIR}/jiuming_v5.tflite}"
CUTOFF="${CUTOFF:-0.30}"
WINDOW="${WINDOW:-3}"
PULSE_SOCKET="${PULSE_SOCKET:-/var/lib/homeassistant/audio/external/pulse.sock}"
DURATION="${DURATION:-3}"

if [[ ! -f "${MODEL}" ]]; then
  echo "[error] 模型不存在: ${MODEL}"
  exit 1
fi

if [[ ! -S "${PULSE_SOCKET}" ]]; then
  echo "[error] PulseAudio socket 不存在: ${PULSE_SOCKET}"
  exit 1
fi

# 优先使用 work/.venv（统一 venv），回退 inference/.venv，最后 python3
if [[ -x "${ROOT_DIR}/work/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/work/.venv/bin/python"
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

echo "[info] Python: ${PYTHON}"
echo "[info] Model : ${MODEL}"
echo "[info] Params: cutoff=${CUTOFF}, window=${WINDOW}"
echo "[info] Chunk : ${DURATION}s/次（按 Enter 触发一次检测）"
echo "[info] Pulse : ${PULSE_SOCKET}"
echo ""
echo "[info] 开始检测（Ctrl+C 退出）"
echo "[info] 请对着 USB 耳机麦克风说：help me"
echo ""

while true; do
  read -r -p "[按 Enter 开始 ${DURATION}s 录音并检测] " _

  TMPWAV="/tmp/usb_test_$$.wav"
  trap 'rm -f "${TMPWAV}"' EXIT

  echo "[info] 录音中 ${DURATION}s..."
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
print(f"[info] 录音完成: {tmpwav}")
PYEOF

  "${PYTHON}" "${INF_DIR}/detect.py" \
    --model "${MODEL}" \
    --wav "${TMPWAV}" \
    --cutoff "${CUTOFF}" \
    --window "${WINDOW}" \
    --verbose

  rm -f "${TMPWAV}"
  echo ""
done


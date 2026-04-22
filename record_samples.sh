#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 连续录制唤醒词样本（3 秒/条）
#
# 用法：
#   ./scripts/record_samples.sh                    # 默认录 50 条
#   COUNT=100 ./scripts/record_samples.sh          # 录 100 条
#   OUTPUT_DIR=data/my_samples ./scripts/record_samples.sh
#   PREFIX=test OFFSET=50 ./scripts/record_samples.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

usage() {
  cat <<EOF
用法: ./record_samples.sh [选项]

连续录制唤醒词样本（默认 3 秒/条）

选项（通过环境变量设置）:
  COUNT=N          录制数量（默认 50）
  DURATION=N       每条时长秒数（默认 3）
  OUTPUT_DIR=PATH  输出目录（默认 data/real_voices）
  PREFIX=STR       文件名前缀（默认 real）
  OFFSET=N         起始编号（默认 0）
  PULSE_SOCKET=PATH  PulseAudio socket 路径

示例:
  ./record_samples.sh                              # 默认录 50 条
  COUNT=100 ./record_samples.sh                    # 录 100 条
  COUNT=20 PREFIX=test OFFSET=50 ./record_samples.sh
  OUTPUT_DIR=data/my_samples ./record_samples.sh
EOF
  exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

COUNT="${COUNT:-50}"
DURATION="${DURATION:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/real_voices}"
PREFIX="${PREFIX:-real}"
OFFSET="${OFFSET:-0}"
PULSE_SOCKET="${PULSE_SOCKET:-/var/lib/homeassistant/audio/external/pulse.sock}"

mkdir -p "${OUTPUT_DIR}"

# ── PulseAudio 检查 ──────────────────────────────────────────────────
if [[ ! -S "${PULSE_SOCKET}" ]]; then
  echo "[error] PulseAudio socket 不存在: ${PULSE_SOCKET}"
  exit 1
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
echo "║         唤醒词样本录制工具                   ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  录制数量: ${COUNT} 条"
echo "║  每条时长: ${DURATION} 秒"
echo "║  输出目录: ${OUTPUT_DIR}"
echo "║  文件前缀: ${PREFIX}"
echo "║  起始编号: ${OFFSET}"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "[提示] 每次录音前会有倒计时"
echo "[提示] 按 Ctrl+C 随时中止"
echo ""
read -r -p "准备好了吗？按 Enter 开始录制... " _

recorded=0
for i in $(seq 0 $((COUNT - 1))); do
  idx=$((OFFSET + i))
  fname=$(printf "%s_%04d.wav" "${PREFIX}" "${idx}")
  fpath="${OUTPUT_DIR}/${fname}"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  第 $((i + 1))/${COUNT} 条  →  ${fname}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # 倒计时
  echo -n "  准备..."
  sleep 1
  echo ""
  echo "  🎙️  开始说！录音中 (${DURATION}s)..."

  # 用 parec 录制 raw PCM，然后转 WAV
  TMPRAW="/tmp/rec_raw_$$.pcm"
  timeout $((DURATION + 1)) parec \
    --rate=16000 --format=s16le --channels=1 --latency-msec=50 \
    --raw 2>/dev/null | head -c $((16000 * 2 * DURATION)) > "${TMPRAW}" || true

  # raw PCM -> WAV (用 python 写 header，避免额外依赖)
  python3 - "${TMPRAW}" "${fpath}" "${DURATION}" <<'PYEOF'
import sys, wave

raw_path, wav_path, dur = sys.argv[1], sys.argv[2], float(sys.argv[3])
rate = 16000
expected = int(rate * 2 * dur)

with open(raw_path, "rb") as f:
    raw = f.read()

# 补齐或截断
raw = raw.ljust(expected, b"\x00")[:expected]

with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(raw)
PYEOF

  rm -f "${TMPRAW}"
  echo "  ✅ 已保存: ${fname}"
  recorded=$((recorded + 1))

  # 最后一条不需要等待
  if [[ $i -lt $((COUNT - 1)) ]]; then
    sleep 0.5
  fi
done

echo ""
echo "══════════════════════════════════════════════"
echo "  录制完成！共 ${recorded} 条"
echo "  保存在: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════"

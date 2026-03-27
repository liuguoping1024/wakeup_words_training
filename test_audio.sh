#!/usr/bin/env bash
# 录音 N 秒，立即回放
# 用法：bash test_audio.sh [时长秒数，默认3]

PULSE_SOCKET="/var/lib/homeassistant/audio/external/pulse.sock"
DURATION="${1:-3}"
RATE=16000
TMPWAV="/tmp/test_audio_$$.wav"

export PULSE_SERVER="unix:${PULSE_SOCKET}"

# ── 检查 socket ───────────────────────────────────────────────────────────────
if [[ ! -S "${PULSE_SOCKET}" ]]; then
  echo "[error] PulseAudio socket 不存在: ${PULSE_SOCKET}"
  exit 1
fi

# ── 自动获取当前默认麦克风并设增益 100% ──────────────────────────────────────
MIC_SOURCE=$(pactl info 2>/dev/null | awk '/Default Source:/{print $3}')
if [[ -n "${MIC_SOURCE}" ]]; then
  pactl set-source-volume "${MIC_SOURCE}" 100% \
    && echo "[info] 麦克风增益: 100%  (${MIC_SOURCE})" \
    || echo "[warn] 设置增益失败，继续"
else
  echo "[warn] 无法获取默认麦克风，跳过增益设置"
fi

# ── 录音（直接让 python 负责录音+封 WAV+播放，避免 shell 管道退出码问题）────
echo ""
echo "● 录音中 ${DURATION}s ... 请对着麦克风说话"
echo ""

python3 - "${DURATION}" "${TMPWAV}" "${RATE}" <<'PYEOF'
import sys, os, subprocess, wave, struct, math

duration = float(sys.argv[1])
tmpwav   = sys.argv[2]
rate     = int(sys.argv[3])
pulse    = "unix:/var/lib/homeassistant/audio/external/pulse.sock"
env      = {**os.environ, "PULSE_SERVER": pulse}

# ── 录音 ──────────────────────────────────────────────────────────────────────
n_bytes = int(rate * 1 * 2 * duration)
proc = subprocess.Popen(
    ["parec", f"--rate={rate}", "--format=s16le", "--channels=1", "--latency-msec=50"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
)
raw = b""
while len(raw) < n_bytes:
    chunk = proc.stdout.read(min(4096, n_bytes - len(raw)))
    if not chunk:
        break
    raw += chunk
proc.terminate()
proc.wait()

# 补零防止数据不足
raw = raw.ljust(n_bytes, b"\x00")[:n_bytes]

# ── 封装 WAV ──────────────────────────────────────────────────────────────────
with wave.open(tmpwav, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(raw)

# RMS（纯标准库，不依赖 numpy）
n_samples = len(raw) // 2
samples = struct.unpack(f"<{n_samples}h", raw)
rms = math.sqrt(sum(s * s for s in samples) / n_samples)
print(f"[info] 录音完成  RMS={rms:.0f}  时长={duration:.1f}s")

if rms < 50:
    print("[warn] 音量极低（RMS<50），麦克风可能没有拾到声音")

# ── 播放 ──────────────────────────────────────────────────────────────────────
print("")
print("▶  播放录音...")
r = subprocess.run(["paplay", tmpwav], env=env)
if r.returncode != 0:
    print(f"[error] paplay 返回 {r.returncode}")
else:
    print("   播放完毕")

os.unlink(tmpwav)
PYEOF

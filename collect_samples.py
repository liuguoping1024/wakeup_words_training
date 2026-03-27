#!/usr/bin/env python3
"""
真实人声采集脚本 —— 基于 parec（PulseAudio）
设备：USB PnP Sound Device，通过 /var/lib/homeassistant/audio/external/pulse.sock

用法：
  python3 collect_samples.py                       # 采集 50 条 "help me"
  python3 collect_samples.py --count 20            # 采集 20 条
  python3 collect_samples.py --out data/my_voices  # 指定输出目录
  python3 collect_samples.py --dur 2.5             # 每条 2.5 秒
"""
import argparse
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────────────────────

PULSE_SOCKET  = "/var/lib/homeassistant/audio/external/pulse.sock"
SAMPLE_RATE   = 16000    # 目标采样率（micro-wake-word 要求）
CHANNELS      = 1
RECORD_SECS   = 3.0      # 默认录音时长
MIN_RMS       = 150.0    # 最低音量阈值（过低视为无声）

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def record_parec(duration: float) -> bytes:
    """
    通过 parec 从 PulseAudio socket 录音，返回原始 s16le PCM bytes。
    精确字节数 = rate * channels * 2 * duration
    """
    n_bytes = int(SAMPLE_RATE * CHANNELS * 2 * duration)
    env = {**os.environ, "PULSE_SERVER": f"unix:{PULSE_SOCKET}"}

    proc = subprocess.Popen(
        [
            "parec",
            f"--rate={SAMPLE_RATE}",
            "--format=s16le",
            f"--channels={CHANNELS}",
            "--latency-msec=50",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    raw = b""
    try:
        while len(raw) < n_bytes:
            chunk = proc.stdout.read(min(4096, n_bytes - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    # 补零（防止极端情况下数据不足）
    if len(raw) < n_bytes:
        raw += b"\x00" * (n_bytes - len(raw))
    return raw[:n_bytes]


def compute_rms(raw_pcm: bytes) -> float:
    n = len(raw_pcm) // 2
    samples = struct.unpack(f"<{n}h", raw_pcm)
    return math.sqrt(sum(s * s for s in samples) / n)


def save_wav(path: Path, raw_pcm: bytes, rate: int = SAMPLE_RATE):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw_pcm)


def check_pulse_socket() -> bool:
    if not Path(PULSE_SOCKET).exists():
        print(f"[error] PulseAudio socket 不存在: {PULSE_SOCKET}")
        return False
    env = {**os.environ, "PULSE_SERVER": f"unix:{PULSE_SOCKET}"}
    try:
        r = subprocess.run(["pactl", "info"], capture_output=True, timeout=3, env=env)
        if r.returncode == 0:
            print(f"[info] PulseAudio 连接正常  socket={PULSE_SOCKET}")
            return True
    except Exception:
        pass
    print(f"[warn] pactl 测试失败，但 socket 存在，继续尝试...")
    return True


def get_default_source() -> str:
    """从 PulseAudio 自动获取当前默认麦克风 source 名称。"""
    env = {**os.environ, "PULSE_SERVER": f"unix:{PULSE_SOCKET}"}
    try:
        r = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=3, env=env)
        for line in r.stdout.splitlines():
            if "Default Source:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def ensure_mic_volume(source: str, target: int = 100) -> None:
    """将指定 source 的麦克风增益设为 target%，避免录音音量偏低。"""
    env = {**os.environ, "PULSE_SERVER": f"unix:{PULSE_SOCKET}"}
    try:
        r = subprocess.run(
            ["pactl", "set-source-volume", source, f"{target}%"],
            capture_output=True, timeout=3, env=env,
        )
        if r.returncode == 0:
            print(f"[info] 麦克风增益已设为 {target}%  ({source})")
        else:
            print(f"[warn] 设置麦克风增益失败")
    except Exception as e:
        print(f"[warn] set-source-volume 异常: {e}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="唤醒词人声采集（parec/PulseAudio）")
    ap.add_argument("--phrase",   default="help me",         help="唤醒词提示文字")
    ap.add_argument("--count",    type=int,   default=50,    help="采集条数（默认 50）")
    ap.add_argument("--out",      default="data/real_voices",help="输出目录")
    ap.add_argument("--dur",      type=float, default=3.0,   help="每条录音时长（秒）")
    ap.add_argument("--min-rms",  type=float, default=MIN_RMS,
                    help=f"最低音量 RMS（默认 {MIN_RMS}，低于此值重录）")
    args = ap.parse_args()

    # 检查依赖
    if subprocess.run(["which", "parec"], capture_output=True).returncode != 0:
        print("[error] 找不到 parec，请安装：apt-get install -y pulseaudio-utils")
        sys.exit(1)

    # 检查 PulseAudio
    if not check_pulse_socket():
        sys.exit(1)

    # 自动检测当前麦克风并设增益
    mic_source = get_default_source()
    if mic_source:
        print(f"[info] 当前麦克风: {mic_source}")
        ensure_mic_volume(mic_source, 100)
    else:
        print("[warn] 无法获取默认麦克风，跳过增益设置")

    # 准备输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 从现有文件数量确定起始编号
    existing = sorted(out_dir.glob("real_*.wav"))
    start_idx = len(existing)

    print()
    print("=" * 58)
    print(f"  唤醒词人声采集")
    print(f"  短语   : 「{args.phrase}」")
    print(f"  目标   : {args.count} 条  × {args.dur}s")
    print(f"  输出   : {out_dir}/")
    print(f"  已有   : {start_idx} 条，本次从 real_{start_idx:04d}.wav 开始")
    print(f"  音量   : RMS >= {args.min_rms:.0f}")
    print("=" * 58)
    print()
    print("  操作说明：")
    print("    • 按 Enter → 开始录音")
    print("    • 麦克风亮起后，清晰说出唤醒词，正常语速")
    print("    • 音量太低会自动提示重录")
    print("    • Ctrl+C 随时退出（已录内容保留）")
    print()

    collected = 0
    idx       = start_idx
    skipped   = 0

    try:
        while collected < args.count:
            remaining = args.count - collected
            try:
                input(f"  [{collected+1:>2}/{args.count}] 按 Enter 开始录音（剩余 {remaining} 条）...")
            except EOFError:
                print()
                break

            print(f"  ● 录音中 {args.dur:.1f}s ... 请说：「{args.phrase}」", end="", flush=True)

            raw_pcm = record_parec(args.dur)
            rms = compute_rms(raw_pcm)

            print(f"  完成  (RMS={rms:.0f})", end="")

            if rms < args.min_rms:
                skipped += 1
                print(f"\n  ✗ 音量太低（< {args.min_rms:.0f}），请靠近麦克风重试（已跳过 {skipped} 次）\n")
                continue

            fname = out_dir / f"real_{idx:04d}.wav"
            save_wav(fname, raw_pcm)
            print(f"  → ✓ {fname.name}")
            collected += 1
            idx += 1

    except KeyboardInterrupt:
        print("\n\n  [中断] 用户退出")

    print()
    print("=" * 58)
    total_in_dir = len(sorted(out_dir.glob("real_*.wav")))
    print(f"  本次采集：{collected} 条  |  跳过低音量：{skipped} 次")
    print(f"  目录总计：{total_in_dir} 条  ({out_dir}/)")
    print("=" * 58)
    print()
    if collected > 0:
        print("  下一步：将这些 WAV 文件用于训练")
        print(f"  训练时确保 {out_dir}/ 中的文件会被 05_generate_features.py 检测到")


if __name__ == "__main__":
    main()

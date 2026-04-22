#!/usr/bin/env python3
"""
唤醒词检测入口

用法：
  # 实时麦克风（自动使用 HA PulseAudio socket）
  python3 detect.py --model help_me.tflite

  # 指定 PulseAudio socket（默认已适配 HA）
  python3 detect.py --model help_me.tflite --pulse-socket /var/lib/homeassistant/audio/external/pulse.sock

  # 测试 WAV 文件
  python3 detect.py --model help_me.tflite --wav test.wav

  # 批量测试目录
  python3 detect.py --model help_me.tflite --wav-dir ./samples/

  # 显示逐帧概率（调阈值用）
  python3 detect.py --model help_me.tflite --wav test.wav --verbose

参数：
  --cutoff      触发阈值 (0~1)，默认 0.10
  --window      连续几帧超阈值才触发，默认 5
  --pulse-socket  PulseAudio unix socket 路径
"""
import argparse
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from runtime import WakeWordDetector, SAMPLE_RATE, STRIDE_SAMPLES

# HA hassio_audio 暴露的 PulseAudio socket
HA_PULSE_SOCKET = "/var/lib/homeassistant/audio/external/pulse.sock"


# ─── PulseAudio 工具 ─────────────────────────────────────────────────────────

def setup_pulse(socket_path: str) -> bool:
    """设置 PULSE_SERVER 环境变量，返回是否连接成功"""
    if not Path(socket_path).exists():
        return False
    os.environ["PULSE_SERVER"] = f"unix:{socket_path}"
    try:
        r = subprocess.run(
            ["pactl", "info"],
            capture_output=True, timeout=3,
            env={**os.environ, "PULSE_SERVER": f"unix:{socket_path}"},
        )
        return r.returncode == 0
    except Exception:
        return False


def get_default_source(socket_path: str) -> str:
    """从 PulseAudio 获取默认输入源名称"""
    try:
        r = subprocess.run(
            ["pactl", "info"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "PULSE_SERVER": f"unix:{socket_path}"},
        )
        for line in r.stdout.splitlines():
            if "Default Source:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


# ─── WAV 辅助 ────────────────────────────────────────────────────────────────

def load_wav_16k_mono(path: str) -> np.ndarray:
    """加载 WAV 并统一转为 16kHz 单声道 int16"""
    import soundfile as sf
    import resampy

    data, sr = sf.read(path, always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = resampy.resample(data.astype(np.float32), sr, SAMPLE_RATE)
    return (data * 32767).clip(-32768, 32767).astype(np.int16)


# ─── 测试 WAV ────────────────────────────────────────────────────────────────

def run_wav(detector: WakeWordDetector, wav_path: str, verbose: bool) -> int:
    """
    离线评估单个 WAV 文件，统计是否有触发。

    注意：这里的逻辑与 scripts/eval_model.py 中的 test_file 保持一致：
      - 整段音频只喂一次 feed_and_score(pcm)
      - 在返回的分数序列上做滑动窗口判定
      - 只关心「是否触发过一次」，返回 0/1
    """
    from collections import deque

    pcm = load_wav_16k_mono(wav_path)
    if len(pcm) == 0:
        return 0

    # 获取整段音频的分数序列（不再二次 feed）
    scores = detector.feed_and_score(pcm)
    if not scores:
        if verbose:
            print("[warn] 无得分输出，可能音频过短")
        return 0

    win: deque[float] = deque(maxlen=detector.window_count)
    triggered = False
    max_prob = 0.0

    for idx, s in enumerate(scores):
        max_prob = max(max_prob, s)
        win.append(s)
        if (
            len(win) == detector.window_count
            and all(v >= detector.cutoff for v in win)
        ):
            triggered = True
            if verbose:
                t = idx * STRIDE_SAMPLES / SAMPLE_RATE
                print(f"\n>>> 检测到唤醒词！  t≈{t:.2f}s  prob={s:.3f}")
            break

    if verbose:
        print(f"[info] {Path(wav_path).name} max_prob={max_prob:.3f}  hit={triggered}")

    return 1 if triggered else 0


def run_wav_dir(detector: WakeWordDetector, wav_dir: str, verbose: bool):
    wavs = sorted(Path(wav_dir).glob("*.wav"))
    if not wavs:
        print(f"[warn] 没有找到 WAV 文件：{wav_dir}")
        return

    hit, miss = 0, 0
    for wav in wavs:
        detector.reset()
        n = run_wav(detector, str(wav), verbose=False)
        status = "✓" if n > 0 else "✗"
        print(f"  {status}  {wav.name}  ({n} 次触发)")
        if n > 0:
            hit += 1
        else:
            miss += 1

    total = hit + miss
    print(f"\n结果: {hit}/{total} 检测到  ({hit/total*100:.1f}%)")


# ─── 实时麦克风（via PulseAudio socket）────────────────────────────────────

def run_mic(detector: WakeWordDetector, pulse_socket: str, verbose: bool = False):
    try:
        import pyaudio
    except ImportError:
        print("[error] 需要 pyaudio：pip install pyaudio")
        sys.exit(1)

    # 设置 PulseAudio 环境
    if not setup_pulse(pulse_socket):
        print(f"[warn] 无法连接 PulseAudio socket: {pulse_socket}")
        print("[warn] 尝试使用默认音频设备...")
        pulse_ok = False
    else:
        pulse_ok = True
        src = get_default_source(pulse_socket)
        print(f"[info] PulseAudio 连接成功")
        print(f"[info] 默认输入源: {src}")

    import resampy

    # 强制用 48000Hz（USB 麦克风原生采样率），pyaudio 汇报的 44100 不准确
    # PulseAudio 会负责格式转换，我们在 Python 再从 48k 重采样到 16k
    MIC_RATE = 48000
    MIC_CHUNK = int(MIC_RATE * STRIDE_SAMPLES / SAMPLE_RATE)  # 10ms @ 48kHz = 480 samples

    print(f"[info] 实时检测中，阈值={detector.cutoff}  说 'help me' 触发，Ctrl+C 退出\n")
    print(f"[info] 录音参数: {MIC_RATE}Hz -> 重采样至 {SAMPLE_RATE}Hz，chunk={MIC_CHUNK}")

    pa = pyaudio.PyAudio()

    # 找 PulseAudio 输入设备索引
    device_idx = None
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and "pulse" in d["name"].lower():
            device_idx = i
            print(f"[info] 使用设备 [{i}] {d['name']}")
            break

    # 依次尝试不同采样率，找到能打开的
    opened_rate = None
    stream = None
    for try_rate in [48000, 44100, 16000]:
        try_chunk = int(try_rate * STRIDE_SAMPLES / SAMPLE_RATE)
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=try_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=try_chunk,
            )
            opened_rate = try_rate
            MIC_RATE = try_rate
            MIC_CHUNK = try_chunk
            print(f"[info] 音频流打开成功: {try_rate}Hz  chunk={try_chunk}")
            break
        except Exception as e:
            print(f"[warn] {try_rate}Hz 失败: {e}")

    if stream is None:
        print("[error] 无法打开任何音频流")
        pa.terminate()
        sys.exit(1)

    last_t = 0.0

    try:
        while True:
            raw = stream.read(MIC_CHUNK, exception_on_overflow=False)
            chunk_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # 重采样到 16kHz
            if MIC_RATE != SAMPLE_RATE:
                chunk_f32 = resampy.resample(chunk_f32, MIC_RATE, SAMPLE_RATE)

            chunk_i16 = (chunk_f32 * 32767).clip(-32768, 32767).astype(np.int16)

            # feed_and_score 获取概率列表，同时推进内部状态
            scores = detector.feed_and_score(chunk_i16)
            if scores:
                prob = scores[-1]
                if verbose:
                    bar = "█" * int(prob * 40)
                    print(f"\r[{bar:<40}] {prob:.3f}", end="", flush=True)

                # 手动检测触发（替代 feed()，避免重复处理）
                detector._score_buf.extend(scores)
                if (
                    len(detector._score_buf) >= detector.window_count
                    and all(s >= detector.cutoff for s in list(detector._score_buf)[-detector.window_count:])
                ):
                    now = time.time()
                    if now - last_t > 1.0:
                        print(f"\n>>> 检测到唤醒词！ {time.strftime('%H:%M:%S')}")
                        last_t = now
                        detector._score_buf.clear()
    except KeyboardInterrupt:
        print("\n[info] 退出")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="help_me 唤醒词检测")
    ap.add_argument("--model",        default="help_me.tflite", help="tflite 模型路径")
    ap.add_argument("--wav",          default=None,  help="测试单个 WAV 文件")
    ap.add_argument("--wav-dir",      default=None,  help="批量测试目录下所有 WAV")
    ap.add_argument("--cutoff",       type=float, default=0.10, help="触发阈值（默认 0.10）")
    ap.add_argument("--window",       type=int,   default=5,    help="连续帧数（默认 5）")
    ap.add_argument("--verbose",      action="store_true",      help="显示逐帧概率")
    ap.add_argument("--pulse-socket", default=HA_PULSE_SOCKET,
                    help=f"PulseAudio socket 路径（默认 {HA_PULSE_SOCKET}）")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"[error] 找不到模型文件：{args.model}")
        sys.exit(1)

    detector = WakeWordDetector(args.model, cutoff=args.cutoff, window_count=args.window)
    print(f"[info] 模型加载成功  cutoff={args.cutoff}  window={args.window}")

    if args.wav_dir:
        run_wav_dir(detector, args.wav_dir, args.verbose)
    elif args.wav:
        n = run_wav(detector, args.wav, args.verbose)
        print(f"\n共检测到 {n} 次唤醒词")
    else:
        run_mic(detector, args.pulse_socket, verbose=args.verbose)


if __name__ == "__main__":
    main()

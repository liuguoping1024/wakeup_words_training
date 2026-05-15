#!/usr/bin/env python3
"""
预处理真实录音：
1. 裁剪前后静音（保留少量 padding）
2. 音量归一化到统一 RMS
3. 居中放置到固定长度（1.5s），前后补静音
4. 输出 16kHz mono int16 WAV

用法：
  python3 preprocess_real_voices.py \
    --input data/real_voices_jiuming \
    --output data/real_voices_jiuming_norm
"""
import argparse
import os
import wave
import struct
import numpy as np
from pathlib import Path


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def save_wav(path, audio, sr=16000):
    audio_i16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def trim_silence(audio, sr, threshold_db=-35, pad_ms=100):
    """裁剪前后静音，保留 pad_ms 的 padding"""
    threshold = 10 ** (threshold_db / 20)
    frame_len = int(sr * 0.01)  # 10ms frames
    pad_samples = int(sr * pad_ms / 1000)

    # 找到第一个超过阈值的帧
    start = 0
    for i in range(0, len(audio) - frame_len, frame_len):
        rms = np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        if rms > threshold:
            start = max(0, i - pad_samples)
            break

    # 找到最后一个超过阈值的帧
    end = len(audio)
    for i in range(len(audio) - frame_len, 0, -frame_len):
        rms = np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        if rms > threshold:
            end = min(len(audio), i + frame_len + pad_samples)
            break

    return audio[start:end]


def normalize_rms(audio, target_rms=0.08):
    """归一化到目标 RMS"""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-6:
        return audio
    gain = target_rms / current_rms
    # 限制增益，避免过度放大噪声
    gain = min(gain, 5.0)
    return np.clip(audio * gain, -1.0, 1.0)


def center_pad(audio, sr, target_duration_s=1.5):
    """居中放置到固定长度，前后补静音"""
    target_len = int(sr * target_duration_s)
    if len(audio) >= target_len:
        # 如果太长，取中间部分
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len]
    else:
        # 居中放置
        pad_before = (target_len - len(audio)) // 2
        pad_after = target_len - len(audio) - pad_before
        return np.concatenate([
            np.zeros(pad_before),
            audio,
            np.zeros(pad_after),
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-rms", type=float, default=0.08)
    parser.add_argument("--target-duration", type=float, default=1.5,
                        help="目标时长（秒），0 表示不做 padding")
    parser.add_argument("--trim-db", type=float, default=-35,
                        help="静音裁剪阈值 (dB)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wavs = sorted(Path(args.input).glob("*.wav"))
    print(f"输入: {len(wavs)} 个 WAV")

    stats = {"trimmed": 0, "too_short": 0, "ok": 0}

    for wav_path in wavs:
        audio, sr = load_wav(str(wav_path))

        # 1. 裁剪静音
        trimmed = trim_silence(audio, sr, threshold_db=args.trim_db)
        if len(trimmed) < sr * 0.15:  # 少于 150ms，可能有问题
            stats["too_short"] += 1
            print(f"  [warn] {wav_path.name}: 裁剪后太短 ({len(trimmed)/sr:.3f}s)，跳过裁剪")
            trimmed = audio

        orig_dur = len(audio) / sr
        trim_dur = len(trimmed) / sr
        if trim_dur < orig_dur * 0.8:
            stats["trimmed"] += 1

        # 2. 音量归一化
        normalized = normalize_rms(trimmed, target_rms=args.target_rms)

        # 3. 居中 padding
        if args.target_duration > 0:
            final = center_pad(normalized, sr, args.target_duration)
        else:
            final = normalized

        # 4. 保存
        out_path = os.path.join(args.output, wav_path.name)
        save_wav(out_path, final, sr)
        stats["ok"] += 1

    print(f"\n完成: {stats['ok']} 个文件")
    print(f"  裁剪了静音: {stats['trimmed']}")
    print(f"  太短跳过裁剪: {stats['too_short']}")

    # 验证输出
    out_wavs = sorted(Path(args.output).glob("*.wav"))
    rms_list = []
    for w in out_wavs:
        a, _ = load_wav(str(w))
        rms_list.append(np.sqrt(np.mean(a ** 2)))
    rms_arr = np.array(rms_list)
    print(f"\n输出 RMS: min={rms_arr.min():.4f} median={np.median(rms_arr):.4f} max={rms_arr.max():.4f}")


if __name__ == "__main__":
    main()

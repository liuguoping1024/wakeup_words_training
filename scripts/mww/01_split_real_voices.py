#!/usr/bin/env python3
"""
从 data/sounds 中的长 WAV 文件切分出单条唤醒词语音。

每个 WAV 文件约 12 分钟，包含 ~100-150 次"你好树实"。
使用能量 VAD 检测语音段，切分为独立 WAV（16kHz mono）。

用法：
  python 01_split_real_voices.py \
    --sounds-dir data/sounds \
    --output-dir data/positive_raw/nihao_shushi \
    --platforms Smartspeaker,smartspeaker,Android,Apple,IOS
"""
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SR = 16000


def load_audio_16k(path: Path) -> np.ndarray:
    """读取任意采样率的 WAV，返回 16kHz mono float32。"""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import resampy
            audio = resampy.resample(audio, sr, TARGET_SR)
        except ImportError:
            new_len = int(len(audio) * TARGET_SR / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
    return audio


def split_utterances(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 50,
    threshold_factor: float = 3.0,
    merge_gap_ms: int = 300,
    min_dur_s: float = 0.3,
    max_dur_s: float = 3.0,
    pad_ms: int = 150,
) -> list[np.ndarray]:
    """基于能量 VAD 切分语音段。"""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len

    # 计算每帧 RMS
    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    threshold = np.median(rms) * threshold_factor

    # 找连续语音段
    is_speech = rms > threshold
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(is_speech):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, n_frames))

    # 合并间距过小的段
    merge_gap = int(merge_gap_ms / frame_ms)
    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # 按时长过滤并提取
    min_frames = int(min_dur_s / (frame_ms / 1000))
    max_frames = int(max_dur_s / (frame_ms / 1000))
    pad_samples = int(sr * pad_ms / 1000)

    clips = []
    for s, e in merged:
        if min_frames <= (e - s) <= max_frames:
            start_sample = max(0, s * frame_len - pad_samples)
            end_sample = min(len(audio), e * frame_len + pad_samples)
            clips.append(audio[start_sample:end_sample])

    return clips


def main():
    parser = argparse.ArgumentParser(description="切分真实唤醒词录音")
    parser.add_argument("--sounds-dir", default="data/sounds")
    parser.add_argument("--output-dir", default="data/positive_raw/nihao_shushi")
    parser.add_argument(
        "--platforms",
        default=None,
        help="逗号分隔的平台目录名（默认自动扫描所有子目录）",
    )
    parser.add_argument("--min-dur", type=float, default=0.3)
    parser.add_argument("--max-dur", type=float, default=3.0)
    parser.add_argument("--threshold-factor", type=float, default=3.0,
                        help="能量阈值 = median(rms) * factor")
    args = parser.parse_args()

    sounds_dir = Path(args.sounds_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    platforms = (
        [p.strip() for p in args.platforms.split(",")]
        if args.platforms
        else None
    )

    # 收集所有 WAV 文件（自动扫描所有子目录或按指定平台名过滤）
    wav_files = []
    for speaker_dir in sorted(sounds_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for plat_dir in sorted(speaker_dir.iterdir()):
            if not plat_dir.is_dir():
                continue
            if platforms and plat_dir.name not in platforms:
                continue
            for wav in sorted(plat_dir.glob("*.wav")):
                wav_files.append(wav)

    print(f"[info] 找到 {len(wav_files)} 个 WAV 文件")

    total_clips = 0
    clip_idx = 0

    for wav_path in wav_files:
        speaker = wav_path.parent.parent.name
        platform = wav_path.parent.name

        try:
            audio = load_audio_16k(wav_path)
        except Exception as e:
            print(f"  [warn] 跳过 {wav_path}: {e}")
            continue

        dur = len(audio) / TARGET_SR
        clips = split_utterances(
            audio,
            threshold_factor=args.threshold_factor,
            min_dur_s=args.min_dur,
            max_dur_s=args.max_dur,
        )

        for clip in clips:
            out_name = f"{speaker}_{platform}_{clip_idx:05d}.wav"
            out_path = output_dir / out_name
            # 归一化到 [-0.95, 0.95]
            peak = np.max(np.abs(clip))
            if peak > 0:
                clip = clip / peak * 0.95
            sf.write(str(out_path), clip, TARGET_SR)
            clip_idx += 1

        total_clips += len(clips)
        print(f"  {speaker}/{platform}/{wav_path.name}: "
              f"{dur:.0f}s → {len(clips)} 条")

    print(f"\n[done] 共切分 {total_clips} 条唤醒词样本 → {output_dir}")


if __name__ == "__main__":
    main()

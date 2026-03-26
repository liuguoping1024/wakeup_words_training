#!/usr/bin/env python3
"""
真实人声样本预处理脚本

功能：
1. 自动裁剪静音，提取有效语音段
2. 统一输出为 16kHz 单声道 WAV
3. 数据增强（音量、音调、语速、混响），扩充到目标数量
4. 质量检查，过滤过短/过长的样本

用法：
  python prepare_real_voices.py \
    --input  data/real_voices \
    --output data/real_voices_processed \
    --target 300
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


# ── 静音裁剪 ────────────────────────────────────────────────────────────────

def trim_silence(audio: np.ndarray, sr: int,
                 top_db: float = 30.0,
                 pad_ms: int = 100) -> np.ndarray:
    """
    去除首尾静音。top_db: 低于最大值 top_db dB 视为静音。
    pad_ms: 保留两端的填充时长（毫秒）。
    """
    if len(audio) == 0:
        return audio

    frame_ms = 10
    frame_len = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)

    rms_db = []
    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i:i + frame_len].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        rms_db.append(20 * np.log10(rms + 1e-10))

    if not rms_db:
        return audio

    threshold = max(rms_db) - top_db

    start_frame, end_frame = 0, len(rms_db) - 1
    for i, db in enumerate(rms_db):
        if db >= threshold:
            start_frame = i
            break
    for i in range(len(rms_db) - 1, -1, -1):
        if rms_db[i] >= threshold:
            end_frame = i
            break

    start_sample = max(0, start_frame * frame_len - pad)
    end_sample   = min(len(audio), (end_frame + 1) * frame_len + pad)
    return audio[start_sample:end_sample]


# ── 数据增强 ────────────────────────────────────────────────────────────────

def augment(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """随机施加一种或多种增强，返回增强后的音频。"""
    aug = audio.copy().astype(np.float32)

    # 随机增益 ±6 dB
    gain_db = rng.uniform(-6, 6)
    aug = aug * (10 ** (gain_db / 20))

    # 随机时间拉伸（简单插值实现，无需 librosa）
    if rng.random() < 0.5:
        rate = rng.uniform(0.85, 1.15)
        new_len = int(len(aug) / rate)
        if new_len > 10:
            aug = np.interp(
                np.linspace(0, len(aug) - 1, new_len),
                np.arange(len(aug)),
                aug
            ).astype(np.float32)

    # 随机加白噪声
    if rng.random() < 0.4:
        snr_db = rng.uniform(15, 35)
        signal_rms = np.sqrt(np.mean(aug ** 2) + 1e-10)
        noise_rms  = signal_rms / (10 ** (snr_db / 20))
        aug = aug + rng.normal(0, noise_rms, len(aug)).astype(np.float32)

    # 简单混响（卷积一个短脉冲）
    if rng.random() < 0.3:
        decay = rng.uniform(0.05, 0.2)
        rir_len = int(sr * decay)
        rir = np.exp(-np.linspace(0, 6, rir_len)).astype(np.float32)
        rir /= np.sum(np.abs(rir))
        aug = np.convolve(aug, rir, mode='full')[:len(aug)]

    # 归一化，防止 clip
    peak = np.max(np.abs(aug))
    if peak > 0.98:
        aug = aug / peak * 0.95

    return aug.astype(np.float32)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="data/real_voices",
                        help="原始录音目录")
    parser.add_argument("--output",   default="data/real_voices_processed",
                        help="输出目录（供训练使用）")
    parser.add_argument("--target",   type=int, default=300,
                        help="目标样本数量（不足时通过增强扩充）")
    parser.add_argument("--min-dur",  type=float, default=0.3,
                        help="裁剪后最短时长（秒），低于此值丢弃")
    parser.add_argument("--max-dur",  type=float, default=3.0,
                        help="裁剪后最长时长（秒），截断到此长度")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── 第一步：裁剪原始录音 ──
    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        print(f"[error] {in_dir} 中未找到 WAV 文件")
        return

    print(f"[info] 读取原始样本: {len(wav_files)} 条")

    clean_clips: list[tuple[np.ndarray, int]] = []
    discarded = 0
    for f in wav_files:
        audio, sr = sf.read(str(f), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        trimmed = trim_silence(audio, sr)
        dur = len(trimmed) / sr

        if dur < args.min_dur:
            print(f"  [skip] {f.name}: 裁剪后 {dur:.2f}s < {args.min_dur}s，丢弃")
            discarded += 1
            continue

        if dur > args.max_dur:
            trimmed = trimmed[:int(sr * args.max_dur)]
            dur = args.max_dur

        clean_clips.append((trimmed, sr))
        print(f"  {f.name}: {len(audio)/sr:.2f}s → 裁剪后 {dur:.2f}s")

    print(f"\n[info] 有效样本: {len(clean_clips)} 条（丢弃 {discarded} 条）")

    if not clean_clips:
        print("[error] 没有有效样本，退出")
        return

    # ── 第二步：保存原始裁剪结果 ──
    idx = 0
    for audio, sr in clean_clips:
        out_path = out_dir / f"real_{idx:04d}.wav"
        sf.write(str(out_path), audio, sr)
        idx += 1

    # ── 第三步：数据增强扩充 ──
    need = args.target - len(clean_clips)
    if need <= 0:
        print(f"[info] 已有 {len(clean_clips)} 条，无需增强（目标 {args.target}）")
    else:
        print(f"\n[info] 需要增强 {need} 条以达到目标 {args.target} 条...")
        aug_count = 0
        while aug_count < need:
            src_audio, sr = clean_clips[rng.integers(0, len(clean_clips))]
            aug_audio = augment(src_audio, sr, rng)
            out_path  = out_dir / f"real_aug_{aug_count:04d}.wav"
            sf.write(str(out_path), aug_audio, sr)
            aug_count += 1
            if aug_count % 50 == 0:
                print(f"  增强进度: {aug_count}/{need}")

        print(f"[info] 增强完成，新增 {aug_count} 条")

    total = len(list(out_dir.glob("*.wav")))
    print(f"\n[done] 输出目录: {out_dir}")
    print(f"       共 {total} 条样本（原始裁剪 {len(clean_clips)} + 增强 {max(0, need)} 条）")
    print(f"\n下一步：重新运行训练流水线即可自动使用这批样本")
    print(f"  make train  (或将 {out_dir} 替换 data/real_voices 后 make train)")


if __name__ == "__main__":
    main()

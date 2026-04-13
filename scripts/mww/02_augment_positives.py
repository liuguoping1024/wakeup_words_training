#!/usr/bin/env python3
"""
对切分后的正样本做数据增强，扩充训练集多样性。

增强方式：增益、时间拉伸、白噪声、简单混响。
与 prepare_real_voices.py 类似，但这里的输入已经是切好的单条样本。

用法：
  python 02_augment_positives.py \
    --input  data/positive_raw/nihao_shushi \
    --output data/positive_augmented/nihao_shushi \
    --target 5000
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def augment(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """随机施加增强。"""
    aug = audio.copy().astype(np.float32)

    # 随机增益 ±6 dB
    gain_db = rng.uniform(-6, 6)
    aug = aug * (10 ** (gain_db / 20))

    # 随机时间拉伸
    if rng.random() < 0.5:
        rate = rng.uniform(0.85, 1.15)
        new_len = int(len(aug) / rate)
        if new_len > 10:
            aug = np.interp(
                np.linspace(0, len(aug) - 1, new_len),
                np.arange(len(aug)),
                aug,
            ).astype(np.float32)

    # 随机白噪声
    if rng.random() < 0.4:
        snr_db = rng.uniform(15, 35)
        signal_rms = np.sqrt(np.mean(aug ** 2) + 1e-10)
        noise_rms = signal_rms / (10 ** (snr_db / 20))
        aug = aug + rng.normal(0, noise_rms, len(aug)).astype(np.float32)

    # 简单混响
    if rng.random() < 0.3:
        decay = rng.uniform(0.05, 0.2)
        rir_len = int(sr * decay)
        rir = np.exp(-np.linspace(0, 6, rir_len)).astype(np.float32)
        rir /= np.sum(np.abs(rir))
        aug = np.convolve(aug, rir, mode="full")[: len(aug)]

    # 归一化
    peak = np.max(np.abs(aug))
    if peak > 0.98:
        aug = aug / peak * 0.95

    return aug.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/positive_raw/nihao_shushi")
    parser.add_argument("--output", default="data/positive_augmented/nihao_shushi")
    parser.add_argument("--target", type=int, default=5000,
                        help="目标样本总数（原始+增强）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 读取所有原始样本
    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        print(f"[error] {in_dir} 中未找到 WAV 文件")
        return

    print(f"[info] 原始样本: {len(wav_files)} 条")

    clips = []
    for f in wav_files:
        audio, sr = sf.read(str(f), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        clips.append(audio)

    # 先复制原始样本
    idx = 0
    for i, clip in enumerate(clips):
        sf.write(str(out_dir / f"pos_{idx:05d}.wav"), clip, TARGET_SR)
        idx += 1

    # 增强扩充
    need = args.target - len(clips)
    if need <= 0:
        print(f"[info] 已有 {len(clips)} 条 >= 目标 {args.target}，无需增强")
    else:
        print(f"[info] 需增强 {need} 条...")
        for i in range(need):
            src = clips[rng.integers(0, len(clips))]
            aug = augment(src, TARGET_SR, rng)
            sf.write(str(out_dir / f"pos_{idx:05d}.wav"), aug, TARGET_SR)
            idx += 1
            if (i + 1) % 500 == 0:
                print(f"  增强进度: {i + 1}/{need}")

    total = len(list(out_dir.glob("*.wav")))
    print(f"[done] 共 {total} 条样本 → {out_dir}")


if __name__ == "__main__":
    main()

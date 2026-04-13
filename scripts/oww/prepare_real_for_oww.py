#!/usr/bin/env python3
"""
将真实录音正样本转换为 openWakeWord 特征格式，跳过 TTS 生成。

用法：
  python prepare_real_for_oww.py \
    --positive-dir /workspace/data/positive_raw/nihao_shushi \
    --output-dir /workspace/outputs/oww \
    --model-name nihao_shushi \
    --background-dirs /workspace/data/augmentation/audioset_16k /workspace/data/augmentation/fma_16k \
    --rir-dir /workspace/data/augmentation/mit_rirs
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def load_wav_16k(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pos_dir = Path(args.positive_dir)
    wavs = sorted(pos_dir.glob("*.wav"))
    if not wavs:
        log.error(f"No WAV files in {pos_dir}")
        sys.exit(1)

    log.info(f"Found {len(wavs)} positive clips")
    random.shuffle(wavs)

    # Split train/val
    n_val = min(args.n_val, len(wavs) // 5)
    val_wavs = wavs[:n_val]
    train_wavs = wavs[n_val:n_val + args.n_train]

    # Create output dirs
    model_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy clips
    for i, wav in enumerate(train_wavs):
        audio = load_wav_16k(wav)
        out = os.path.join(train_dir, f"pos_{i:05d}.wav")
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    for i, wav in enumerate(val_wavs):
        audio = load_wav_16k(wav)
        out = os.path.join(test_dir, f"pos_{i:05d}.wav")
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    log.info(f"Prepared {len(train_wavs)} train + {len(val_wavs)} val clips in {model_dir}")

    # Also create empty negative dirs (OWW will generate adversarial negatives via TTS for English,
    # but for Chinese we skip that - the ACAV100M features serve as negatives)
    neg_train = os.path.join(model_dir, "negative_train")
    neg_test = os.path.join(model_dir, "negative_test")
    os.makedirs(neg_train, exist_ok=True)
    os.makedirs(neg_test, exist_ok=True)

    log.info("Done. Run augment_clips and train_model next.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
「你好树实」模型评估脚本

从切分好的正样本中随机抽取测试集评估召回率，
可选负样本测试误触发率。

用法：
  python eval_model.py \
    --model  outputs/nihao_shushi.tflite \
    --pos    data/positive_raw/nihao_shushi \
    --cutoff 0.10 \
    --window 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from runtime import WakeWordDetector, SAMPLE_RATE


def load_wav_as_int16(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        print(f"  [warn] 无法读取 {path.name}: {e}")
        return np.zeros(0, dtype=np.int16)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        except ImportError:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)), audio,
            ).astype(np.float32)
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def test_file(detector: WakeWordDetector, pcm: np.ndarray) -> tuple[bool, float]:
    detector.reset()
    scores = detector.feed_and_score(pcm)
    if not scores:
        return False, 0.0
    max_prob = max(scores)
    from collections import deque
    win: deque[float] = deque(maxlen=detector.window_count)
    triggered = False
    for s in scores:
        win.append(s)
        if (len(win) == detector.window_count
                and all(v >= detector.cutoff for v in win)):
            triggered = True
            break
    return triggered, max_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/nihao_shushi.tflite")
    parser.add_argument("--pos", default="data/positive_raw/nihao_shushi")
    parser.add_argument("--neg", default=None)
    parser.add_argument("--cutoff", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-pos", type=int, default=200)
    parser.add_argument("--max-neg", type=int, default=200)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        model_path = REPO_ROOT / args.model
    if not model_path.exists():
        print(f"[error] 模型不存在: {args.model}")
        sys.exit(1)

    print(f"模型: {model_path}")
    print(f"cutoff={args.cutoff}  window={args.window}\n")

    detector = WakeWordDetector(str(model_path), cutoff=args.cutoff, window_count=args.window)

    # 正样本
    pos_dir = Path(args.pos)
    if not pos_dir.exists():
        pos_dir = REPO_ROOT / args.pos
    pos_wavs = sorted(pos_dir.glob("*.wav"))[:args.max_pos]
    if not pos_wavs:
        print(f"[error] 正样本目录无 WAV: {pos_dir}")
        sys.exit(1)

    print(f"=== 正样本测试: {len(pos_wavs)} 条 ===")
    tp = fn = 0
    probs = []
    for wav in pos_wavs:
        pcm = load_wav_as_int16(wav)
        if len(pcm) == 0:
            continue
        triggered, prob = test_file(detector, pcm)
        probs.append(prob)
        flag = "✓" if triggered else "✗"
        print(f"  [{flag}] {wav.name:40s}  max_prob={prob:.3f}")
        if triggered:
            tp += 1
        else:
            fn += 1

    total = tp + fn
    recall = tp / total * 100 if total > 0 else 0
    print(f"\n召回率: {tp}/{total} = {recall:.1f}%")
    if probs:
        print(f"概率: min={min(probs):.3f}  max={max(probs):.3f}  avg={np.mean(probs):.3f}")

    # 负样本
    if args.neg:
        neg_dir = Path(args.neg)
        if not neg_dir.exists():
            neg_dir = REPO_ROOT / args.neg
        neg_wavs = sorted(neg_dir.rglob("*.wav"))[:args.max_neg]
        if neg_wavs:
            print(f"\n=== 负样本测试: {len(neg_wavs)} 条 ===")
            fp = tn = 0
            for wav in neg_wavs:
                pcm = load_wav_as_int16(wav)
                if len(pcm) == 0:
                    continue
                triggered, prob = test_file(detector, pcm)
                if triggered:
                    fp += 1
                    print(f"  [误触发] {wav.name}  max_prob={prob:.3f}")
                else:
                    tn += 1
            total_neg = fp + tn
            fpr = fp / total_neg * 100 if total_neg > 0 else 0
            print(f"\n误触发率: {fp}/{total_neg} = {fpr:.1f}%")

    print(f"\n{'='*50}")
    print(f"汇总  cutoff={args.cutoff}  window={args.window}")
    print(f"  召回率: {recall:.1f}%  ({tp}/{total})")


if __name__ == "__main__":
    main()

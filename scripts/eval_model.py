#!/usr/bin/env python3
"""
模型准确率评估脚本

用真实人声正样本 + 可选负样本，批量测试模型召回率和误触发率。

用法：
  python eval_model.py \
    --model  outputs/help_me.tflite \
    --pos    data/real_voices \
    --neg    data/augmentation/audioset_16k \   # 可选，测误触发
    --cutoff 0.10 \
    --window 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ── 把 inference/ 加入路径 ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from runtime import WakeWordDetector, SAMPLE_RATE


def load_wav_as_int16(path: Path) -> np.ndarray:
    """用 soundfile 读取 WAV，重采样到 16kHz，返回 int16。"""
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
            # 简单线性插值回退
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)

    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def test_file(detector: WakeWordDetector, pcm: np.ndarray) -> tuple[bool, float]:
    """
    将整段音频喂给检测器，返回 (是否触发, 最高概率)。
    每次测试前重置 streaming 状态。
    """
    detector.reset()
    scores = detector.feed_and_score(pcm)
    if not scores:
        return False, 0.0

    max_prob = max(scores)

    # 复现滑动窗口触发逻辑
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
    parser.add_argument("--model",   default="outputs/help_me.tflite")
    parser.add_argument("--pos",     default="data/real_voices",
                        help="正样本目录（包含唤醒词的 WAV）")
    parser.add_argument("--neg",     default=None,
                        help="负样本目录（不含唤醒词的 WAV，测误触发）")
    parser.add_argument("--cutoff",  type=float, default=0.10)
    parser.add_argument("--window",  type=int,   default=3)
    parser.add_argument("--max-neg", type=int,   default=200,
                        help="负样本最多测多少条（避免太慢）")
    args = parser.parse_args()

    model_path = REPO_ROOT / args.model
    if not model_path.exists():
        print(f"[error] 模型文件不存在: {model_path}")
        sys.exit(1)

    print(f"模型: {model_path}")
    print(f"cutoff={args.cutoff}  window={args.window}")
    print()

    detector = WakeWordDetector(
        str(model_path),
        cutoff=args.cutoff,
        window_count=args.window,
    )

    # ── 正样本测试（召回率） ──────────────────────────────────────────────────
    pos_dir = REPO_ROOT / args.pos
    pos_wavs = sorted(pos_dir.glob("*.wav"))
    if not pos_wavs:
        print(f"[error] 正样本目录无 WAV: {pos_dir}")
        sys.exit(1)

    print(f"=== 正样本测试（召回率）: {len(pos_wavs)} 条 ===")
    tp = fp_pos = 0
    probs_pos = []
    for wav in pos_wavs:
        pcm = load_wav_as_int16(wav)
        if len(pcm) == 0:
            continue
        triggered, prob = test_file(detector, pcm)
        probs_pos.append(prob)
        flag = "✓" if triggered else "✗"
        print(f"  [{flag}] {wav.name:30s}  max_prob={prob:.3f}")
        if triggered:
            tp += 1
        else:
            fp_pos += 1

    total_pos = tp + fp_pos
    recall = tp / total_pos * 100 if total_pos > 0 else 0
    print(f"\n召回率: {tp}/{total_pos} = {recall:.1f}%")
    print(f"概率分布: min={min(probs_pos):.3f}  max={max(probs_pos):.3f}  "
          f"avg={sum(probs_pos)/len(probs_pos):.3f}")

    # ── 负样本测试（误触发率） ────────────────────────────────────────────────
    if args.neg:
        neg_dir = REPO_ROOT / args.neg
        neg_wavs = sorted(neg_dir.rglob("*.wav"))[:args.max_neg]
        if not neg_wavs:
            print(f"\n[warn] 负样本目录无 WAV: {neg_dir}")
        else:
            print(f"\n=== 负样本测试（误触发率）: {len(neg_wavs)} 条 ===")
            fp_neg = tn = 0
            probs_neg = []
            for wav in neg_wavs:
                pcm = load_wav_as_int16(wav)
                if len(pcm) == 0:
                    continue
                triggered, prob = test_file(detector, pcm)
                probs_neg.append(prob)
                if triggered:
                    fp_neg += 1
                    print(f"  [误触发] {wav.name}  max_prob={prob:.3f}")
                else:
                    tn += 1

            total_neg = fp_neg + tn
            fpr = fp_neg / total_neg * 100 if total_neg > 0 else 0
            print(f"\n误触发率: {fp_neg}/{total_neg} = {fpr:.1f}%")
            if probs_neg:
                print(f"概率分布: min={min(probs_neg):.3f}  max={max(probs_neg):.3f}  "
                      f"avg={sum(probs_neg)/len(probs_neg):.3f}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"汇总  cutoff={args.cutoff}  window={args.window}")
    print(f"  召回率: {recall:.1f}%  ({tp}/{total_pos})")
    if args.neg and neg_wavs:
        print(f"  误触发率: {fpr:.1f}%  ({fp_neg}/{total_neg})")
    print()
    if recall < 50:
        print("⚠  召回率偏低，建议：")
        print("   1. 降低 --cutoff（如 0.05）")
        print("   2. 降低 --window（如 1 或 2）")
        print("   3. 增加真实人声样本后重新训练")
    elif recall >= 80:
        print("✓  召回率良好")


if __name__ == "__main__":
    main()

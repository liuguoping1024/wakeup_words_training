#!/usr/bin/env python3
"""
扫描正样本目录，用模型检测每个文件的 max_prob。
找出概率极低的可疑样本（可能是错误混入的负样本）。
在 wakeword-mww Docker 内运行。
"""
import argparse, os, sys, wave, json
import numpy as np

sys.path.insert(0, "/workspace/inference")
from runtime import WakeWordDetector, SAMPLE_RATE


def load_wav_16k(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio[::ch]
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio.astype(np.float32), int(len(audio) * 16000 / sr)).astype(np.int16)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="低于此概率的样本视为可疑")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多扫描多少文件（0=全部）")
    parser.add_argument("--sample", type=int, default=500,
                        help="随机抽样数量（加速扫描）")
    args = parser.parse_args()

    import random
    random.seed(42)

    all_wavs = sorted([f for f in os.listdir(args.pos_dir) if f.endswith(".wav")])
    total = len(all_wavs)
    print(f"正样本目录: {args.pos_dir} ({total} files)")

    # 随机抽样
    if args.sample > 0 and args.sample < total:
        wavs = random.sample(all_wavs, args.sample)
        print(f"随机抽样: {len(wavs)} / {total}")
    else:
        wavs = all_wavs

    suspicious = []
    good = 0
    errors = 0

    for i, fname in enumerate(wavs):
        path = os.path.join(args.pos_dir, fname)
        try:
            detector = WakeWordDetector(args.model, cutoff=0.5, window_count=1)
            audio = load_wav_16k(path)
            scores = detector.feed_and_score(audio)
            max_prob = max(scores) if scores else 0.0
        except Exception as e:
            errors += 1
            continue

        if max_prob < args.threshold:
            suspicious.append((fname, max_prob))
        else:
            good += 1

        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(wavs)}, suspicious={len(suspicious)}, good={good}")

    print(f"\n=== 结果 ===")
    print(f"扫描: {len(wavs)}, 正常: {good}, 可疑(prob<{args.threshold}): {len(suspicious)}, 错误: {errors}")

    if suspicious:
        # 按来源分组统计
        by_source = {}
        for fname, prob in suspicious:
            if fname.startswith("cosy"):
                src = "CosyVoice"
            elif fname.startswith("piper"):
                src = "Piper"
            elif fname.startswith("edge"):
                src = "edge-tts"
            else:
                src = "other"
            by_source.setdefault(src, []).append((fname, prob))

        print(f"\n可疑样本按来源:")
        for src, items in sorted(by_source.items()):
            print(f"  {src}: {len(items)} 条")

        print(f"\n概率最低的 20 条:")
        for fname, prob in sorted(suspicious, key=lambda x: x[1])[:20]:
            print(f"  {prob:.4f}  {fname}")


if __name__ == "__main__":
    main()

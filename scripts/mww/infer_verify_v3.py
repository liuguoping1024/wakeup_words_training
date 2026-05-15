#!/usr/bin/env python3
"""
用 inference/runtime.py 的 WakeWordDetector 对测试音频跑推理。
这样和 ARM 设备上的推理完全一致。

在 wakeword-mww Docker 内运行（需要 pymicro_features）。
"""
import argparse, json, os, sys, wave
import numpy as np

# 把 inference 目录加入 path
sys.path.insert(0, "/workspace/inference")
from runtime import WakeWordDetector, SAMPLE_RATE, STRIDE_SAMPLES


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
        audio_f = audio.astype(np.float32)
        audio = resample(audio_f, int(len(audio_f) * 16000 / sr)).astype(np.int16)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff", type=float, default=0.88)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    with open(os.path.join(args.test_dir, "manifest.json")) as f:
        manifest = json.load(f)

    results = []
    for item in manifest:
        wav_path = os.path.join(args.test_dir, item["file"])
        if not os.path.exists(wav_path):
            continue

        # 每个文件重新创建 detector（重置状态，和 ARM 设备一致）
        detector = WakeWordDetector(
            args.model, cutoff=args.cutoff, window_count=args.window
        )

        audio = load_wav_16k(wav_path)
        scores = detector.feed_and_score(audio)

        max_prob = max(scores) if scores else 0.0

        # 滑窗检测
        from collections import deque
        win = deque(maxlen=args.window)
        detected = False
        for s in scores:
            win.append(s)
            if len(win) == args.window and all(v >= args.cutoff for v in win):
                detected = True
                break

        item["max_prob"] = round(max_prob, 4)
        item["detected"] = detected
        item["scores_summary"] = f"len={len(scores)}, max={max_prob:.3f}"
        results.append(item)

    # 打印
    print(f"模型: {args.model}")
    print(f"cutoff={args.cutoff}, window={args.window}")
    print("=" * 70)
    print(f"{'文件':<20} {'类别':<10} {'max_prob':>8} {'检测':>6} {'判定':>8}")
    print("=" * 70)

    errors = []
    for r in sorted(results, key=lambda x: x["file"]):
        cat = r["category"]
        det = r["detected"]
        mp = r["max_prob"]

        if cat in ("positive", "dialect"):
            mark = "✓" if det else "✗漏报"
            correct = det
        else:
            mark = "✓" if not det else "✗误触"
            correct = not det

        if not correct:
            errors.append(r)

        print(f"{r['file']:<20} {cat:<10} {mp:>8.4f} {'是' if det else '否':>6} {mark:>8}")

    print("=" * 70)
    for cat in ["positive", "dialect", "negative"]:
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        if cat in ("positive", "dialect"):
            hits = sum(1 for r in items if r["detected"])
            print(f"  {cat}: {hits}/{len(items)} 触发 (recall={hits/len(items)*100:.1f}%)")
        else:
            fps = sum(1 for r in items if r["detected"])
            print(f"  {cat}: {fps}/{len(items)} 误触发 (FPR={fps/len(items)*100:.1f}%)")

    print(f"\n总计: {len(results)} 样本, 错误: {len(errors)}")

    with open(os.path.join(args.test_dir, "results_v3.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

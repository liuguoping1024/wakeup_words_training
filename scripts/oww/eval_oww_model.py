#!/usr/bin/env python3
"""
openWakeWord 模型评估脚本。

评估指标：
  1. 正样本召回率（Recall）：真实录音中能正确检测到唤醒词的比例
  2. 误触发率（FPR）：负样本中错误触发的比例
  3. 每小时误触发次数（FP/Hour）：在连续语音中的误触发频率

用法（Docker 内）：
  python3 eval_oww_model.py \
    --model /workspace/outputs/oww/nihao_shushi.onnx \
    --pos /workspace/data/positive_raw/nihao_shushi \
    --neg /workspace/data/negative_datasets/dinner_party \
    --threshold 0.5 \
    --max-pos 500
"""
import argparse
import logging
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def load_wav_int16(path):
    """读取 WAV 文件为 16-bit int16 数组。"""
    try:
        with wave.open(str(path), "rb") as f:
            sr = f.getframerate()
            ch = f.getnchannels()
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
        if sr != 16000:
            # 简单重采样
            new_len = int(len(data) * 16000 / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_len),
                np.arange(len(data)), data.astype(np.float32),
            ).astype(np.int16)
        return data
    except Exception as e:
        # 尝试用 soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
            return (audio * 32767).clip(-32768, 32767).astype(np.int16)
        except Exception as e2:
            log.warning(f"无法读取 {path}: {e2}")
            return np.zeros(0, dtype=np.int16)


def eval_positive(model, pos_dir, threshold, max_samples, model_name):
    """评估正样本召回率。"""
    wavs = sorted(Path(pos_dir).glob("*.wav"))
    if not wavs:
        # 尝试递归搜索
        wavs = sorted(Path(pos_dir).rglob("*.wav"))
    if not wavs:
        log.error(f"没有找到 WAV 文件: {pos_dir}")
        return

    # 随机抽样
    rng = np.random.RandomState(42)
    if len(wavs) > max_samples:
        indices = rng.choice(len(wavs), max_samples, replace=False)
        wavs = [wavs[i] for i in sorted(indices)]

    log.info(f"正样本测试: {len(wavs)} 条 (threshold={threshold})")

    tp = 0
    fn = 0
    max_scores = []
    failed = 0

    for i, wav in enumerate(wavs):
        pcm = load_wav_int16(wav)
        if len(pcm) < 1600:  # 太短，跳过
            failed += 1
            continue

        # 用 predict_clip 模拟流式推理
        predictions = model.predict_clip(str(wav) if wav.suffix == ".wav" else pcm, padding=1)

        # 提取该模型的最大分数
        scores = []
        for pred in predictions:
            if model_name in pred:
                scores.append(pred[model_name])

        if not scores:
            fn += 1
            max_scores.append(0.0)
            continue

        max_score = max(scores)
        max_scores.append(max_score)

        if max_score >= threshold:
            tp += 1
        else:
            fn += 1

        if (i + 1) % 100 == 0:
            current_recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            log.info(f"  进度: {i+1}/{len(wavs)}, 当前召回率: {current_recall:.1f}%")

    total = tp + fn
    recall = tp / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"正样本结果 (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  召回率: {tp}/{total} = {recall:.1f}%")
    if max_scores:
        print(f"  分数: min={min(max_scores):.4f}  max={max(max_scores):.4f}  "
              f"avg={np.mean(max_scores):.4f}  median={np.median(max_scores):.4f}")
        # 分数分布
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            above = sum(1 for s in max_scores if s >= t)
            print(f"    >= {t:.1f}: {above}/{total} ({above/total*100:.1f}%)")
    if failed > 0:
        print(f"  跳过: {failed} 条（文件太短或无法读取）")

    return {"recall": recall, "tp": tp, "fn": fn, "total": total,
            "scores": max_scores}


def eval_negative_clips(model, neg_dir, threshold, max_samples, model_name):
    """评估负样本误触发率（短音频片段）。"""
    wavs = sorted(Path(neg_dir).rglob("*.wav"))
    if not wavs:
        log.warning(f"没有找到负样本: {neg_dir}")
        return None

    rng = np.random.RandomState(42)
    if len(wavs) > max_samples:
        indices = rng.choice(len(wavs), max_samples, replace=False)
        wavs = [wavs[i] for i in sorted(indices)]

    log.info(f"负样本测试: {len(wavs)} 条 (threshold={threshold})")

    fp = 0
    tn = 0
    fp_files = []

    for i, wav in enumerate(wavs):
        pcm = load_wav_int16(wav)
        if len(pcm) < 1600:
            continue

        predictions = model.predict_clip(str(wav) if wav.suffix == ".wav" else pcm, padding=0)
        scores = [pred.get(model_name, 0) for pred in predictions]
        max_score = max(scores) if scores else 0

        if max_score >= threshold:
            fp += 1
            fp_files.append((wav.name, max_score))
        else:
            tn += 1

        if (i + 1) % 100 == 0:
            log.info(f"  进度: {i+1}/{len(wavs)}")

    total = fp + tn
    fpr = fp / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"负样本结果 (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  误触发率: {fp}/{total} = {fpr:.1f}%")
    if fp_files:
        print(f"  误触发文件 (前20):")
        for name, score in fp_files[:20]:
            print(f"    {name}: {score:.4f}")

    return {"fpr": fpr, "fp": fp, "tn": tn, "total": total}


def eval_continuous_audio(model, audio_dir, threshold, model_name):
    """评估连续长音频的每小时误触发次数。"""
    wavs = sorted(Path(audio_dir).rglob("*.wav"))
    if not wavs:
        log.warning(f"没有找到连续音频: {audio_dir}")
        return None

    log.info(f"连续音频测试: {len(wavs)} 个文件 (threshold={threshold})")

    total_seconds = 0
    total_fp = 0
    chunk_size = 1280  # 80ms

    for wav_path in wavs:
        pcm = load_wav_int16(wav_path)
        if len(pcm) < chunk_size:
            continue

        duration = len(pcm) / 16000
        total_seconds += duration

        # 流式推理
        model.reset()
        fp_in_file = 0
        for i in range(0, len(pcm) - chunk_size, chunk_size):
            chunk = pcm[i:i+chunk_size]
            pred = model.predict(chunk)
            score = pred.get(model_name, 0)
            if score >= threshold:
                fp_in_file += 1

        total_fp += fp_in_file
        if fp_in_file > 0:
            log.info(f"  {wav_path.name}: {fp_in_file} FP in {duration:.1f}s")

    total_hours = total_seconds / 3600
    fp_per_hour = total_fp / total_hours if total_hours > 0 else 0

    print(f"\n{'='*60}")
    print(f"连续音频结果 (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  总时长: {total_hours:.2f} 小时 ({total_seconds:.0f} 秒)")
    print(f"  误触发: {total_fp} 次")
    print(f"  FP/Hour: {fp_per_hour:.2f}")

    return {"fp_per_hour": fp_per_hour, "total_fp": total_fp,
            "total_hours": total_hours}


def main():
    parser = argparse.ArgumentParser(description="OWW 模型评估")
    parser.add_argument("--model", required=True, help="ONNX 模型路径")
    parser.add_argument("--pos", required=True, help="正样本目录")
    parser.add_argument("--neg", default=None, help="负样本目录（短音频片段）")
    parser.add_argument("--continuous", default=None, help="连续长音频目录（计算 FP/Hour）")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-pos", type=int, default=500)
    parser.add_argument("--max-neg", type=int, default=500)
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        log.error(f"模型不存在: {model_path}")
        sys.exit(1)

    # 从文件名推断模型名
    model_name = Path(model_path).stem
    log.info(f"模型: {model_path}")
    log.info(f"模型名: {model_name}")

    # 加载 OWW 模型
    from openwakeword.model import Model
    oww_model = Model(
        wakeword_models=[model_path],
        inference_framework="onnx",
    )

    print(f"\n{'#'*60}")
    print(f"  OWW 模型评估: {model_name}")
    print(f"  模型: {model_path}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'#'*60}")

    # 1. 正样本评估
    pos_result = eval_positive(oww_model, args.pos, args.threshold, args.max_pos, model_name)

    # 2. 负样本评估（短片段）
    neg_result = None
    if args.neg and os.path.exists(args.neg):
        neg_result = eval_negative_clips(oww_model, args.neg, args.threshold, args.max_neg, model_name)

    # 3. 连续音频评估（FP/Hour）
    cont_result = None
    if args.continuous and os.path.exists(args.continuous):
        cont_result = eval_continuous_audio(oww_model, args.continuous, args.threshold, model_name)

    # 汇总
    print(f"\n{'#'*60}")
    print(f"  汇总 (threshold={args.threshold})")
    print(f"{'#'*60}")
    if pos_result:
        print(f"  召回率: {pos_result['recall']:.1f}% ({pos_result['tp']}/{pos_result['total']})")
    if neg_result:
        print(f"  误触发率: {neg_result['fpr']:.1f}% ({neg_result['fp']}/{neg_result['total']})")
    if cont_result:
        print(f"  FP/Hour: {cont_result['fp_per_hour']:.2f} ({cont_result['total_fp']} in {cont_result['total_hours']:.2f}h)")

    # 多阈值对比
    if pos_result and pos_result["scores"]:
        print(f"\n{'='*60}")
        print(f"  多阈值对比")
        print(f"{'='*60}")
        print(f"  {'Threshold':>10s}  {'Recall':>8s}  {'Count':>10s}")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            above = sum(1 for s in pos_result["scores"] if s >= t)
            total = pos_result["total"]
            r = above / total * 100 if total > 0 else 0
            print(f"  {t:>10.1f}  {r:>7.1f}%  {above:>5d}/{total}")


if __name__ == "__main__":
    main()

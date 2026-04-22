#!/usr/bin/env python3
"""
OWW 中文唤醒词训练 v3：混合数据源。

正样本来源（三管齐下）：
  1. 真实录音 (data/positive_raw/nihao_shushi) — 99,454 条，40 位说话人
  2. Piper huayan TTS — 单说话人，已在 outputs/oww/nihao_shushi/ 生成
  3. edge-tts 19 种中文声音 — 在宿主机预生成

负样本来源：
  1. edge-tts 对抗性负样本（发音相近的中文短语）
  2. ACAV100M 预计算特征（英文语音，作为通用负样本）

用法（Docker 内）：
  python3 train_zh_mixed.py --keyword-id nihao_shushi \
    --real-dir /workspace/data/positive_raw/nihao_shushi \
    --edge-tts-dir /workspace/outputs/oww/nihao_shushi_v3/edge_tts_positive \
    --piper-dir /workspace/outputs/oww/nihao_shushi/positive_train \
    --neg-train-dir /workspace/outputs/oww/nihao_shushi_v3/negative_train \
    --neg-test-dir /workspace/outputs/oww/nihao_shushi_v3/negative_test
"""
import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_wav_16k(path):
    """读取 WAV 并转为 16kHz mono int16。"""
    try:
        sr, data = scipy.io.wavfile.read(str(path))
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            audio = data
        else:
            audio = data.astype(np.float32) / np.iinfo(data.dtype).max
    except Exception:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000)
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def collect_wavs(directory, max_n=None, shuffle=True, seed=42):
    """收集目录下的 WAV 文件路径。"""
    wavs = sorted(Path(directory).glob("*.wav"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(wavs)
    if max_n:
        wavs = wavs[:max_n]
    return wavs


def prepare_mixed_positive(real_dir, edge_tts_dir, piper_dir, output_train_dir, output_test_dir,
                            n_real=15000, n_edge=8000, n_piper=2000, n_test=3000, seed=42):
    """混合三种来源的正样本。"""
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    existing_train = len(list(Path(output_train_dir).glob("*.wav")))
    existing_test = len(list(Path(output_test_dir).glob("*.wav")))
    total_target = n_real + n_edge + n_piper
    if existing_train >= total_target * 0.9 and existing_test >= n_test * 0.9:
        log.info(f"混合正样本已存在: {existing_train} train, {existing_test} test")
        return

    rng = random.Random(seed)

    # 收集各来源
    real_wavs = collect_wavs(real_dir, seed=seed) if os.path.exists(real_dir) else []
    edge_wavs = collect_wavs(edge_tts_dir, seed=seed) if os.path.exists(edge_tts_dir) else []
    piper_wavs = collect_wavs(piper_dir, seed=seed) if os.path.exists(piper_dir) else []

    log.info(f"数据源: 真实录音={len(real_wavs)}, edge-tts={len(edge_wavs)}, piper={len(piper_wavs)}")

    # 划分测试集（从真实录音中取）
    test_wavs = real_wavs[:n_test]
    real_train = real_wavs[n_test:n_test + n_real]
    edge_train = edge_wavs[:n_edge]
    piper_train = piper_wavs[:n_piper]

    all_train = []
    for src, wavs, label in [("real", real_train, "real"),
                               ("edge", edge_train, "edge"),
                               ("piper", piper_train, "piper")]:
        for w in wavs:
            all_train.append((w, label))

    rng.shuffle(all_train)
    log.info(f"混合训练集: {len(all_train)} 条 (real={len(real_train)}, edge={len(edge_train)}, piper={len(piper_train)})")
    log.info(f"测试集: {len(test_wavs)} 条 (真实录音)")

    # 写入训练集
    for i, (wav, label) in enumerate(tqdm(all_train, desc="写入训练集")):
        out = os.path.join(output_train_dir, f"{label}_{i:06d}.wav")
        if os.path.exists(out):
            continue
        try:
            pcm = load_wav_16k(wav)
            scipy.io.wavfile.write(out, 16000, pcm)
        except Exception as e:
            log.warning(f"跳过 {wav}: {e}")

    # 写入测试集
    for i, wav in enumerate(tqdm(test_wavs, desc="写入测试集")):
        out = os.path.join(output_test_dir, f"real_{i:06d}.wav")
        if os.path.exists(out):
            continue
        try:
            pcm = load_wav_16k(wav)
            scipy.io.wavfile.write(out, 16000, pcm)
        except Exception as e:
            log.warning(f"跳过 {wav}: {e}")

    final_train = len(list(Path(output_train_dir).glob("*.wav")))
    final_test = len(list(Path(output_test_dir).glob("*.wav")))
    log.info(f"最终: {final_train} train, {final_test} test")


def compute_clip_length(clip_dir, n_sample=100):
    clips = sorted(Path(clip_dir).glob("*.wav"))[:n_sample]
    durations = []
    for c in clips:
        sr, dat = scipy.io.wavfile.read(str(c))
        durations.append(len(dat))
    median_dur = int(np.median(durations))
    total_length = int(round(median_dur / 1000) * 1000) + 12000
    total_length = max(total_length, 32000)
    if abs(total_length - 32000) <= 4000:
        total_length = 32000
    log.info(f"clip total_length: {total_length} ({total_length/16000:.2f}s)")
    return total_length


def augment_and_features(clip_dir, output_npy, total_length, augment_rounds,
                          background_paths, rir_paths, batch_size=16):
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    clips = [str(p) for p in sorted(Path(clip_dir).glob("*.wav"))]
    if not clips:
        log.warning(f"空目录: {clip_dir}")
        return
    all_clips = clips * max(1, augment_rounds)
    log.info(f"增强: {len(clips)} × {augment_rounds} = {len(all_clips)}")

    gen = augment_clips(all_clips, total_length=total_length, batch_size=batch_size,
                         background_clip_paths=background_paths, RIR_paths=rir_paths)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    ncpu = 1 if device == "gpu" else max(1, (os.cpu_count() or 2) // 2)
    compute_features_from_generator(gen, n_total=len(all_clips), clip_duration=total_length,
                                     output_file=output_npy, device=device, ncpu=ncpu)
    log.info(f"特征: {output_npy} -> {np.load(output_npy, mmap_mode='r').shape}")


def train_model(args, model_dir):
    from openwakeword.train import Model, mmap_batch_generator

    pos_feat_train = os.path.join(model_dir, "positive_features_train.npy")
    pos_feat_test = os.path.join(model_dir, "positive_features_test.npy")
    neg_feat_train = os.path.join(model_dir, "negative_features_train.npy")
    neg_feat_test = os.path.join(model_dir, "negative_features_test.npy")

    input_shape = np.load(pos_feat_test, mmap_mode='r').shape[1:]
    log.info(f"input_shape: {input_shape}")

    for name, path in [("pos_train", pos_feat_train), ("pos_test", pos_feat_test),
                        ("neg_train", neg_feat_train), ("neg_test", neg_feat_test)]:
        if os.path.exists(path):
            log.info(f"  {name}: {np.load(path, mmap_mode='r').shape}")

    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=args.layer_size, n_blocks=args.n_blocks,
                seconds_per_example=1280 * input_shape[0] / 16000)

    def reshape_fn(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    feature_data_files = {"positive": pos_feat_train}
    batch_n_per_class = {"positive": 200}
    data_transforms = {}
    label_transforms = {"positive": lambda x: [1 for _ in x]}

    # ACAV100M — 降低比例，避免压制中文正样本
    if os.path.exists(args.acav_features):
        feature_data_files["ACAV100M"] = args.acav_features
        batch_n_per_class["ACAV100M"] = 100
        data_transforms["ACAV100M"] = reshape_fn
        label_transforms["ACAV100M"] = lambda x: [0 for _ in x]

    # 对抗性负样本
    if os.path.exists(neg_feat_train):
        feature_data_files["adversarial_negative"] = neg_feat_train
        batch_n_per_class["adversarial_negative"] = 128
        label_transforms["adversarial_negative"] = lambda x: [0 for _ in x]

    log.info(f"batch_n_per_class: {batch_n_per_class}")
    # 正样本占比: 200/(200+100+128) = 46.7%

    batch_generator = mmap_batch_generator(
        feature_data_files, n_per_class=batch_n_per_class,
        data_transform_funcs=data_transforms, label_transform_funcs=label_transforms)

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, gen): self.gen = gen
        def __iter__(self): return self.gen

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(IterDS(batch_generator), batch_size=None,
                                           num_workers=n_cpus, prefetch_factor=16)

    X_val_fp = np.load(args.fp_val_data)
    X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])
    X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
    X_val_fp_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
        batch_size=len(X_val_fp_labels))

    X_val_pos = np.load(pos_feat_test)
    X_val_neg = np.load(neg_feat_test) if os.path.exists(neg_feat_test) else np.zeros((100, *input_shape), dtype=np.float32)
    labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)
    X_val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((X_val_pos, X_val_neg))), torch.from_numpy(labels)),
        batch_size=len(labels))

    log.info(f"训练: {args.steps} steps, layer={args.layer_size}, blocks={args.n_blocks}, penalty={args.max_neg_weight}")
    best_model = oww.auto_train(
        X_train=X_train, X_val=X_val_loader, false_positive_val_data=X_val_fp_loader,
        steps=args.steps, max_negative_weight=args.max_neg_weight, target_fp_per_hour=1.0)

    oww.export_model(model=best_model, model_name=args.keyword_id, output_dir=args.output_dir)
    log.info(f"模型: {args.output_dir}/{args.keyword_id}.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-id", default="nihao_shushi")
    parser.add_argument("--real-dir", default="/workspace/data/positive_raw/nihao_shushi")
    parser.add_argument("--edge-tts-dir", default="/workspace/outputs/oww/nihao_shushi_v3/edge_tts_positive")
    parser.add_argument("--piper-dir", default="/workspace/outputs/oww/nihao_shushi/positive_train")
    parser.add_argument("--neg-train-dir", default="/workspace/outputs/oww/nihao_shushi_v3/negative_train")
    parser.add_argument("--neg-test-dir", default="/workspace/outputs/oww/nihao_shushi_v3/negative_test")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--acav-features", default="/workspace/data/oww/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    parser.add_argument("--fp-val-data", default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--background-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k", "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--n-real", type=int, default=15000)
    parser.add_argument("--n-edge", type=int, default=8000)
    parser.add_argument("--n-piper", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--augment-rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=115000)
    parser.add_argument("--layer-size", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--max-neg-weight", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, "/workspace/work/openWakeWord")

    model_dir = os.path.join(args.output_dir, args.keyword_id + "_v3")
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")

    # Step 1: 混合正样本
    log.info("=" * 60)
    log.info("Step 1: 混合正样本 (真实录音 + edge-tts + piper)")
    log.info("=" * 60)
    prepare_mixed_positive(args.real_dir, args.edge_tts_dir, args.piper_dir,
                            train_dir, test_dir,
                            n_real=args.n_real, n_edge=args.n_edge, n_piper=args.n_piper,
                            n_test=args.n_test, seed=args.seed)

    # Step 2: 复制负样本
    log.info("=" * 60)
    log.info("Step 2: 检查负样本")
    log.info("=" * 60)
    neg_train = os.path.join(model_dir, "negative_train")
    neg_test = os.path.join(model_dir, "negative_test")
    if not os.path.exists(neg_train) or len(list(Path(neg_train).glob("*.wav"))) == 0:
        if os.path.exists(args.neg_train_dir):
            os.symlink(os.path.abspath(args.neg_train_dir), neg_train) if not os.path.exists(neg_train) else None
        else:
            os.makedirs(neg_train, exist_ok=True)
    if not os.path.exists(neg_test) or len(list(Path(neg_test).glob("*.wav"))) == 0:
        if os.path.exists(args.neg_test_dir):
            os.symlink(os.path.abspath(args.neg_test_dir), neg_test) if not os.path.exists(neg_test) else None
        else:
            os.makedirs(neg_test, exist_ok=True)

    neg_train_count = len(list(Path(neg_train).glob("*.wav")))
    neg_test_count = len(list(Path(neg_test).glob("*.wav")))
    log.info(f"负样本: {neg_train_count} train, {neg_test_count} test")

    # Step 3: 增强 + 特征
    log.info("=" * 60)
    log.info("Step 3: 数据增强 + 特征计算")
    log.info("=" * 60)

    rir_paths = [i.path for i in os.scandir(args.rir_dir)] if os.path.exists(args.rir_dir) else []
    background_paths = []
    for bg_dir in args.background_dirs:
        if os.path.exists(bg_dir):
            background_paths.extend([i.path for i in os.scandir(bg_dir)])

    total_length = compute_clip_length(test_dir)

    if args.overwrite:
        for f in Path(model_dir).glob("*_features_*.npy"):
            f.unlink()

    for name, clip_dir, rounds in [
        ("positive_features_train", train_dir, args.augment_rounds),
        ("positive_features_test", test_dir, 1),
        ("negative_features_train", neg_train, args.augment_rounds),
        ("negative_features_test", neg_test, 1),
    ]:
        npy_path = os.path.join(model_dir, f"{name}.npy")
        if not os.path.exists(npy_path):
            n_wavs = len(list(Path(clip_dir).glob("*.wav")))
            if n_wavs > 0:
                log.info(f"计算 {name} ({n_wavs} clips × {rounds} rounds)...")
                augment_and_features(clip_dir, npy_path, total_length, rounds,
                                      background_paths, rir_paths, args.batch_size)
            else:
                log.warning(f"跳过 {name}: 无 WAV 文件")
        else:
            log.info(f"{name} 已存在: {np.load(npy_path, mmap_mode='r').shape}")

    # Step 4: 训练
    log.info("=" * 60)
    log.info("Step 4: 训练")
    log.info("=" * 60)

    # 把 model_dir 下的特征路径传给 train
    args.output_dir_orig = args.output_dir
    args.output_dir = os.path.dirname(model_dir)  # outputs/oww
    args.keyword_id_orig = args.keyword_id
    args.keyword_id = args.keyword_id + "_v3"
    train_model(args, model_dir)

    # 复制最终模型到标准位置
    src = os.path.join(args.output_dir, args.keyword_id + ".onnx")
    dst = os.path.join(args.output_dir, args.keyword_id_orig + ".onnx")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        log.info(f"模型复制: {src} -> {dst}")

    log.info("=" * 60)
    log.info("全部完成!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

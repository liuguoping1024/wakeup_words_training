#!/usr/bin/env python3
"""
OWW 中文唤醒词训练脚本（Docker 内运行）。

跳过 Piper TTS 生成，直接使用：
  - 正样本：真实录音（data/positive_raw/<keyword_id>/）
  - 对抗性负样本：宿主机预生成的 edge-tts 音频
  - 背景负样本：ACAV100M 预计算特征
  - 数据增强：OWW 标准 augment pipeline（RIR + 背景噪声 + 音频效果）

与原 train_nihao_oww.py 的区别：
  1. 更多正样本（15k train vs 10k）
  2. 更多增强轮数（2 vs 1）
  3. 更大模型（layer_size=64 vs 32）
  4. 更多训练步数（80k vs 50k）
  5. 清理了代码结构，去掉了容器内 TTS 生成的死代码

用法（在 Docker 内）：
  python3 -u train_zh_oww.py \
    --keyword-id nihao_shushi \
    --positive-dir /workspace/data/positive_raw/nihao_shushi \
    --steps 80000 --layer-size 64 --augment-rounds 2
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
    """读取 WAV 并转为 16kHz mono float32。"""
    sr, data = scipy.io.wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import resampy
        data = resampy.resample(data, sr, 16000)
    return data


def write_wav_16k(path, audio):
    """写入 16kHz 16-bit PCM WAV。"""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(str(path), 16000, pcm)


def prepare_positive_clips(src_dir, train_dir, test_dir, n_train, n_test, seed=42):
    """从真实录音中划分 train/test 集。"""
    wavs = sorted(Path(src_dir).glob("*.wav"))
    if not wavs:
        log.error(f"没有找到 WAV 文件: {src_dir}")
        sys.exit(1)

    log.info(f"真实录音: {len(wavs)} 条")

    # 检查是否已经准备好
    existing_train = len(list(Path(train_dir).glob("*.wav"))) if os.path.exists(train_dir) else 0
    existing_test = len(list(Path(test_dir).glob("*.wav"))) if os.path.exists(test_dir) else 0
    if existing_train >= n_train * 0.95 and existing_test >= n_test * 0.95:
        log.info(f"正样本已准备好: {existing_train} train, {existing_test} test")
        return

    rng = random.Random(seed)
    shuffled = list(wavs)
    rng.shuffle(shuffled)

    n_test = min(n_test, len(shuffled) // 5)
    test_wavs = shuffled[:n_test]
    train_wavs = shuffled[n_test:n_test + n_train]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    log.info(f"准备正样本: {len(train_wavs)} train + {len(test_wavs)} test")

    for i, wav in enumerate(tqdm(train_wavs, desc="copy train")):
        out = os.path.join(train_dir, f"pos_{i:06d}.wav")
        if os.path.exists(out):
            continue
        audio = load_wav_16k(wav)
        write_wav_16k(out, audio)

    for i, wav in enumerate(tqdm(test_wavs, desc="copy test")):
        out = os.path.join(test_dir, f"pos_{i:06d}.wav")
        if os.path.exists(out):
            continue
        audio = load_wav_16k(wav)
        write_wav_16k(out, audio)

    log.info(f"正样本准备完成: {len(os.listdir(train_dir))} train, {len(os.listdir(test_dir))} test")


def compute_clip_length(clip_dir, n_sample=100):
    """根据样本时长计算 total_length（采样点数）。"""
    clips = sorted(Path(clip_dir).glob("*.wav"))[:n_sample]
    durations = []
    for c in clips:
        sr, dat = scipy.io.wavfile.read(str(c))
        durations.append(len(dat))

    median_dur = int(np.median(durations))
    # 加 750ms buffer，向上取整到 1000
    total_length = int(round(median_dur / 1000) * 1000) + 12000
    total_length = max(total_length, 32000)
    if abs(total_length - 32000) <= 4000:
        total_length = 32000

    log.info(f"clip total_length: {total_length} samples ({total_length/16000:.2f}s), "
             f"median raw: {median_dur} ({median_dur/16000:.2f}s)")
    return total_length


def augment_and_compute_features(clip_dir, output_npy, total_length, augment_rounds,
                                  background_paths, rir_paths, batch_size=16):
    """增强音频并计算 OWW 特征。"""
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    clips = [str(p) for p in sorted(Path(clip_dir).glob("*.wav"))]
    if not clips:
        log.warning(f"目录为空，跳过: {clip_dir}")
        return

    # 重复 clips 以实现多轮增强
    all_clips = clips * max(1, augment_rounds)
    log.info(f"增强: {len(clips)} clips × {max(1, augment_rounds)} rounds = {len(all_clips)} total")

    gen = augment_clips(
        all_clips,
        total_length=total_length,
        batch_size=batch_size,
        background_clip_paths=background_paths,
        RIR_paths=rir_paths,
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"
    ncpu = 1 if device == "gpu" else max(1, (os.cpu_count() or 2) // 2)

    compute_features_from_generator(
        gen,
        n_total=len(all_clips),
        clip_duration=total_length,
        output_file=output_npy,
        device=device,
        ncpu=ncpu,
    )
    log.info(f"特征保存: {output_npy} -> {np.load(output_npy, mmap_mode='r').shape}")


def train_model(args, model_dir):
    """训练 OWW 模型。"""
    from openwakeword.train import Model, mmap_batch_generator

    pos_feat_train = os.path.join(model_dir, "positive_features_train.npy")
    pos_feat_test = os.path.join(model_dir, "positive_features_test.npy")
    neg_feat_train = os.path.join(model_dir, "negative_features_train.npy")
    neg_feat_test = os.path.join(model_dir, "negative_features_test.npy")

    # 检查特征文件
    for f in [pos_feat_train, pos_feat_test]:
        if not os.path.exists(f):
            log.error(f"缺少特征文件: {f}")
            sys.exit(1)

    input_shape = np.load(pos_feat_test, mmap_mode='r').shape[1:]
    log.info(f"input_shape: {input_shape}")

    # 打印数据统计
    for name, path in [("pos_train", pos_feat_train), ("pos_test", pos_feat_test),
                        ("neg_train", neg_feat_train), ("neg_test", neg_feat_test)]:
        if os.path.exists(path):
            shape = np.load(path, mmap_mode='r').shape
            log.info(f"  {name}: {shape}")
        else:
            log.warning(f"  {name}: 不存在")

    # 创建模型
    oww = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type="dnn",
        layer_dim=args.layer_size,
        seconds_per_example=1280 * input_shape[0] / 16000,
    )

    # 数据变换：确保负样本 shape 匹配
    def reshape_fn(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    # 构建特征文件字典和批次配置
    # 正样本 batch 比例提高到 150（默认 50 太少，被负样本淹没）
    feature_data_files = {
        "positive": pos_feat_train,
    }
    batch_n_per_class = {
        "positive": 150,
    }
    data_transforms = {}
    label_transforms = {
        "positive": lambda x: [1 for _ in x],
    }

    # ACAV100M 负样本
    # 注意：ACAV100M 是英文语音特征，对中文唤醒词来说分布差异很大。
    # 如果 batch 比例太高（如 1024），模型会学到"把所有中文都判为负"。
    # 降低到 256，让正样本有足够的学习信号。
    if os.path.exists(args.acav_features):
        feature_data_files["ACAV100M"] = args.acav_features
        batch_n_per_class["ACAV100M"] = 256
        data_transforms["ACAV100M"] = reshape_fn
        label_transforms["ACAV100M"] = lambda x: [0 for _ in x]
        log.info(f"ACAV100M 负样本: {np.load(args.acav_features, mmap_mode='r').shape}")
    else:
        log.warning("ACAV100M 特征文件不存在，仅使用对抗性负样本")

    # 对抗性负样本（中文发音相近的短语）
    if os.path.exists(neg_feat_train):
        feature_data_files["adversarial_negative"] = neg_feat_train
        neg_shape = np.load(neg_feat_train, mmap_mode='r').shape
        # 对抗性负样本比正样本更重要，给更高的 batch 比例
        batch_n_per_class["adversarial_negative"] = min(150, max(50, neg_shape[0] // 10))
        label_transforms["adversarial_negative"] = lambda x: [0 for _ in x]
        log.info(f"对抗性负样本: {neg_shape}, batch_n={batch_n_per_class['adversarial_negative']}")
    else:
        log.warning("无对抗性负样本特征，模型可能对相似发音误触发")

    log.info(f"batch_n_per_class: {batch_n_per_class}")

    # 创建训练数据加载器
    batch_generator = mmap_batch_generator(
        feature_data_files,
        n_per_class=batch_n_per_class,
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, gen):
            self.gen = gen
        def __iter__(self):
            return self.gen

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(
        IterDS(batch_generator), batch_size=None,
        num_workers=n_cpus, prefetch_factor=16,
    )

    # 验证数据：false positive 验证集
    X_val_fp = np.load(args.fp_val_data)
    X_val_fp = np.array([
        X_val_fp[i:i+input_shape[0]]
        for i in range(0, X_val_fp.shape[0] - input_shape[0], 1)
    ])
    X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
    X_val_fp_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)
        ),
        batch_size=len(X_val_fp_labels),
    )

    # 验证数据：正样本 + 对抗性负样本
    X_val_pos = np.load(pos_feat_test)
    if os.path.exists(neg_feat_test):
        X_val_neg = np.load(neg_feat_test)
    else:
        X_val_neg = np.zeros((100, *input_shape), dtype=np.float32)

    labels = np.hstack((
        np.ones(X_val_pos.shape[0]),
        np.zeros(X_val_neg.shape[0]),
    )).astype(np.float32)

    X_val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
            torch.from_numpy(labels),
        ),
        batch_size=len(labels),
    )

    # 训练
    # max_negative_weight 降低到 500（默认 1500 对中文太激进，
    # 因为 ACAV100M 英文负样本和中文正样本分布差异大，
    # 高权重会让模型把所有中文都判为负）
    log.info(f"开始训练: {args.steps} steps, layer_size={args.layer_size}")
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val_loader,
        false_positive_val_data=X_val_fp_loader,
        steps=args.steps,
        max_negative_weight=500,
        target_fp_per_hour=0.5,
    )

    # 导出
    oww.export_model(
        model=best_model,
        model_name=args.keyword_id,
        output_dir=args.output_dir,
    )
    log.info(f"模型导出: {args.output_dir}/{args.keyword_id}.onnx")


def main():
    parser = argparse.ArgumentParser(description="OWW 中文唤醒词训练")
    parser.add_argument("--keyword-id", default="nihao_shushi")
    parser.add_argument("--positive-dir", default="/workspace/data/positive_raw/nihao_shushi")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--acav-features",
                        default="/workspace/data/oww/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    parser.add_argument("--fp-val-data",
                        default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--background-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k",
                                 "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--n-pos-train", type=int, default=15000)
    parser.add_argument("--n-pos-test", type=int, default=3000)
    parser.add_argument("--augment-rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--layer-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-features", action="store_true",
                        help="强制重新计算特征（删除已有 .npy）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 添加 OWW 到 path
    oww_dir = "/workspace/work/openWakeWord"
    if os.path.exists(oww_dir):
        sys.path.insert(0, oww_dir)

    model_dir = os.path.join(args.output_dir, args.keyword_id)
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    neg_train_dir = os.path.join(model_dir, "negative_train")
    neg_test_dir = os.path.join(model_dir, "negative_test")

    # ── Step 1: 准备正样本 ──
    log.info("=" * 60)
    log.info("Step 1: 准备正样本")
    log.info("=" * 60)
    prepare_positive_clips(args.positive_dir, train_dir, test_dir,
                           args.n_pos_train, args.n_pos_test, seed=args.seed)

    # ── Step 2: 检查负样本 ──
    log.info("=" * 60)
    log.info("Step 2: 检查对抗性负样本")
    log.info("=" * 60)
    neg_train_count = len(list(Path(neg_train_dir).glob("*.wav"))) if os.path.exists(neg_train_dir) else 0
    neg_test_count = len(list(Path(neg_test_dir).glob("*.wav"))) if os.path.exists(neg_test_dir) else 0
    log.info(f"对抗性负样本: {neg_train_count} train, {neg_test_count} test")
    if neg_train_count == 0:
        log.warning("没有对抗性负样本! 请先在宿主机运行 generate_zh_negatives.py")
        log.warning("将仅使用 ACAV100M 作为负样本")

    # ── Step 3: 数据增强 + 计算特征 ──
    log.info("=" * 60)
    log.info("Step 3: 数据增强 + 计算特征")
    log.info("=" * 60)

    # 收集背景音频和 RIR 路径
    rir_paths = [i.path for i in os.scandir(args.rir_dir)] if os.path.exists(args.rir_dir) else []
    background_paths = []
    for bg_dir in args.background_dirs:
        if os.path.exists(bg_dir):
            background_paths.extend([i.path for i in os.scandir(bg_dir)])
    log.info(f"RIR: {len(rir_paths)} files, Background: {len(background_paths)} files")

    # 计算 clip 长度
    total_length = compute_clip_length(test_dir)

    # 如果 --overwrite-features，删除已有特征
    if args.overwrite_features:
        for f in Path(model_dir).glob("*_features_*.npy"):
            log.info(f"删除旧特征: {f}")
            f.unlink()

    # 计算正样本特征
    pos_feat_train = os.path.join(model_dir, "positive_features_train.npy")
    pos_feat_test = os.path.join(model_dir, "positive_features_test.npy")

    if not os.path.exists(pos_feat_train):
        log.info("计算正样本训练特征...")
        augment_and_compute_features(
            train_dir, pos_feat_train, total_length,
            args.augment_rounds, background_paths, rir_paths, args.batch_size,
        )
    else:
        log.info(f"正样本训练特征已存在: {np.load(pos_feat_train, mmap_mode='r').shape}")

    if not os.path.exists(pos_feat_test):
        log.info("计算正样本测试特征...")
        augment_and_compute_features(
            test_dir, pos_feat_test, total_length,
            1, background_paths, rir_paths, args.batch_size,  # test 只做 1 轮增强
        )
    else:
        log.info(f"正样本测试特征已存在: {np.load(pos_feat_test, mmap_mode='r').shape}")

    # 计算负样本特征
    neg_feat_train = os.path.join(model_dir, "negative_features_train.npy")
    neg_feat_test = os.path.join(model_dir, "negative_features_test.npy")

    if neg_train_count > 0 and not os.path.exists(neg_feat_train):
        log.info("计算负样本训练特征...")
        augment_and_compute_features(
            neg_train_dir, neg_feat_train, total_length,
            args.augment_rounds, background_paths, rir_paths, args.batch_size,
        )
    elif os.path.exists(neg_feat_train):
        log.info(f"负样本训练特征已存在: {np.load(neg_feat_train, mmap_mode='r').shape}")

    if neg_test_count > 0 and not os.path.exists(neg_feat_test):
        log.info("计算负样本测试特征...")
        augment_and_compute_features(
            neg_test_dir, neg_feat_test, total_length,
            1, background_paths, rir_paths, args.batch_size,
        )
    elif os.path.exists(neg_feat_test):
        log.info(f"负样本测试特征已存在: {np.load(neg_feat_test, mmap_mode='r').shape}")

    # ── Step 4: 训练 ──
    log.info("=" * 60)
    log.info("Step 4: 训练模型")
    log.info("=" * 60)
    train_model(args, model_dir)

    log.info("=" * 60)
    log.info(f"全部完成! 模型: {args.output_dir}/{args.keyword_id}.onnx")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

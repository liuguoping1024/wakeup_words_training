#!/usr/bin/env python3
"""
OWW 中文唤醒词训练 v4：全中文数据，三引擎混合。

正样本：真实录音 20k + CosyVoice2 5k + edge-tts 1.2k + Piper 少量
负样本：CosyVoice2 5k + edge-tts 1.4k（全中文，去掉 ACAV100M 英文）
测试集：真实录音 3k（正）+ CosyVoice2 1k（负）

参数对齐 openwakeword.com: 64x3 @ 115k steps, penalty 100
"""
import argparse, logging, os, random, shutil, sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_wav_16k(path):
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


def collect_wavs(d, max_n=None, seed=42):
    wavs = sorted(Path(d).glob("*.wav"))
    rng = random.Random(seed)
    rng.shuffle(wavs)
    return wavs[:max_n] if max_n else wavs


def copy_wavs(wavs, out_dir, prefix, skip_existing=True):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i, w in enumerate(tqdm(wavs, desc=f"copy {prefix}")):
        out = os.path.join(out_dir, f"{prefix}_{i:06d}.wav")
        if skip_existing and os.path.exists(out):
            count += 1
            continue
        try:
            pcm = load_wav_16k(w)
            scipy.io.wavfile.write(out, 16000, pcm)
            count += 1
        except Exception as e:
            pass
    return count


def prepare_data(args):
    """准备混合数据集。"""
    model_dir = os.path.join(args.output_dir, "nihao_shushi_v4")
    os.makedirs(model_dir, exist_ok=True)
    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    neg_train_dir = os.path.join(model_dir, "negative_train")
    neg_test_dir = os.path.join(model_dir, "negative_test")

    # 检查是否已准备好
    existing_train = len(list(Path(train_dir).glob("*.wav"))) if os.path.exists(train_dir) else 0
    if existing_train > 20000 and not args.overwrite:
        log.info(f"数据已存在 ({existing_train} train)，跳过准备")
        return model_dir

    # === 正样本 ===
    log.info("=== 准备正样本 ===")
    real_wavs = collect_wavs(args.real_dir, seed=args.seed)
    cosyvoice_pos = collect_wavs(args.cosyvoice_pos_dir, seed=args.seed) if os.path.exists(args.cosyvoice_pos_dir) else []
    edge_pos = collect_wavs(args.edge_pos_dir, seed=args.seed) if os.path.exists(args.edge_pos_dir) else []

    log.info(f"数据源: 真实={len(real_wavs)}, CosyVoice={len(cosyvoice_pos)}, edge-tts={len(edge_pos)}")

    # 测试集：从真实录音取
    test_wavs = real_wavs[:args.n_test]
    real_train = real_wavs[args.n_test:args.n_test + args.n_real]
    cosy_train = cosyvoice_pos[:args.n_cosyvoice]
    edge_train = edge_pos[:args.n_edge]

    log.info(f"正样本: real={len(real_train)}, cosyvoice={len(cosy_train)}, edge={len(edge_train)}")
    log.info(f"测试集: {len(test_wavs)} 条真实录音")

    # 写入
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    copy_wavs(real_train, train_dir, "real")
    copy_wavs(cosy_train, train_dir, "cosy")
    copy_wavs(edge_train, train_dir, "edge")
    copy_wavs(test_wavs, test_dir, "real")

    # === 负样本 ===
    log.info("=== 准备负样本 ===")
    cosy_neg_train = collect_wavs(args.cosyvoice_neg_train_dir, seed=args.seed) if os.path.exists(args.cosyvoice_neg_train_dir) else []
    cosy_neg_test = collect_wavs(args.cosyvoice_neg_test_dir, seed=args.seed) if os.path.exists(args.cosyvoice_neg_test_dir) else []
    edge_neg_train = collect_wavs(args.edge_neg_train_dir, seed=args.seed) if os.path.exists(args.edge_neg_train_dir) else []

    log.info(f"负样本: cosyvoice_train={len(cosy_neg_train)}, cosyvoice_test={len(cosy_neg_test)}, edge_train={len(edge_neg_train)}")

    os.makedirs(neg_train_dir, exist_ok=True)
    os.makedirs(neg_test_dir, exist_ok=True)
    copy_wavs(cosy_neg_train, neg_train_dir, "cosy")
    copy_wavs(edge_neg_train, neg_train_dir, "edge")
    copy_wavs(cosy_neg_test, neg_test_dir, "cosy")

    total_train = len(list(Path(train_dir).glob("*.wav")))
    total_test = len(list(Path(test_dir).glob("*.wav")))
    total_neg_train = len(list(Path(neg_train_dir).glob("*.wav")))
    total_neg_test = len(list(Path(neg_test_dir).glob("*.wav")))
    log.info(f"最终: pos_train={total_train}, pos_test={total_test}, neg_train={total_neg_train}, neg_test={total_neg_test}")

    return model_dir


def compute_clip_length(clip_dir, n_sample=100):
    clips = sorted(Path(clip_dir).glob("*.wav"))[:n_sample]
    durations = [len(scipy.io.wavfile.read(str(c))[1]) for c in clips]
    total_length = int(round(np.median(durations) / 1000) * 1000) + 12000
    total_length = max(total_length, 32000)
    if abs(total_length - 32000) <= 4000:
        total_length = 32000
    log.info(f"clip total_length: {total_length} ({total_length/16000:.2f}s)")
    return total_length


def augment_and_features(clip_dir, output_npy, total_length, augment_rounds, bg_paths, rir_paths, bs=16):
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator
    clips = [str(p) for p in sorted(Path(clip_dir).glob("*.wav"))]
    if not clips:
        log.warning(f"空目录: {clip_dir}")
        return
    all_clips = clips * max(1, augment_rounds)
    log.info(f"增强: {len(clips)} × {augment_rounds} = {len(all_clips)}")
    gen = augment_clips(all_clips, total_length=total_length, batch_size=bs,
                         background_clip_paths=bg_paths, RIR_paths=rir_paths)
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
                layer_dim=64, n_blocks=3,
                seconds_per_example=1280 * input_shape[0] / 16000)

    def reshape_fn(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    # 全中文数据，不用 ACAV100M
    feature_data_files = {"positive": pos_feat_train}
    batch_n_per_class = {"positive": 200}
    data_transforms = {}
    label_transforms = {"positive": lambda x: [1 for _ in x]}

    if os.path.exists(neg_feat_train):
        neg_n = np.load(neg_feat_train, mmap_mode='r').shape[0]
        feature_data_files["adversarial_negative"] = neg_feat_train
        batch_n_per_class["adversarial_negative"] = min(200, max(50, neg_n // 50))
        label_transforms["adversarial_negative"] = lambda x: [0 for _ in x]

    log.info(f"batch_n_per_class: {batch_n_per_class}")

    batch_gen = mmap_batch_generator(feature_data_files, n_per_class=batch_n_per_class,
                                      data_transform_funcs=data_transforms,
                                      label_transform_funcs=label_transforms)

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, g): self.g = g
        def __iter__(self): return self.g

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(IterDS(batch_gen), batch_size=None,
                                           num_workers=n_cpus, prefetch_factor=16)

    # Validation: false positive
    fp_val = np.load(args.fp_val_data)
    fp_val = np.array([fp_val[i:i+input_shape[0]] for i in range(0, fp_val.shape[0]-input_shape[0], 1)])
    fp_labels = np.zeros(fp_val.shape[0]).astype(np.float32)
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_val), torch.from_numpy(fp_labels)),
        batch_size=len(fp_labels))

    # Validation: pos + neg
    X_pos = np.load(pos_feat_test)
    X_neg = np.load(neg_feat_test) if os.path.exists(neg_feat_test) else np.zeros((100, *input_shape), dtype=np.float32)
    labels = np.hstack((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0]))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((X_pos, X_neg))), torch.from_numpy(labels)),
        batch_size=len(labels))

    log.info(f"训练: 115k steps, 64x3, penalty=100, 全中文数据")
    best_model = oww.auto_train(X_train=X_train, X_val=X_val, false_positive_val_data=X_val_fp,
                                 steps=115000, max_negative_weight=100, target_fp_per_hour=1.0)

    oww.export_model(model=best_model, model_name="nihao_shushi", output_dir=args.output_dir)
    log.info(f"模型: {args.output_dir}/nihao_shushi.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="/workspace/data/positive_raw/nihao_shushi")
    parser.add_argument("--cosyvoice-pos-dir", default="/workspace/outputs/cosyvoice_clips/positive")
    parser.add_argument("--cosyvoice-neg-train-dir", default="/workspace/outputs/cosyvoice_clips/negative_train")
    parser.add_argument("--cosyvoice-neg-test-dir", default="/workspace/outputs/cosyvoice_clips/negative_test")
    parser.add_argument("--edge-pos-dir", default="/workspace/outputs/oww/nihao_shushi_v3/edge_tts_positive")
    parser.add_argument("--edge-neg-train-dir", default="/workspace/outputs/oww/nihao_shushi_v3/negative_train")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--fp-val-data", default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--background-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k", "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--n-real", type=int, default=20000)
    parser.add_argument("--n-cosyvoice", type=int, default=5000)
    parser.add_argument("--n-edge", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--augment-rounds", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, "/workspace/work/openWakeWord")

    # Step 1: 准备数据
    log.info("=" * 60)
    log.info("Step 1: 准备混合数据集")
    log.info("=" * 60)
    model_dir = prepare_data(args)

    # Step 2: 增强 + 特征
    log.info("=" * 60)
    log.info("Step 2: 数据增强 + 特征计算")
    log.info("=" * 60)
    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    neg_train_dir = os.path.join(model_dir, "negative_train")
    neg_test_dir = os.path.join(model_dir, "negative_test")

    rir_paths = [i.path for i in os.scandir(args.rir_dir)] if os.path.exists(args.rir_dir) else []
    bg_paths = []
    for d in args.background_dirs:
        if os.path.exists(d):
            bg_paths.extend([i.path for i in os.scandir(d)])

    total_length = compute_clip_length(test_dir)

    if args.overwrite:
        for f in Path(model_dir).glob("*_features_*.npy"):
            f.unlink()

    for name, cdir, rounds in [
        ("positive_features_train", train_dir, args.augment_rounds),
        ("positive_features_test", test_dir, 1),
        ("negative_features_train", neg_train_dir, args.augment_rounds),
        ("negative_features_test", neg_test_dir, 1),
    ]:
        npy = os.path.join(model_dir, f"{name}.npy")
        if not os.path.exists(npy):
            n = len(list(Path(cdir).glob("*.wav")))
            if n > 0:
                log.info(f"计算 {name} ({n} × {rounds})...")
                augment_and_features(cdir, npy, total_length, rounds, bg_paths, rir_paths)
        else:
            log.info(f"{name} 已存在: {np.load(npy, mmap_mode='r').shape}")

    # Step 3: 训练
    log.info("=" * 60)
    log.info("Step 3: 训练")
    log.info("=" * 60)
    train_model(args, model_dir)

    log.info("=" * 60)
    log.info("全部完成!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OWW "救命" 训练：使用 MWW 训练过的素材。

正样本：真实录音 1k + 多说话人 CosyVoice 5k + Piper 2k = 8k
负样本：CosyVoice 对抗性短语 13k train + 7k test
无 ACAV100M，全中文数据。
64x3 @ 115k steps, penalty 250
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
        else:
            audio = data.astype(np.float32) / np.iinfo(data.dtype).max
    except:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000)
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def prepare_dir(src_dirs, dst_dir, max_n=None, seed=42):
    """从多个源目录收集 WAV 到目标目录。"""
    os.makedirs(dst_dir, exist_ok=True)
    existing = len(list(Path(dst_dir).glob("*.wav")))
    if existing > 100:
        log.info(f"  {dst_dir}: 已有 {existing} 条")
        return existing

    all_wavs = []
    for d in src_dirs:
        if os.path.exists(d):
            all_wavs.extend(sorted(Path(d).glob("*.wav")))
    rng = random.Random(seed)
    rng.shuffle(all_wavs)
    if max_n:
        all_wavs = all_wavs[:max_n]

    for i, w in enumerate(tqdm(all_wavs, desc=f"copy to {Path(dst_dir).name}")):
        out = os.path.join(dst_dir, f"{i:06d}.wav")
        try:
            pcm = load_wav_16k(w)
            scipy.io.wavfile.write(out, 16000, pcm)
        except:
            pass
    total = len(list(Path(dst_dir).glob("*.wav")))
    log.info(f"  {dst_dir}: {total} 条")
    return total


def compute_clip_length(clip_dir, n=100):
    clips = sorted(Path(clip_dir).glob("*.wav"))[:n]
    durs = [len(scipy.io.wavfile.read(str(c))[1]) for c in clips]
    tl = int(round(np.median(durs) / 1000) * 1000) + 12000
    tl = max(tl, 32000)
    if abs(tl - 32000) <= 4000:
        tl = 32000
    log.info(f"clip total_length: {tl} ({tl/16000:.2f}s)")
    return tl


def augment_and_features(clip_dir, npy, total_length, rounds, bg, rir, bs=16):
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator
    clips = [str(p) for p in sorted(Path(clip_dir).glob("*.wav"))]
    if not clips:
        return
    all_clips = clips * max(1, rounds)
    log.info(f"增强: {len(clips)} × {rounds} = {len(all_clips)}")
    gen = augment_clips(all_clips, total_length=total_length, batch_size=bs,
                         background_clip_paths=bg, RIR_paths=rir)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    compute_features_from_generator(gen, n_total=len(all_clips), clip_duration=total_length,
                                     output_file=npy, device=device, ncpu=1)
    log.info(f"特征: {npy} -> {np.load(npy, mmap_mode='r').shape}")


def train(args, model_dir):
    from openwakeword.train import Model, mmap_batch_generator

    pos_train = os.path.join(model_dir, "positive_features_train.npy")
    pos_test = os.path.join(model_dir, "positive_features_test.npy")
    neg_train = os.path.join(model_dir, "negative_features_train.npy")
    neg_test = os.path.join(model_dir, "negative_features_test.npy")

    input_shape = np.load(pos_test, mmap_mode='r').shape[1:]
    log.info(f"input_shape: {input_shape}")

    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=64, n_blocks=3,
                seconds_per_example=1280 * input_shape[0] / 16000)

    feature_data_files = {"positive": pos_train, "adversarial_negative": neg_train}
    batch_n = {"positive": 200, "adversarial_negative": 200}
    label_transforms = {
        "positive": lambda x: [1 for _ in x],
        "adversarial_negative": lambda x: [0 for _ in x],
    }

    log.info(f"batch: {batch_n}, 正样本占比 50%")

    batch_gen = mmap_batch_generator(feature_data_files, n_per_class=batch_n,
                                      label_transform_funcs=label_transforms)

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, g): self.g = g
        def __iter__(self): return self.g

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(IterDS(batch_gen), batch_size=None,
                                           num_workers=n_cpus, prefetch_factor=16)

    fp_val = np.load(args.fp_val_data)
    fp_val = np.array([fp_val[i:i+input_shape[0]] for i in range(0, fp_val.shape[0]-input_shape[0], 1)])
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_val),
                                         torch.from_numpy(np.zeros(fp_val.shape[0]).astype(np.float32))),
        batch_size=len(fp_val))

    X_pos = np.load(pos_test)
    X_neg = np.load(neg_test) if os.path.exists(neg_test) else np.zeros((100, *input_shape), dtype=np.float32)
    labels = np.hstack((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0]))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((X_pos, X_neg))), torch.from_numpy(labels)),
        batch_size=len(labels))

    log.info(f"训练: 115k steps, 64x3, penalty=250, 全中文")
    best = oww.auto_train(X_train=X_train, X_val=X_val, false_positive_val_data=X_val_fp,
                           steps=115000, max_negative_weight=250, target_fp_per_hour=0.5)

    oww.export_model(model=best, model_name="jiuming", output_dir=args.output_dir)
    log.info(f"模型: {args.output_dir}/jiuming.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-dir", default="/workspace/data/positive_raw/jiuming_v3")
    parser.add_argument("--neg-train-dirs", nargs="+", default=[
        "/workspace/outputs/cosyvoice_clips_jiuming/negative_train",
        "/workspace/outputs/cosyvoice_clips_jiuming_gpu1_neg/negative_train",
        "/workspace/outputs/cosyvoice_clips_jiuming_gpu2/negative_train",
    ])
    parser.add_argument("--neg-test-dirs", nargs="+", default=[
        "/workspace/outputs/cosyvoice_clips_jiuming/negative_test",
        "/workspace/outputs/cosyvoice_clips_jiuming_gpu2_neg/negative_test",
        "/workspace/outputs/cosyvoice_clips_jiuming_gpu3/negative_test",
    ])
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--fp-val-data", default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--bg-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k", "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--augment-rounds", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, "/workspace/work/openWakeWord")

    model_dir = os.path.join(args.output_dir, "jiuming_oww")
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    neg_train_dir = os.path.join(model_dir, "negative_train")
    neg_test_dir = os.path.join(model_dir, "negative_test")

    # Step 1: 准备数据
    log.info("=" * 60)
    log.info("Step 1: 准备数据")

    # 正样本：从 jiuming_v3 抽 6000 train + 2000 test
    all_pos = sorted(Path(args.pos_dir).glob("*.wav"))
    rng = random.Random(args.seed)
    rng.shuffle(all_pos)
    test_wavs = all_pos[:2000]
    train_wavs = all_pos[2000:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if len(list(Path(train_dir).glob("*.wav"))) < 100 or args.overwrite:
        for i, w in enumerate(tqdm(train_wavs, desc="pos_train")):
            try:
                pcm = load_wav_16k(w)
                scipy.io.wavfile.write(os.path.join(train_dir, f"{i:06d}.wav"), 16000, pcm)
            except: pass
        for i, w in enumerate(tqdm(test_wavs, desc="pos_test")):
            try:
                pcm = load_wav_16k(w)
                scipy.io.wavfile.write(os.path.join(test_dir, f"{i:06d}.wav"), 16000, pcm)
            except: pass

    log.info(f"正样本: {len(list(Path(train_dir).glob('*.wav')))} train, {len(list(Path(test_dir).glob('*.wav')))} test")

    # 负样本
    prepare_dir(args.neg_train_dirs, neg_train_dir, max_n=10000, seed=args.seed)
    prepare_dir(args.neg_test_dirs, neg_test_dir, max_n=2000, seed=args.seed)

    # Step 2: 增强 + 特征
    log.info("=" * 60)
    log.info("Step 2: 增强 + 特征")

    rir = [i.path for i in os.scandir(args.rir_dir)] if os.path.exists(args.rir_dir) else []
    bg = []
    for d in args.bg_dirs:
        if os.path.exists(d):
            bg.extend([i.path for i in os.scandir(d)])

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
                augment_and_features(cdir, npy, total_length, rounds, bg, rir)
        else:
            log.info(f"{name}: {np.load(npy, mmap_mode='r').shape}")

    # Step 3: 训练
    log.info("=" * 60)
    log.info("Step 3: 训练")
    train(args, model_dir)

    log.info("全部完成!")


if __name__ == "__main__":
    main()

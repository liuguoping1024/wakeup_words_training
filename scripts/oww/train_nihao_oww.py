#!/usr/bin/env python3
"""
用 openWakeWord 训练中文唤醒词（如「你好树实」）。

方案：
  - 正样本：真实录音（data/positive_raw/nihao_shushi）
  - 对抗性负样本：用 edge-tts 生成发音相近的中文短语
  - 背景负样本：ACAV100M 预计算特征
  - 数据增强：OWW 标准 augment pipeline

用法：
  python train_nihao_oww.py --steps 50000
"""
import argparse
import asyncio
import logging
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# 对抗性负样本短语：发音接近「你好树实」但不是的词
ADVERSARIAL_PHRASES = [
    # 声母/韵母相近
    "你好叔叔", "你好书式", "你好树枝", "你好属实",
    "你好舒适", "你好数十", "你好输食", "你好书市",
    "你好树石", "你好鼠食", "你好束时", "你好述事",
    # 部分匹配
    "你好", "树实", "你好啊", "你好吗",
    "你好世界", "你好师傅", "你好同学", "你好朋友",
    # 声调变化
    "泥好树实", "拟好树实", "你浩树实", "你号树实",
    # 完全不同但常见
    "小爱同学", "天猫精灵", "嘿 Siri", "你好小度",
    "OK Google", "Alexa", "打开灯", "关闭窗帘",
    "今天天气", "几点了", "播放音乐", "设个闹钟",
    # 日常对话片段
    "你好吃吗", "你好看吗", "你好厉害", "你好棒",
    "你好快", "你好慢", "你好大", "你好小",
    "树上有鸟", "实在太好", "书本在哪", "数学题",
]

# edge-tts 中文语音列表
ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural",
    "zh-CN-XiaomengNeural",
    "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural",
    "zh-CN-XiaoruiNeural",
    "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural",
    "zh-CN-XiaoyanNeural",
    "zh-CN-XiaozhenNeural",
    "zh-CN-YunfengNeural",
    "zh-CN-YunhaoNeural",
    "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]


async def generate_tts_clip(text, voice, output_path, rate="+0%"):
    """用 edge-tts 生成单条语音。"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


async def generate_adversarial_clips(output_dir, n_samples, phrases, voices):
    """批量生成对抗性负样本。"""
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    existing = len(list(Path(output_dir).glob("*.wav")))
    if existing >= n_samples * 0.9:
        log.info(f"Adversarial clips already exist ({existing} files), skipping")
        return

    log.info(f"Generating {n_samples} adversarial negative clips with edge-tts...")
    rates = ["-20%", "-10%", "+0%", "+10%", "+20%"]
    count = existing
    retries = 0
    max_retries = 3

    for i in range(existing, n_samples):
        text = random.choice(phrases)
        voice = random.choice(voices)
        rate = random.choice(rates)
        mp3_path = os.path.join(output_dir, f"neg_{i:05d}.mp3")
        wav_path = os.path.join(output_dir, f"neg_{i:05d}.wav")

        if os.path.exists(wav_path):
            count += 1
            continue

        success = False
        for attempt in range(max_retries):
            try:
                await generate_tts_clip(text, voice, mp3_path, rate=rate)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path],
                    check=True
                )
                os.remove(mp3_path)
                count += 1
                success = True
                break
            except Exception as e:
                retries += 1
                await asyncio.sleep(1 + attempt * 2)  # backoff: 1s, 3s, 5s

        if not success:
            log.warning(f"  Failed clip {i} after {max_retries} attempts")

        # Rate limit: small delay between requests
        if i % 10 == 0:
            await asyncio.sleep(0.5)

        if count % 200 == 0 and count > 0:
            log.info(f"  Generated {count}/{n_samples} adversarial clips")

    log.info(f"Generated {count} adversarial clips in {output_dir}")


def generate_positive_tts(output_dir, target_phrase, n_samples, voices):
    """用 edge-tts 生成正样本的 TTS 版本（补充真实录音）。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len(list(Path(output_dir).glob("*.wav")))
    if existing >= n_samples * 0.9:
        log.info(f"TTS positive clips already exist ({existing} files), skipping")
        return

    log.info(f"Generating {n_samples} TTS positive clips...")
    rates = ["-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%"]

    async def _gen():
        import subprocess
        count = 0
        for i in range(n_samples):
            voice = random.choice(voices)
            rate = random.choice(rates)
            mp3_path = os.path.join(output_dir, f"tts_pos_{i:05d}.mp3")
            wav_path = os.path.join(output_dir, f"tts_pos_{i:05d}.wav")
            if os.path.exists(wav_path):
                count += 1
                continue
            for attempt in range(3):
                try:
                    await generate_tts_clip(target_phrase, voice, mp3_path, rate=rate)
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
                         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path],
                        check=True
                    )
                    os.remove(mp3_path)
                    count += 1
                    break
                except Exception as e:
                    await asyncio.sleep(1 + attempt * 2)
            if i % 10 == 0:
                await asyncio.sleep(0.5)
            if count % 200 == 0 and count > 0:
                log.info(f"  Generated {count}/{n_samples} TTS positive clips")
        log.info(f"Generated {count} TTS positive clips")

    asyncio.run(_gen())


def prepare_real_clips(real_dir, train_dir, test_dir, n_train, n_val):
    """复制真实录音到 OWW 目录结构。"""
    wavs = sorted(Path(real_dir).glob("*.wav"))
    log.info(f"Found {len(wavs)} real clips in {real_dir}")

    random.shuffle(wavs)
    n_val = min(n_val, len(wavs) // 5)
    val_wavs = wavs[:n_val]
    train_wavs = wavs[n_val:n_val + n_train]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if len(os.listdir(train_dir)) >= len(train_wavs) * 0.9:
        log.info(f"Real clips already copied ({len(os.listdir(train_dir))} train files)")
        return

    log.info(f"Copying {len(train_wavs)} train + {len(val_wavs)} val real clips...")
    for i, wav in enumerate(tqdm(train_wavs, desc="copy train")):
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
        out = os.path.join(train_dir, f"real_{i:05d}.wav")
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    for i, wav in enumerate(tqdm(val_wavs, desc="copy val")):
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
        out = os.path.join(test_dir, f"real_{i:05d}.wav")
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", default="/workspace/data/positive_raw/nihao_shushi")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--oww-dir", default="/workspace/work/openWakeWord")
    parser.add_argument("--acav-features", default="/workspace/data/oww/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    parser.add_argument("--fp-val-data", default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--background-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k",
                                 "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--target-phrase", default="你好树实")
    parser.add_argument("--n-real-train", type=int, default=10000)
    parser.add_argument("--n-real-val", type=int, default=2000)
    parser.add_argument("--n-tts-positive", type=int, default=2000,
                        help="额外 TTS 正样本数（补充真实录音多样性）")
    parser.add_argument("--n-adversarial", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, args.oww_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = "nihao_shushi"
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(model_dir, "positive_train")
    test_dir = os.path.join(model_dir, "positive_test")
    neg_train_dir = os.path.join(model_dir, "negative_train")
    neg_test_dir = os.path.join(model_dir, "negative_test")

    # ── Step 1: 准备正样本（真实录音 + TTS 补充）──
    log.info("=== Step 1: Prepare positive samples ===")
    prepare_real_clips(args.positive_dir, train_dir, test_dir,
                       args.n_real_train, args.n_real_val)

    # TTS 补充正样本（如果 tts_positive 目录已有足够文件则跳过）
    tts_pos_dir = os.path.join(model_dir, "tts_positive")
    tts_count = len(list(Path(tts_pos_dir).glob("*.wav"))) if os.path.exists(tts_pos_dir) else 0
    if tts_count < args.n_tts_positive * 0.5:
        log.info(f"TTS positive clips insufficient ({tts_count}), skipping TTS generation (use scripts/oww/generate_zh_tts.py on host)")
    else:
        log.info(f"Found {tts_count} TTS positive clips")

    # 把 TTS 正样本也复制到 train 目录
    tts_wavs = list(Path(tts_pos_dir).glob("*.wav"))
    for wav in tts_wavs:
        dst = os.path.join(train_dir, wav.name)
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(str(wav), dst)
    log.info(f"Total positive train clips: {len(os.listdir(train_dir))}")

    # ── Step 2: 检查对抗性负样本（已在宿主机用 edge-tts 生成）──
    log.info("=== Step 2: Check adversarial negatives ===")
    neg_train_count = len(list(Path(neg_train_dir).glob("*.wav"))) if os.path.exists(neg_train_dir) else 0
    neg_test_count = len(list(Path(neg_test_dir).glob("*.wav"))) if os.path.exists(neg_test_dir) else 0
    log.info(f"Adversarial negatives: {neg_train_count} train, {neg_test_count} test")
    if neg_train_count == 0:
        log.warning("No adversarial negatives found! Run scripts/oww/generate_zh_tts.py on host first.")
        log.warning("Continuing without adversarial negatives (using ACAV100M only).")

    # ── Step 3: 数据增强 + 计算特征 ──
    log.info("=== Step 3: Augment and compute features ===")
    from openwakeword.utils import compute_features_from_generator
    from openwakeword.train import augment_clips

    rir_paths = [i.path for i in os.scandir(args.rir_dir)]
    background_paths = []
    for bg_dir in args.background_dirs:
        if os.path.exists(bg_dir):
            background_paths.extend([i.path for i in os.scandir(bg_dir)])

    # Determine clip length
    sample_wavs = [str(p) for p in Path(test_dir).glob("*.wav")][:50]
    durations = []
    for w in sample_wavs:
        sr, dat = scipy.io.wavfile.read(w)
        durations.append(len(dat))
    total_length = int(round(np.median(durations) / 1000) * 1000) + 12000
    total_length = max(total_length, 32000)
    if abs(total_length - 32000) <= 4000:
        total_length = 32000
    log.info(f"Clip total_length: {total_length} samples ({total_length/16000:.2f}s)")

    pos_feat_train = os.path.join(model_dir, "positive_features_train.npy")
    neg_feat_train = os.path.join(model_dir, "negative_features_train.npy")
    pos_feat_test = os.path.join(model_dir, "positive_features_test.npy")
    neg_feat_test = os.path.join(model_dir, "negative_features_test.npy")

    device = "gpu" if torch.cuda.is_available() else "cpu"
    n_cpus = max(1, (os.cpu_count() or 1) // 2)
    ncpu = n_cpus if device == "cpu" else 1

    if not os.path.exists(pos_feat_train):
        log.info("Computing positive features...")
        pos_clips = [str(i) for i in Path(train_dir).glob("*.wav")]
        gen = augment_clips(pos_clips, total_length=total_length, batch_size=16,
                            background_clip_paths=background_paths, RIR_paths=rir_paths)
        compute_features_from_generator(gen, n_total=len(pos_clips),
                                         clip_duration=total_length, output_file=pos_feat_train,
                                         device=device, ncpu=ncpu)

        pos_test_clips = [str(i) for i in Path(test_dir).glob("*.wav")]
        gen = augment_clips(pos_test_clips, total_length=total_length, batch_size=16,
                            background_clip_paths=background_paths, RIR_paths=rir_paths)
        compute_features_from_generator(gen, n_total=len(pos_test_clips),
                                         clip_duration=total_length, output_file=pos_feat_test,
                                         device=device, ncpu=ncpu)
    else:
        log.info("Positive features already exist")

    if not os.path.exists(neg_feat_train):
        log.info("Computing negative features...")
        neg_clips = [str(i) for i in Path(neg_train_dir).glob("*.wav")]
        if neg_clips:
            gen = augment_clips(neg_clips, total_length=total_length, batch_size=16,
                                background_clip_paths=background_paths, RIR_paths=rir_paths)
            compute_features_from_generator(gen, n_total=len(neg_clips),
                                             clip_duration=total_length, output_file=neg_feat_train,
                                             device=device, ncpu=ncpu)

            neg_test_clips = [str(i) for i in Path(neg_test_dir).glob("*.wav")]
            gen = augment_clips(neg_test_clips, total_length=total_length, batch_size=16,
                                background_clip_paths=background_paths, RIR_paths=rir_paths)
            compute_features_from_generator(gen, n_total=len(neg_test_clips),
                                             clip_duration=total_length, output_file=neg_feat_test,
                                             device=device, ncpu=ncpu)
    else:
        log.info("Negative features already exist")

    # ── Step 4: 训练 ──
    log.info("=== Step 4: Train model ===")
    from openwakeword.train import Model, mmap_batch_generator
    from openwakeword.utils import AudioFeatures

    input_shape = np.load(pos_feat_test).shape[1:]
    log.info(f"Input shape: {input_shape}")

    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=32, seconds_per_example=1280 * input_shape[0] / 16000)

    def f(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
        return x

    feature_data_files = {
        "ACAV100M": args.acav_features,
        "positive": pos_feat_train,
        "adversarial_negative": neg_feat_train,
    }

    batch_n_per_class = {
        "ACAV100M": 1024,
        "positive": 50,
        "adversarial_negative": 50,
    }

    data_transforms = {"ACAV100M": f}
    label_transforms = {
        "positive": lambda x: [1 for _ in x],
        "ACAV100M": lambda x: [0 for _ in x],
        "adversarial_negative": lambda x: [0 for _ in x],
    }

    batch_generator = mmap_batch_generator(
        feature_data_files, n_per_class=batch_n_per_class,
        data_transform_funcs=data_transforms, label_transform_funcs=label_transforms
    )

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, gen):
            self.gen = gen
        def __iter__(self):
            return self.gen

    n_cpus = max(1, (os.cpu_count() or 1) // 2)
    X_train = torch.utils.data.DataLoader(IterDS(batch_generator),
                                           batch_size=None, num_workers=n_cpus, prefetch_factor=16)

    # Validation data
    X_val_fp = np.load(args.fp_val_data)
    X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])
    X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
        batch_size=len(X_val_fp_labels)
    )

    X_val_pos = np.load(pos_feat_test)
    X_val_neg = np.load(neg_feat_test) if os.path.exists(neg_feat_test) else np.zeros_like(X_val_pos[:100])
    labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
            torch.from_numpy(labels)
        ),
        batch_size=len(labels)
    )

    best_model = oww.auto_train(
        X_train=X_train, X_val=X_val, false_positive_val_data=X_val_fp,
        steps=args.steps, max_negative_weight=1500, target_fp_per_hour=0.2
    )

    oww.export_model(model=best_model, model_name=model_name, output_dir=args.output_dir)
    log.info(f"=== Done! Model: {args.output_dir}/{model_name}.onnx ===")


if __name__ == "__main__":
    main()

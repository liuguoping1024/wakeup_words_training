#!/usr/bin/env python3
"""
openWakeWord 训练脚本

支持两种正样本来源：
  1. TTS 合成（help_me 等英文唤醒词）
  2. 真实录音（nihao_shushi 等中文唤醒词）

用法：
  # TTS 模式（英文）
  python train_oww.py --keyword-phrase "help me" --keyword-id help_me --mode tts

  # 真实语音模式（中文）
  python train_oww.py --keyword-phrase "你好树实" --keyword-id nihao_shushi --mode real \
    --real-positive-dir /workspace/data/positive_raw/nihao_shushi
"""
import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def build_config(args):
    """Build OWW training YAML config."""
    config = {
        "model_name": args.keyword_id,
        "target_phrase": [args.keyword_phrase],
        "custom_negative_phrases": [],
        "n_samples": args.n_samples,
        "n_samples_val": args.n_samples_val,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": args.psg_dir,
        "output_dir": args.output_dir,
        "rir_paths": [args.rir_dir],
        "background_paths": args.background_dirs,
        "background_paths_duplication_rate": [1] * len(args.background_dirs),
        "false_positive_validation_data_path": args.fp_val_data,
        "augmentation_rounds": 1,
        "feature_data_files": {},
        "batch_n_per_class": {
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": args.steps,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }

    # Add ACAV100M features if available
    if args.acav_features and os.path.exists(args.acav_features):
        config["feature_data_files"]["ACAV100M"] = args.acav_features
        config["batch_n_per_class"]["ACAV100M"] = 1024

    return config


def download_oww_data(args):
    """Download OWW-specific data (features, validation set)."""
    import requests
    from tqdm import tqdm

    ensure_dir(args.oww_data_dir)

    # ACAV100M features (~2000 hrs)
    acav_path = os.path.join(args.oww_data_dir, "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    if not os.path.exists(acav_path):
        log.info("Downloading ACAV100M features (~4GB, this will take a while)...")
        url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        r = requests.get(url, stream=True)
        total = int(r.headers.get("content-length", 0))
        with open(acav_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), total=total // 8192, desc="ACAV100M"):
                f.write(chunk)
    else:
        log.info(f"ACAV100M features already exist: {acav_path}")

    # Validation set
    val_path = os.path.join(args.oww_data_dir, "validation_set_features.npy")
    if not os.path.exists(val_path):
        log.info("Downloading validation set features...")
        url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
        r = requests.get(url, stream=True)
        with open(val_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        log.info(f"Validation features already exist: {val_path}")

    return acav_path, val_path


def prepare_real_positive_clips(real_dir, output_dir, n_train, n_val):
    """
    Copy real voice clips into OWW positive train/test directories.
    For real voice mode, we skip TTS generation and use actual recordings.
    """
    import soundfile as sf
    import random

    wavs = sorted(Path(real_dir).glob("*.wav"))
    if not wavs:
        log.error(f"No WAV files found in {real_dir}")
        sys.exit(1)

    log.info(f"Found {len(wavs)} real voice clips in {real_dir}")

    random.seed(42)
    random.shuffle(wavs)

    # Split into train/val
    n_val_actual = min(n_val, len(wavs) // 5)
    val_wavs = wavs[:n_val_actual]
    train_wavs = wavs[n_val_actual:]

    train_dir = os.path.join(output_dir, "positive_train")
    test_dir = os.path.join(output_dir, "positive_test")
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    for wav in train_wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
        out = os.path.join(train_dir, wav.name)
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    for wav in val_wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
        out = os.path.join(test_dir, wav.name)
        scipy.io.wavfile.write(out, 16000, (audio * 32767).clip(-32768, 32767).astype(np.int16))

    log.info(f"Prepared {len(train_wavs)} train + {len(val_wavs)} val clips")
    return len(train_wavs), len(val_wavs)


def main():
    parser = argparse.ArgumentParser(description="openWakeWord training")
    parser.add_argument("--keyword-phrase", required=True)
    parser.add_argument("--keyword-id", required=True)
    parser.add_argument("--mode", choices=["tts", "real"], default="tts",
                        help="tts=TTS合成正样本, real=真实录音正样本")
    parser.add_argument("--real-positive-dir", default=None,
                        help="真实正样本目录 (mode=real 时必须)")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-samples-val", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--oww-data-dir", default="/workspace/data/oww")
    parser.add_argument("--oww-dir", default="/workspace/work/openWakeWord")
    parser.add_argument("--psg-dir", default="/workspace/work/piper-sample-generator-oww")
    parser.add_argument("--rir-dir", default="/workspace/data/augmentation/mit_rirs")
    parser.add_argument("--background-dirs", nargs="+",
                        default=["/workspace/data/augmentation/audioset_16k",
                                 "/workspace/data/augmentation/fma_16k"])
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.mode == "real" and not args.real_positive_dir:
        parser.error("--real-positive-dir is required when mode=real")

    ensure_dir(args.output_dir)
    ensure_dir(args.oww_data_dir)

    # Add OWW to path
    sys.path.insert(0, args.oww_dir)

    # Download OWW-specific data
    if not args.skip_download:
        acav_path, val_path = download_oww_data(args)
    else:
        acav_path = os.path.join(args.oww_data_dir, "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
        val_path = os.path.join(args.oww_data_dir, "validation_set_features.npy")

    args.acav_features = acav_path
    args.fp_val_data = val_path

    # Build config
    config = build_config(args)

    # Model output directory
    model_output_dir = os.path.join(args.output_dir, args.keyword_id)
    config["output_dir"] = model_output_dir
    ensure_dir(model_output_dir)

    # For real mode, prepare clips manually
    if args.mode == "real":
        n_train, n_val = prepare_real_positive_clips(
            args.real_positive_dir,
            os.path.join(model_output_dir, args.keyword_id),
            args.n_samples, args.n_samples_val
        )
        config["n_samples"] = n_train
        config["n_samples_val"] = n_val

    # Save config
    config_path = os.path.join(model_output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    log.info(f"Config saved to {config_path}")

    # Run OWW training via its train.py
    train_script = os.path.join(args.oww_dir, "openwakeword", "train.py")

    if args.mode == "tts":
        # Step 1: Generate clips
        log.info("=== Step 1: Generate synthetic clips ===")
        os.system(f"python3 {train_script} --training_config {config_path} --generate_clips")

    # Step 2: Augment clips
    log.info("=== Step 2: Augment clips ===")
    os.system(f"python3 {train_script} --training_config {config_path} --augment_clips")

    # Step 3: Train model
    log.info("=== Step 3: Train model ===")
    os.system(f"python3 {train_script} --training_config {config_path} --train_model --convert_to_tflite")

    log.info(f"=== Done! Model saved to {model_output_dir} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
将对抗性负样本转为 MWW 的 mmap 负样本格式。
在 wakeword-mww Docker 内运行。

用法：
  python3 generate_adversarial_features.py \
    --input-dir /workspace/outputs/adversarial_merged \
    --output-dir /workspace/data/negative_datasets/zh_adversarial/zh_adversarial \
    --data-dir /workspace/data
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="/workspace/data")
    args = parser.parse_args()

    import os, shutil, random
    random.seed(42)

    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    input_dir = args.input_dir
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    wav_count = len(list(Path(input_dir).glob("*.wav")))
    print(f"[info] 输入: {input_dir} ({wav_count} WAV)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 增强配置
    impulse_paths = []
    if (data_dir / "rirs_noises").exists():
        impulse_paths.append(str(data_dir / "rirs_noises"))
    if (data_dir / "augmentation" / "mit_rirs").exists():
        impulse_paths.append(str(data_dir / "augmentation" / "mit_rirs"))

    background_paths = []
    if (data_dir / "musan" / "musan" / "noise").exists():
        background_paths.append(str(data_dir / "musan" / "musan" / "noise"))
    if (data_dir / "musan" / "musan" / "music").exists():
        background_paths.append(str(data_dir / "musan" / "musan" / "music"))
    if (data_dir / "augmentation" / "fma_16k").exists():
        background_paths.append(str(data_dir / "augmentation" / "fma_16k"))
    if (data_dir / "augmentation" / "audioset_16k").exists():
        background_paths.append(str(data_dir / "augmentation" / "audioset_16k"))

    print(f"[info] 增强: impulse={len(impulse_paths)}, background={len(background_paths)}")

    clips = Clips(
        input_directory=input_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=impulse_paths if impulse_paths else [str(data_dir / "augmentation" / "mit_rirs")],
        background_paths=background_paths if background_paths else [str(data_dir / "augmentation" / "fma_16k")],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split in ["training", "validation", "testing"]:
        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        split_name = {"training": "train", "validation": "validation", "testing": "test"}[split]
        mmap_dir = str(split_out / "zh_adversarial_mmap")

        if Path(mmap_dir).exists():
            print(f"[skip] {mmap_dir} 已存在")
            continue

        print(f"[info] 生成 {split} 特征...")
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            split_spectrogram_duration_s=3.2,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=mmap_dir,
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=1
            ),
            batch_size=100,
            verbose=True,
        )

    print("[done] 对抗性负样本特征生成完成")


if __name__ == "__main__":
    main()

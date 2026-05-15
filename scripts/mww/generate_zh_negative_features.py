#!/usr/bin/env python3
"""
将中文对抗性负样本 WAV 转为 MWW 的 mmap 频谱特征格式。

这些负样本会被加入 MWW 训练，解决"你好XX"误触发问题。

用法（Docker 内）：
  python3 generate_zh_negative_features.py \
    --input-dirs /workspace/outputs/cosyvoice_clips/negative_train \
                 /workspace/outputs/cosyvoice_clips_jiuming/negative_train \
    --output-dir /workspace/data/negative_datasets/zh_adversarial/zh_adversarial \
    --data-dir /workspace/data
"""
import argparse
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="中文负样本 WAV 目录（可多个）")
    parser.add_argument("--output-dir", required=True,
                        help="输出 mmap 目录")
    parser.add_argument("--data-dir", default="/workspace/data",
                        help="增强数据根目录")
    parser.add_argument("--split-seed", type=int, default=10)
    args = parser.parse_args()

    # 合并所有输入目录的 WAV 到一个临时目录
    import os, shutil
    merged_dir = "/tmp/zh_neg_merged"
    os.makedirs(merged_dir, exist_ok=True)
    count = 0
    for d in args.input_dirs:
        for wav in sorted(Path(d).glob("*.wav")):
            dst = os.path.join(merged_dir, f"neg_{count:06d}.wav")
            if not os.path.exists(dst):
                shutil.copy2(str(wav), dst)
            count += 1
    print(f"[info] 合并 {count} 条中文负样本")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = Clips(
        input_directory=merged_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=args.split_seed,
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
        impulse_paths=[str(data_dir / "augmentation" / "mit_rirs")],
        background_paths=[
            str(data_dir / "augmentation" / "fma_16k"),
            str(data_dir / "augmentation" / "audioset_16k"),
        ],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split in ["training", "validation", "testing"]:
        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        split_name = {"training": "train", "validation": "validation", "testing": "test"}[split]
        repetition = 2 if split == "training" else 1
        slide_frames = 10 if split != "testing" else 1

        # 用 split_spectrogram_duration_s 切分长音频为短片段
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            split_spectrogram_duration_s=3.2,
            step_ms=10,
        )

        mmap_dir = str(split_out / "zh_adversarial_mmap")
        if Path(mmap_dir).exists():
            print(f"[skip] {mmap_dir} 已存在")
            continue

        print(f"[info] 生成 {split} 特征...")
        RaggedMmap.from_generator(
            out_dir=mmap_dir,
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )

    print("[done] 中文负样本特征生成完成")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
生成频谱特征（RaggedMmap），供 micro-wake-word 训练使用。

与 scripts/05_generate_features.py 逻辑一致，但正样本目录指向真实语音切分结果。

用法：
  python 03_generate_features.py \
    --positive-dir data/positive_augmented/nihao_shushi \
    --data-dir data \
    --output-dir data/generated_augmented_features
"""
import argparse
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=10)
    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure(output_dir)

    n_wavs = len(list(positive_dir.glob("*.wav")))
    print(f"[info] 正样本目录: {positive_dir} ({n_wavs} 条)")

    clips = Clips(
        input_directory=str(positive_dir),
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
        ensure(split_out)

        split_name = "train"
        repetition = 2
        slide_frames = 10

        if split == "validation":
            split_name = "validation"
            repetition = 1
        elif split == "testing":
            split_name = "test"
            repetition = 1
            slide_frames = 1

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=str(split_out / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
    parser.add_argument("--real-voice-dir", default=None,
                        help="真实人声 WAV 目录，会与 TTS 正样本合并使用")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=10)
    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure(output_dir)

    # 如果提供了真实人声目录，先把 WAV 文件复制到正样本目录一起训练
    if args.real_voice_dir:
        import shutil
        real_dir = Path(args.real_voice_dir)
        real_wavs = list(real_dir.glob("*.wav"))
        if real_wavs:
            print(f"[info] 加入真实人声样本 {len(real_wavs)} 条，来自 {real_dir}")
            for wav in real_wavs:
                dst = positive_dir / f"real_{wav.name}"
                shutil.copy2(wav, dst)
        else:
            print(f"[warn] 真实人声目录 {real_dir} 中未找到 WAV 文件")

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

#!/usr/bin/env python3
"""
将 AISHELL-1 中文语音转为 MWW 的 mmap 负样本格式。
同时处理 CosyVoice 生成的无关中文负样本。

用法（Docker 内）：
  python3 generate_aishell_negative_features.py \
    --aishell-dir /workspace/data/aishell/data_aishell/wav \
    --cosyvoice-neg-dir /workspace/outputs/jiuming_correct/negative \
    --output-dir /workspace/data/negative_datasets/zh_chinese/zh_chinese \
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
    parser.add_argument("--aishell-dir", required=True)
    parser.add_argument("--cosyvoice-neg-dirs", nargs="+", default=[])
    parser.add_argument("--musan-speech-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--max-aishell", type=int, default=50000,
                        help="最多使用多少条 AISHELL 样本")
    args = parser.parse_args()

    import os, shutil, random
    random.seed(42)

    # 合并所有中文负样本到临时目录
    merged_dir = "/tmp/zh_neg_all"
    os.makedirs(merged_dir, exist_ok=True)
    count = 0

    # AISHELL-1（随机抽样，避免太多）
    aishell_wavs = list(Path(args.aishell_dir).rglob("*.wav"))
    random.shuffle(aishell_wavs)
    for wav in aishell_wavs[:args.max_aishell]:
        dst = os.path.join(merged_dir, f"aishell_{count:06d}.wav")
        if not os.path.exists(dst):
            shutil.copy2(str(wav), dst)
        count += 1
    print(f"[info] AISHELL-1: {min(len(aishell_wavs), args.max_aishell)} 条")

    # CosyVoice 无关中文负样本
    for d in args.cosyvoice_neg_dirs:
        if os.path.exists(d):
            for wav in sorted(Path(d).glob("*.wav")):
                dst = os.path.join(merged_dir, f"cosy_{count:06d}.wav")
                if not os.path.exists(dst):
                    shutil.copy2(str(wav), dst)
                count += 1
    print(f"[info] 总计: {count} 条中文负样本")

    # MUSAN speech（如果有）
    if args.musan_speech_dir and os.path.exists(args.musan_speech_dir):
        musan_wavs = list(Path(args.musan_speech_dir).rglob("*.wav"))
        for wav in musan_wavs[:5000]:
            dst = os.path.join(merged_dir, f"musan_{count:06d}.wav")
            if not os.path.exists(dst):
                shutil.copy2(str(wav), dst)
            count += 1
        print(f"[info] MUSAN speech: {min(len(musan_wavs), 5000)} 条")

    print(f"[info] 合并总计: {count} 条")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 RIRS_NOISES 和 MUSAN 做增强（如果有）
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

    print(f"[info] 增强: impulse={len(impulse_paths)} dirs, background={len(background_paths)} dirs")

    clips = Clips(
        input_directory=merged_dir,
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
        repetition = 1  # 负样本不需要重复
        mmap_dir = str(split_out / "zh_chinese_mmap")

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
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )

    print("[done] 中文负样本特征生成完成")


if __name__ == "__main__":
    main()

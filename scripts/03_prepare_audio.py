#!/usr/bin/env python3
"""
Prepare augmentation audio files (MIT RIR, AudioSet, FMA) resampled to 16 kHz.
Deliberately avoids datasets.Audio encode path to sidestep torchcodec dependency.
"""
import argparse
import io
import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Audio I/O helpers (no datasets.Audio encode involved)
# ---------------------------------------------------------------------------

def _load_as_mono_16k(src: Path) -> np.ndarray:
    """Load any audio file and return float32 mono array at 16 kHz."""
    import soundfile as sf
    import resampy

    data, sr = sf.read(str(src), always_2d=True)
    # mix down to mono
    data = data.mean(axis=1)
    if sr != 16000:
        data = resampy.resample(data, sr, 16000)
    return data.astype(np.float32)


def _save_wav_16k(arr: np.ndarray, out: Path) -> None:
    pcm = (arr * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(str(out), 16000, pcm)


def convert_dir_to_16k(src_glob: str, out_dir: Path, desc: str = "convert") -> None:
    ensure_dir(out_dir)
    paths = sorted(Path(".").glob(src_glob))
    if not paths:
        print(f"[warn] no files found: {src_glob}")
        return
    for p in tqdm(paths, desc=desc):
        try:
            arr = _load_as_mono_16k(p)
            _save_wav_16k(arr, out_dir / (p.stem + ".wav"))
        except Exception as exc:
            print(f"[warn] skip {p.name}: {exc}")


# ---------------------------------------------------------------------------
# MIT RIR download via huggingface_hub (downloads raw flac, no Audio encode)
# ---------------------------------------------------------------------------

def download_mit_rir(out_dir: Path) -> None:
    ensure_dir(out_dir)
    if any(out_dir.glob("*.wav")):
        print(f"[skip] MIT RIR exists: {out_dir}")
        return

    print("[info] Downloading MIT RIR via huggingface_hub ...")
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id="davidscripka/MIT_environmental_impulse_responses",
            repo_type="dataset",
            ignore_patterns=["*.json", "*.md", "*.parquet"],
        )
        flacs = list(Path(local).rglob("*.flac"))
        wavs  = list(Path(local).rglob("*.wav"))
        files = flacs + wavs
        if not files:
            raise FileNotFoundError("No audio files found in snapshot")
        for f in tqdm(files, desc="mit_rir"):
            arr = _load_as_mono_16k(f)
            _save_wav_16k(arr, out_dir / (f.stem + ".wav"))
    except Exception as exc:
        print(f"[warn] MIT RIR download failed: {exc}")
        print("[warn] Continuing without RIR augmentation.")


# ---------------------------------------------------------------------------
# AudioSet: try HF datasets streaming but read raw bytes, skip Audio encode
# ---------------------------------------------------------------------------

def export_audioset_via_hf(out_dir: Path, max_clips: int = 3000) -> None:
    ensure_dir(out_dir)
    if any(out_dir.glob("*.wav")):
        print(f"[skip] AudioSet 16k exists: {out_dir}")
        return

    print("[info] Trying AudioSet via HF datasets (raw bytes path) ...")
    try:
        import datasets as hf_datasets
        # Use trust_remote_code=False; request the raw bytes config
        ds = hf_datasets.load_dataset(
            "agkphysics/AudioSet",
            "balanced",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        print(f"[warn] AudioSet HF init failed: {exc}")
        print("[warn] Skipping AudioSet; using FMA + MIT RIR only.")
        return

    count = 0
    try:
        import soundfile as sf
        import resampy

        for row in tqdm(ds, desc="audioset_hf"):
            try:
                audio = row.get("audio") or {}
                raw_bytes = audio.get("bytes")
                if not raw_bytes:
                    # some rows carry path instead
                    path_val = audio.get("path")
                    if path_val and Path(path_val).exists():
                        arr = _load_as_mono_16k(Path(path_val))
                    else:
                        continue
                else:
                    data, sr = sf.read(io.BytesIO(raw_bytes), always_2d=True)
                    data = data.mean(axis=1).astype(np.float32)
                    if sr != 16000:
                        data = resampy.resample(data, sr, 16000)
                    arr = data

                _save_wav_16k(arr, out_dir / f"audioset_{count:06d}.wav")
                count += 1
                if count >= max_clips:
                    break
            except Exception as row_exc:
                print(f"[warn] skip row: {row_exc}")
    except Exception as exc:
        print(f"[warn] AudioSet streaming interrupted: {exc}")

    print(f"[info] Exported AudioSet clips: {count}")
    if count == 0:
        print("[warn] No AudioSet clips exported; training will use FMA + MIT RIR only.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    os.chdir(data_dir)

    download_mit_rir(data_dir / "augmentation" / "mit_rirs")

    audioset_16k = data_dir / "augmentation" / "audioset_16k"
    if not any(audioset_16k.glob("*.wav")):
        flacs = list(Path(".").glob("augmentation/audioset/audio/**/*.flac"))
        if flacs:
            convert_dir_to_16k(
                "augmentation/audioset/audio/**/*.flac",
                audioset_16k,
                desc="audioset_local",
            )
        else:
            export_audioset_via_hf(audioset_16k, max_clips=3000)
    else:
        print(f"[skip] AudioSet 16k exists: {audioset_16k}")

    fma_16k = data_dir / "augmentation" / "fma_16k"
    if not any(fma_16k.glob("*.wav")):
        convert_dir_to_16k(
            "augmentation/fma/fma_small/**/*.mp3",
            fma_16k,
            desc="fma",
        )
    else:
        print(f"[skip] FMA 16k exists: {fma_16k}")


if __name__ == "__main__":
    main()

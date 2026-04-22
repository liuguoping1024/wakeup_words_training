#!/usr/bin/env python3
"""
合并多个 TTS 来源的正样本到 MWW 训练目录。
同时把 TTS 负样本转换为 MWW 可用的负样本特征。

用法：
  python3 scripts/mww/merge_tts_samples.py \
    --keyword-id jiuming \
    --tts-pos-dir data/tts_positive/jiuming \
    --real-dir data/real_voices_jiuming \
    --output-dir data/positive_raw/jiuming \
    --clean
"""
import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-id", required=True)
    parser.add_argument("--tts-pos-dir", required=True,
                        help="TTS 正样本目录（edge + piper + cosyvoice 混合）")
    parser.add_argument("--real-dir", default=None,
                        help="真实录音目录")
    parser.add_argument("--output-dir", required=True,
                        help="MWW 训练用正样本目录")
    parser.add_argument("--clean", action="store_true",
                        help="清空输出目录后重新合并")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.clean and out.exists():
        log.info(f"清空 {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    count = 0

    # 1. 复制真实录音
    if args.real_dir and os.path.isdir(args.real_dir):
        real_wavs = sorted(Path(args.real_dir).glob("*.wav"))
        log.info(f"真实录音: {len(real_wavs)} 条")
        for wav in real_wavs:
            dst = out / f"real_{wav.name}"
            if not dst.exists():
                shutil.copy2(wav, dst)
            count += 1

    # 2. 复制 TTS 正样本（保留原始前缀以区分来源）
    tts_dir = Path(args.tts_pos_dir)
    if tts_dir.exists():
        tts_wavs = sorted(tts_dir.glob("*.wav"))
        log.info(f"TTS 正样本: {len(tts_wavs)} 条")
        for wav in tts_wavs:
            dst = out / wav.name
            if not dst.exists():
                shutil.copy2(wav, dst)
            count += 1

    # 清理残留 mp3
    for mp3 in out.glob("*.mp3"):
        mp3.unlink()
        log.info(f"  删除残留 mp3: {mp3.name}")

    log.info(f"=== 合并完成: {count} 条正样本在 {out} ===")

    # 统计各来源
    all_wavs = list(out.glob("*.wav"))
    sources = {}
    for w in all_wavs:
        prefix = w.name.split("_")[0]
        sources[prefix] = sources.get(prefix, 0) + 1
    for src, n in sorted(sources.items()):
        log.info(f"  {src}: {n} 条")


if __name__ == "__main__":
    main()

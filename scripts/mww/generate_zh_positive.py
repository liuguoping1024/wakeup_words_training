#!/usr/bin/env python3
"""
用 edge-tts 生成中文唤醒词的 TTS 正样本。
在宿主机上运行（不在 Docker 内）。

用法：
  python3.12 scripts/mww/generate_zh_positive.py --phrase "救命" --output-dir data/positive_raw/jiuming --n-samples 2000
  python3.12 scripts/mww/generate_zh_positive.py --phrase "你好树实" --output-dir data/positive_raw/nihao_shushi_tts --n-samples 2000
"""
import argparse
import asyncio
import os
import random
import subprocess
import sys

import edge_tts

ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural", "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaozhenNeural",
    "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural", "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]

RATES = ["-30%", "-20%", "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+20%", "+30%"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", required=True, help="唤醒词文本")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    existing = len([f for f in os.listdir(args.output_dir) if f.endswith(".wav") and f.startswith("tts_")])
    if existing >= args.n_samples * 0.9:
        print(f"Already have {existing}/{args.n_samples} TTS clips, skipping")
        return

    print(f"Generating {args.n_samples} TTS clips for '{args.phrase}'...")
    print(f"Using {len(ZH_VOICES)} voices × {len(RATES)} rates")

    count = existing
    for i in range(existing, args.n_samples):
        voice = random.choice(ZH_VOICES)
        rate = random.choice(RATES)
        mp3 = os.path.join(args.output_dir, f"tts_{i:05d}.mp3")
        wav = os.path.join(args.output_dir, f"tts_{i:05d}.wav")

        if os.path.exists(wav):
            count += 1
            continue

        for attempt in range(3):
            try:
                c = edge_tts.Communicate(args.phrase, voice, rate=rate)
                await c.save(mp3)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                    check=True
                )
                os.remove(mp3)
                count += 1
                break
            except Exception as e:
                await asyncio.sleep(1 + attempt * 2)

        if i % 10 == 0:
            await asyncio.sleep(0.3)

        if count % 200 == 0 and count > 0:
            print(f"  {count}/{args.n_samples}", flush=True)

    print(f"Done: {count} TTS clips in {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

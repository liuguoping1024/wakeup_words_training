#!/usr/bin/env python3
"""
用 edge-tts 生成中文 TTS 样本（正样本 + 对抗性负样本）。
在宿主机上运行（不在 Docker 内），避免容器网络问题。

用法：
  python3.12 scripts/oww/generate_zh_tts.py
"""
import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path

import edge_tts

TARGET_PHRASE = "你好树实"

ADVERSARIAL_PHRASES = [
    "你好叔叔", "你好书式", "你好树枝", "你好属实",
    "你好舒适", "你好数十", "你好输食", "你好书市",
    "你好树石", "你好鼠食", "你好束时", "你好述事",
    "你好", "树实", "你好啊", "你好吗",
    "你好世界", "你好师傅", "你好同学", "你好朋友",
    "泥好树实", "拟好树实", "你浩树实", "你号树实",
    "小爱同学", "天猫精灵", "你好小度",
    "打开灯", "关闭窗帘", "今天天气", "几点了",
    "播放音乐", "设个闹钟", "你好吃吗", "你好看吗",
    "你好厉害", "你好棒", "你好快", "你好慢",
    "树上有鸟", "实在太好", "书本在哪", "数学题",
]

ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural", "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaozhenNeural",
    "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural", "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]

RATES = ["-20%", "-10%", "-5%", "+0%", "+5%", "+10%", "+20%"]


async def gen_clip(text, voice, mp3_path, rate="+0%"):
    c = edge_tts.Communicate(text, voice, rate=rate)
    await c.save(mp3_path)


async def generate_batch(output_dir, texts_or_phrase, n_samples, prefix, is_single_phrase=False):
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_samples * 0.9:
        print(f"  Already have {existing}/{n_samples} clips, skipping")
        return

    print(f"  Generating {n_samples} clips in {output_dir}...")
    count = existing

    for i in range(existing, n_samples):
        text = texts_or_phrase if is_single_phrase else random.choice(texts_or_phrase)
        voice = random.choice(ZH_VOICES)
        rate = random.choice(RATES)
        mp3 = os.path.join(output_dir, f"{prefix}_{i:05d}.mp3")
        wav = os.path.join(output_dir, f"{prefix}_{i:05d}.wav")

        if os.path.exists(wav):
            count += 1
            continue

        for attempt in range(3):
            try:
                await gen_clip(text, voice, mp3, rate=rate)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                    check=True
                )
                os.remove(mp3)
                count += 1
                break
            except Exception as e:
                await asyncio.sleep(2 + attempt * 3)

        if i % 10 == 0:
            await asyncio.sleep(0.3)

        if count % 200 == 0 and count > existing:
            print(f"    {count}/{n_samples}")

    print(f"  Done: {count} clips")


async def main():
    random.seed(42)
    base = "outputs/oww/nihao_shushi"

    print("=== Generating TTS positive clips (你好树实) ===")
    await generate_batch(
        f"{base}/tts_positive", TARGET_PHRASE, 2000, "tts_pos", is_single_phrase=True
    )

    print("=== Generating adversarial negative clips (train) ===")
    await generate_batch(
        f"{base}/negative_train", ADVERSARIAL_PHRASES, 5000, "neg"
    )

    print("=== Generating adversarial negative clips (test) ===")
    await generate_batch(
        f"{base}/negative_test", ADVERSARIAL_PHRASES, 1000, "neg"
    )

    print("=== All TTS generation complete ===")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
用 edge-tts (19种中文声音) 生成正样本和对抗性负样本。
在宿主机上运行（需要网络）。

生成完后，再用 prepare_mixed_dataset.py 把 edge-tts 正样本 + 真实录音 + Piper TTS 混合。

用法：
  python3 scripts/oww/generate_all_zh_clips.py \
    --keyword "你好树实" --output-dir outputs/oww/nihao_shushi_v3 \
    --n-pos 10000 --n-neg 10000
"""
import argparse
import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path

ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural", "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaozhenNeural",
    "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural", "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]

ADVERSARIAL_PHRASES = [
    "你好叔叔", "你好书式", "你好树枝", "你好属实",
    "你好舒适", "你好数十", "你好书市", "你好熟食",
    "你好竖石", "你好疏食", "你好殊实",
    "你好", "树实", "你好啊", "你好吗",
    "你好世界", "你好师傅", "你好同学", "你好朋友",
    "泥好树实", "拟好树实", "你浩树实", "你号树实",
    "小爱同学", "天猫精灵", "你好小度",
    "你好吃吗", "你好看吗", "你好厉害", "你好棒",
    "树上有鸟", "实在太好", "书本在哪", "数学题",
    "打开灯", "关闭窗帘", "今天天气", "播放音乐",
    "谢谢", "再见", "对不起", "没关系", "早上好",
]

RATES = ["-25%", "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+25%"]


async def gen_one(text, voice, mp3_path, wav_path, rate="+0%"):
    import edge_tts
    c = edge_tts.Communicate(text, voice, rate=rate)
    await c.save(mp3_path)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path],
        check=True
    )
    os.remove(mp3_path)


async def generate_batch(output_dir, texts, n_target, prefix, is_single_text=False):
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        print(f"  已有 {existing}/{n_target}，跳过")
        return existing

    print(f"  生成 {n_target - existing} 条到 {output_dir}")
    count = existing
    fails = 0

    for i in range(existing, n_target):
        text = texts if is_single_text else random.choice(texts)
        voice = random.choice(ZH_VOICES)
        rate = random.choice(RATES)
        mp3 = os.path.join(output_dir, f"{prefix}_{i:06d}.mp3")
        wav = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")

        if os.path.exists(wav):
            count += 1
            continue

        for attempt in range(3):
            try:
                await gen_one(text, voice, mp3, wav, rate=rate)
                count += 1
                break
            except Exception:
                fails += 1
                await asyncio.sleep(1 + attempt * 2)

        if i % 10 == 0:
            await asyncio.sleep(0.3)
        if count % 500 == 0 and count > existing:
            print(f"    进度: {count}/{n_target} (失败: {fails})")

    print(f"  完成: {count} 条 (失败: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="你好树实")
    parser.add_argument("--output-dir", default="outputs/oww/nihao_shushi_v3")
    parser.add_argument("--n-pos", type=int, default=10000, help="edge-tts 正样本数")
    parser.add_argument("--n-neg-train", type=int, default=10000, help="对抗性负样本训练集")
    parser.add_argument("--n-neg-test", type=int, default=2000, help="对抗性负样本测试集")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"=== edge-tts 正样本 ({args.keyword}) ===")
    asyncio.run(generate_batch(
        os.path.join(args.output_dir, "edge_tts_positive"),
        args.keyword, args.n_pos, "pos", is_single_text=True
    ))

    print(f"\n=== 对抗性负样本 (train) ===")
    asyncio.run(generate_batch(
        os.path.join(args.output_dir, "negative_train"),
        ADVERSARIAL_PHRASES, args.n_neg_train, "neg"
    ))

    print(f"\n=== 对抗性负样本 (test) ===")
    asyncio.run(generate_batch(
        os.path.join(args.output_dir, "negative_test"),
        ADVERSARIAL_PHRASES, args.n_neg_test, "neg"
    ))

    print("\n全部完成!")


if __name__ == "__main__":
    main()

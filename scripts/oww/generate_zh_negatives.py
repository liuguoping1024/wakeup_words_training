#!/usr/bin/env python3
"""
在宿主机上用 edge-tts 生成中文对抗性负样本。
需要网络访问（Azure TTS 服务），不在 Docker 内运行。

用法：
  python3 scripts/oww/generate_zh_negatives.py \
    --keyword "你好树实" --output-dir outputs/oww/nihao_shushi \
    --n-train 5000 --n-test 1000
"""
import argparse
import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path

# ── 对抗性负样本短语库 ──
# 按类别组织，确保覆盖各种混淆场景
ADVERSARIAL_PHRASES = {
    "nihao_shushi": [
        # 声母/韵母相近（最重要的对抗样本）
        "你好叔叔", "你好书式", "你好树枝", "你好属实",
        "你好舒适", "你好数十", "你好输食", "你好书市",
        "你好树石", "你好鼠食", "你好束时", "你好述事",
        "你好竖石", "你好疏食", "你好殊实", "你好熟食",
        # 部分匹配
        "你好", "树实", "你好啊", "你好吗",
        "你好世界", "你好师傅", "你好同学", "你好朋友",
        # 声调变化
        "泥好树实", "拟好树实", "你浩树实", "你号树实",
        # 其他唤醒词
        "小爱同学", "天猫精灵", "你好小度",
        # 日常对话
        "你好吃吗", "你好看吗", "你好厉害", "你好棒",
        "你好快", "你好慢", "树上有鸟", "实在太好",
        "书本在哪", "数学题", "打开灯", "关闭窗帘",
        "今天天气", "几点了", "播放音乐", "设个闹钟",
        "你好大", "你好小", "你好冷", "你好热",
    ],
    "jiuming": [
        # 声母/韵母相近
        "就命", "旧命", "九明", "酒名", "久鸣",
        "纠明", "揪命", "究命", "九命", "酒命",
        # 部分匹配
        "救", "命", "救人", "命令",
        # 日常
        "救火", "救护车", "生命", "革命",
        "小命", "拼命", "要命", "玩命",
        "打开灯", "关闭窗帘", "今天天气",
        "小爱同学", "天猫精灵", "你好小度",
    ],
}

# 通用负样本（所有中文唤醒词共用）
COMMON_NEGATIVES = [
    "你好", "谢谢", "再见", "对不起", "没关系",
    "早上好", "晚上好", "吃饭了吗", "去哪里",
    "什么时候", "怎么办", "为什么", "好的",
    "一二三四五", "上山打老虎", "小星星",
    "今天星期几", "现在几点", "明天天气怎么样",
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

RATES = ["-25%", "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+25%"]


async def gen_one_clip(text, voice, mp3_path, wav_path, rate="+0%"):
    """生成单条 TTS 语音并转为 16kHz WAV。"""
    import edge_tts
    c = edge_tts.Communicate(text, voice, rate=rate)
    await c.save(mp3_path)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path],
        check=True
    )
    os.remove(mp3_path)


async def generate_clips(output_dir, n_target, phrases, prefix="neg"):
    """批量生成对抗性负样本。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        print(f"  已有 {existing}/{n_target} 条，跳过")
        return existing

    print(f"  生成 {n_target - existing} 条到 {output_dir}")
    count = existing
    fails = 0

    for i in range(existing, n_target):
        text = random.choice(phrases)
        voice = random.choice(ZH_VOICES)
        rate = random.choice(RATES)
        mp3 = os.path.join(output_dir, f"{prefix}_{i:06d}.mp3")
        wav = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")

        if os.path.exists(wav):
            count += 1
            continue

        for attempt in range(3):
            try:
                await gen_one_clip(text, voice, mp3, wav, rate=rate)
                count += 1
                break
            except Exception as e:
                fails += 1
                await asyncio.sleep(1 + attempt * 2)

        # 避免被限流
        if i % 10 == 0:
            await asyncio.sleep(0.3)
        if count % 500 == 0 and count > 0:
            print(f"    进度: {count}/{n_target} (失败: {fails})")

    print(f"  完成: {count} 条 (失败: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser(description="生成中文对抗性负样本")
    parser.add_argument("--keyword", default="你好树实")
    parser.add_argument("--output-dir", default="outputs/oww/nihao_shushi")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 根据 keyword 选择对抗短语
    keyword_key = None
    for key in ADVERSARIAL_PHRASES:
        if key in args.output_dir or key in args.keyword:
            keyword_key = key
            break

    if keyword_key:
        phrases = ADVERSARIAL_PHRASES[keyword_key] + COMMON_NEGATIVES
    else:
        # 未知唤醒词，只用通用负样本
        phrases = COMMON_NEGATIVES
        print(f"警告: 未找到 '{args.keyword}' 的专用对抗短语，仅使用通用负样本")

    neg_train_dir = os.path.join(args.output_dir, "negative_train")
    neg_test_dir = os.path.join(args.output_dir, "negative_test")

    print(f"对抗短语库: {len(phrases)} 条")
    print(f"目标: {args.n_train} train + {args.n_test} test")
    print()

    print("生成 train 负样本:")
    asyncio.run(generate_clips(neg_train_dir, args.n_train, phrases, prefix="neg"))
    print()
    print("生成 test 负样本:")
    asyncio.run(generate_clips(neg_test_dir, args.n_test, phrases, prefix="neg"))
    print()
    print("全部完成!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
用 edge-tts 生成 help help 的正样本 + 对抗性负样本 + 测试集。
在宿主机运行（python3.12）。
"""
import asyncio, os, subprocess, random, json, sys

# ── 正样本词表 ──
POSITIVE_PHRASES = [
    "help help",
    "help help!",
    "help! help!",
    "help help please",
    "help help me",
    "somebody help help",
    "help help help",
]

# ── 对抗性负样本词表 ──
NEGATIVE_PHRASES = [
    # 含 help 但不是 help help
    "help me", "help us", "help him", "help her",
    "can you help", "please help", "I need help",
    "help yourself", "helpful", "helpless",
    "help me please", "help is coming",
    # 类似发音
    "health", "held", "yelp", "kelp", "self", "shelf",
    "hell", "hello", "helicopter",
    # 重复音节但不是 help
    "hip hop", "tick tock", "flip flop", "clip clop",
    "knock knock", "beep beep", "drip drop",
    # 日常英文
    "good morning", "good night", "thank you", "excuse me",
    "turn on the light", "play music", "set an alarm",
    "what time is it", "how are you", "I'm fine",
    "open the door", "close the window",
    "hey siri", "ok google", "alexa",
    "yes", "no", "maybe", "please", "sorry",
    "one two three four five",
    "the weather is nice today",
    "I'm going home", "see you later",
    # 中文（防止中文误触发）
    "你好", "谢谢", "救命", "你好树实",
]

# 英文声音
EN_VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
]

# 中文声音（用于中文负样本）
ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural",
]


async def generate_one(text, voice, output_path):
    import edge_tts
    mp3_path = output_path.replace(".wav", ".mp3")
    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(mp3_path)
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", output_path
        ], capture_output=True, timeout=10)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 500
    except Exception:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return False


async def generate_batch(texts, voices, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for text in texts:
        for voice in voices:
            safe_text = text.replace(" ", "_").replace("!", "").replace("'", "")
            voice_short = voice.split("-")[-1].replace("Neural", "")
            fname = f"{prefix}_{safe_text}_{voice_short}.wav"
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
                continue
            tasks.append((text, voice, fpath))

    random.shuffle(tasks)
    count, fails = 0, 0
    for text, voice, fpath in tasks:
        ok = await generate_one(text, voice, fpath)
        if ok:
            count += 1
        else:
            fails += 1
        if (count + fails) % 50 == 0 and (count + fails) > 0:
            print(f"  [{prefix}] {count} ok, {fails} fail / {len(tasks)}")

    existing = len([f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".wav")])
    print(f"  [{prefix}] 完成: 新增 {count}, 失败 {fails}, 总计 {existing}")
    return existing


async def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/help_help_edgetts"

    print("=== 生成正样本 ===")
    pos_count = await generate_batch(
        POSITIVE_PHRASES, EN_VOICES,
        os.path.join(base_dir, "positive"), "pos"
    )

    print("\n=== 生成对抗性负样本（英文）===")
    neg_en = [p for p in NEGATIVE_PHRASES if not any(ord(c) > 127 for c in p)]
    neg_en_count = await generate_batch(
        neg_en, EN_VOICES,
        os.path.join(base_dir, "negative"), "neg"
    )

    print("\n=== 生成对抗性负样本（中文）===")
    neg_zh = [p for p in NEGATIVE_PHRASES if any(ord(c) > 127 for c in p)]
    neg_zh_count = await generate_batch(
        neg_zh, ZH_VOICES,
        os.path.join(base_dir, "negative"), "negzh"
    )

    print("\n=== 生成测试集 ===")
    # 测试集用不同的声音子集，避免和训练集重叠
    test_voices = EN_VOICES[:5]
    test_pos = await generate_batch(
        POSITIVE_PHRASES[:3], test_voices,
        os.path.join(base_dir, "test"), "tpos"
    )
    test_neg_phrases = [
        "help me", "help us", "please help", "hello",
        "health", "held", "yelp", "hip hop",
        "good morning", "play music", "hey siri", "ok google",
    ]
    test_neg = await generate_batch(
        test_neg_phrases, test_voices,
        os.path.join(base_dir, "test"), "tneg"
    )

    # 建测试集 manifest
    test_dir = os.path.join(base_dir, "test")
    manifest = []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".wav"):
            continue
        if f.startswith("tpos_"):
            cat = "positive"
        elif f.startswith("tneg_"):
            cat = "negative"
        else:
            continue
        manifest.append({"file": f, "text": f.split("_", 1)[1].rsplit("_", 1)[0], "category": cat, "voice": "edgetts"})
    with open(os.path.join(test_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\n=== 总结 ===")
    print(f"正样本: {pos_count}")
    print(f"负样本: {neg_en_count + neg_zh_count}")
    print(f"测试集: {test_pos + test_neg}")


if __name__ == "__main__":
    asyncio.run(main())

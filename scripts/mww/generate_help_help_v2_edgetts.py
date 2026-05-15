#!/usr/bin/env python3
"""
help help v2 edge-tts 生成 — 大幅增加英文声音和对抗性词汇。
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
    "help help please help",
]

# ── 负样本词表（v2 大幅扩展对抗性）──
NEGATIVE_PHRASES = [
    # v1 已有：含 help 但不是 help help
    "help me", "help us", "help him", "help her",
    "can you help", "please help", "I need help",
    "help yourself", "helpful", "helpless",
    "help me please", "help is coming",
    # v1 已有：类似发音
    "health", "healthy", "healthcare",
    "held", "yelp", "kelp", "self", "shelf",
    "hell", "hello", "helicopter",
    "helm", "helmet",
    # v2 新增：-el- / -elp / -elt / -elf 对抗
    "whelp", "swelter", "smelter",
    "herald", "harold",
    "shell", "smell", "spell", "swell",
    "belt", "felt", "melt", "pelt",
    "yellow", "fellow", "mellow",
    # v2 新增：重复音节但不是 help
    "hip hop", "tick tock", "flip flop", "clip clop",
    "knock knock", "beep beep", "drip drop",
    "ding dong", "ping pong", "ding ding",
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
    # 紧急词（避免误触发）
    "fire fire", "emergency", "danger",
]

# 英文声音（全部 22 个）
EN_VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural", "en-US-AnaNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
    "en-HK-SamNeural", "en-HK-YanNeural",
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
        if (count + fails) % 100 == 0 and (count + fails) > 0:
            print(f"  [{prefix}] {count} ok, {fails} fail / {len(tasks)}")

    existing = len([f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".wav")])
    print(f"  [{prefix}] done: new={count}, fail={fails}, total={existing}")
    return existing


async def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/help_help_edgetts_v2"

    print(f"=== 正样本（{len(POSITIVE_PHRASES)} 词 × {len(EN_VOICES)} 声音）===")
    pos = await generate_batch(POSITIVE_PHRASES, EN_VOICES,
                               os.path.join(base_dir, "positive"), "pos")

    print(f"\n=== 负样本（{len(NEGATIVE_PHRASES)} 词 × {len(EN_VOICES)} 声音）===")
    neg = await generate_batch(NEGATIVE_PHRASES, EN_VOICES,
                               os.path.join(base_dir, "negative"), "neg")

    print("\n=== 测试集 ===")
    test_voices = EN_VOICES[:6]
    test_pos = await generate_batch(
        ["help help", "help help!", "help help please"],
        test_voices, os.path.join(base_dir, "test"), "tpos")
    test_neg = await generate_batch(
        ["help me", "help us", "please help", "hello", "health", "healthy",
         "held", "yelp", "hip hop", "good morning", "play music", "hey siri",
         "shell", "swell", "helm", "helmet", "fire fire"],
        test_voices, os.path.join(base_dir, "test"), "tneg")

    test_dir = os.path.join(base_dir, "test")
    manifest = []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".wav"):
            continue
        cat = "positive" if f.startswith("tpos_") else "negative" if f.startswith("tneg_") else None
        if not cat:
            continue
        text = f.split("_", 1)[1].rsplit("_", 1)[0]
        manifest.append({"file": f, "text": text, "category": cat, "voice": "edgetts"})
    with open(os.path.join(test_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\npos: {pos}, neg: {neg}, test: {test_pos + test_neg}")


if __name__ == "__main__":
    asyncio.run(main())

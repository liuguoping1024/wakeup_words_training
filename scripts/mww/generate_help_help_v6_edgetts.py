#!/usr/bin/env python3
"""
help help v6 edge-tts 生成 — 严格正样本 + 扩展负样本（不变）。
"""
import asyncio, os, subprocess, random, json, sys

POSITIVE_PHRASES = [
    "help help",
    "help help!",
    "help! help!",
]

NEGATIVE_PHRASES = [
    # A. help 相关
    "help me", "help him", "help her", "help us", "help them", "help you",
    "help out", "help up", "help off",
    "helpful", "helpless", "helper", "helping",
    "helped me", "helped him",
    "helm", "helmet", "helmsman",
    # B. hel- / -elp / -ell 音近
    "hello", "hell", "hellish",
    "yelp", "yelping", "yelped",
    "kelp", "welp", "whelp",
    "shelf", "shell", "smell", "spell", "swell", "dwell",
    "fell", "tell", "sell", "bell", "cell", "well", "yell",
    "smelly", "belly", "jelly", "telly", "deli",
    # C. -eld / -elt / -elf / -elm
    "held", "melt", "felt", "belt", "pelt", "welt",
    "self", "myself", "yourself", "himself", "herself", "itself",
    "elm", "realm", "whelm",
    # D. 重复音节
    "hip hop", "tick tock", "flip flop", "clip clop", "drip drop",
    "knock knock", "beep beep", "ding dong", "ping pong", "ding ding",
    "chit chat", "willy nilly",
    "bye bye", "night night", "nom nom",
    "so so", "no no", "tut tut",
    # E. 日常双音节指令
    "good morning", "good night", "good evening", "good day", "good bye",
    "thank you", "thanks a lot", "excuse me", "pardon me",
    "come on", "look out", "stand up", "sit down",
    "turn on", "turn off", "open up", "close up",
    "go home", "go away", "see you", "catch you",
    "calm down", "hang on", "hold on", "wait up",
    "shut up", "speak up", "back up", "cheer up",
    # F. 智能助手 / 紧急词
    "hey siri", "ok google", "hey alexa", "hey cortana",
    "call mom", "call dad", "call police", "call doctor",
    "fire fire", "danger danger",
    "emergency", "ambulance",
    "I'm okay", "I'm fine", "not okay",
    # G. 其他对抗
    "yes no", "okay sure",
    "one two", "two three", "three four", "four five",
    "really really", "very very", "super duper",
    "maybe maybe", "please please", "thank thank",
    "sorry sorry",
]

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


async def generate_one(text, voice, output_path, rate="-10%"):
    """rate: 语速 e.g. '-10%' 慢一点, '0%' 正常"""
    import edge_tts
    mp3_path = output_path.replace(".wav", ".mp3")
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate)
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


async def generate_batch(texts, voices, output_dir, prefix, repeat=1, rate="-15%"):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for text in texts:
        for voice in voices:
            for rep in range(repeat):
                safe_text = text.replace(" ", "_").replace("!", "").replace("'", "")
                voice_short = voice.split("-")[-1].replace("Neural", "")
                fname = f"{prefix}_{safe_text}_{voice_short}_{rep}.wav"
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
                    continue
                tasks.append((text, voice, fpath))

    random.shuffle(tasks)
    count, fails = 0, 0
    for text, voice, fpath in tasks:
        ok = await generate_one(text, voice, fpath, rate=rate)
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
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/help_help_edgetts_v6"

    # 正样本：3 词 × 22 声音 × 10 repeat = 660 条，慢速 -15%
    print(f"=== 正样本（rate=-15% 慢速）===")
    pos = await generate_batch(POSITIVE_PHRASES, EN_VOICES,
                               os.path.join(base_dir, "positive"), "pos",
                               repeat=10, rate="-15%")

    # 负样本：~120 词 × 22 声音 = 2640 条，正常语速
    print(f"\n=== 负样本（rate=+0% 正常）===")
    neg = await generate_batch(NEGATIVE_PHRASES, EN_VOICES,
                               os.path.join(base_dir, "negative"), "neg",
                               repeat=1, rate="+0%")

    # 测试集
    print("\n=== 测试集 ===")
    test_voices = EN_VOICES[:6]
    test_pos = await generate_batch(
        POSITIVE_PHRASES, test_voices,
        os.path.join(base_dir, "test"), "tpos",
        repeat=2, rate="-15%")
    test_neg = await generate_batch(
        ["help me", "help us", "please help", "hello", "health", "healthy",
         "held", "yelp", "hip hop", "good morning", "play music", "hey siri",
         "shell", "swell", "helm", "helmet", "fire fire",
         "helpful", "helpless", "myself"],
        test_voices, os.path.join(base_dir, "test"), "tneg",
        repeat=1, rate="+0%")

    test_dir = os.path.join(base_dir, "test")
    manifest = []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".wav"):
            continue
        cat = "positive" if f.startswith("tpos_") else "negative" if f.startswith("tneg_") else None
        if not cat:
            continue
        text = f.split("_", 1)[1].rsplit("_", 1)[0].rsplit("_", 1)[0]
        manifest.append({"file": f, "text": text, "category": cat, "voice": "edgetts"})
    with open(os.path.join(test_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\npos: {pos}, neg: {neg}, test: {test_pos + test_neg}")


if __name__ == "__main__":
    asyncio.run(main())

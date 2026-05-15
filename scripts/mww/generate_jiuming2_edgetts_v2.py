#!/usr/bin/env python3
"""
救命救命 v2 edge-tts 生成 — 大幅增加数量和多样性。
用全部 19 个中文声音，每个变体都生成。
"""
import asyncio, os, subprocess, random, json, sys

# ── 正样本词表（40+ 变体）──
POSITIVE_PHRASES = [
    # 标准
    "救命救命", "救命救命啊", "救命救命呀", "快救命救命",
    "来人救命救命", "救命救命救命",
    # 声母变体
    "久名久名", "揪命揪命", "纠命纠命", "九命九命",
    "究命究命", "酒命酒命",
    # 韵母变体
    "救民救民", "久民久民", "揪民揪民",
    "救名救名", "久名救命", "救命久名",
    # 混合（前后不一致）
    "救民救命", "救命救民", "久名救民", "救民久名",
    "九命救命", "救命九命", "纠命救命", "救命纠命",
    # 带语气
    "救命救命啊啊", "救命救命快来",
    # 酒名/酒命 变体
    "酒名酒名", "酒命救命", "救命酒命",
]

# ── 负样本词表（50+ 对抗性）──
NEGATIVE_PHRASES = [
    # 单次救命（最关键！）
    "救命", "救命啊", "快救命", "来人救命",
    "久名", "救民", "揪命",
    # 含-ming重复但不是救命
    "说明说明", "文明文明", "聪明聪明", "革命革命",
    "光明光明", "生命生命", "要命要命", "拼命拼命",
    "证明证明", "说明", "聪明", "革命",
    # 含-jiu重复
    "九九", "就是就是", "酒酒",
    # 结构相似
    "救火救火", "救人救人", "救护车救护车",
    "要命要命", "玩命玩命", "拼命拼命",
    # 日常中文
    "你好", "谢谢", "再见", "对不起",
    "打开灯", "关闭窗帘", "播放音乐",
    "你好树实", "小爱同学", "天猫精灵",
    "你好百度", "你好谷歌",
    "今天天气怎么样", "几点了",
    "早上好", "晚上好", "吃饭了吗",
]

# 全部 19 个中文声音
ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural",
    "zh-CN-YunyangNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaochenNeural", "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural",
    "zh-CN-XiaomoNeural", "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural",
    "zh-CN-XiaoshuangNeural", "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural",
    "zh-CN-XiaoyouNeural", "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural",
    "zh-CN-YunxiaNeural", "zh-CN-YunzeNeural",
]

# 测试用声音子集
TEST_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural",
    "zh-CN-YunxiNeural", "zh-CN-YunyangNeural", "zh-CN-YunjianNeural",
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
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/jiuming2_edgetts_v2"

    print("=== 正样本（全部 19 声音 × 30 变体）===")
    pos_count = await generate_batch(
        POSITIVE_PHRASES, ZH_VOICES,
        os.path.join(base_dir, "positive"), "pos"
    )

    print("\n=== 负样本（全部 19 声音 × 50 变体）===")
    neg_count = await generate_batch(
        NEGATIVE_PHRASES, ZH_VOICES,
        os.path.join(base_dir, "negative"), "neg"
    )

    print("\n=== 测试集 ===")
    test_pos_phrases = ["救命救命", "久名久名", "救命救命啊", "救民救命"]
    test_neg_phrases = [
        "救命", "救火", "救火救火", "聪明", "说明说明",
        "革命", "要命要命", "你好", "播放音乐", "你好百度",
        "救民", "久名",
    ]
    test_pos = await generate_batch(
        test_pos_phrases, TEST_VOICES,
        os.path.join(base_dir, "test"), "tpos"
    )
    test_neg = await generate_batch(
        test_neg_phrases, TEST_VOICES,
        os.path.join(base_dir, "test"), "tneg"
    )

    # 建 manifest
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
        text = f.split("_", 1)[1].rsplit("_", 1)[0]
        manifest.append({"file": f, "text": text, "category": cat, "voice": "edgetts"})
    with open(os.path.join(test_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\npos: {pos_count}, neg: {neg_count}, test: {test_pos + test_neg}")


if __name__ == "__main__":
    asyncio.run(main())

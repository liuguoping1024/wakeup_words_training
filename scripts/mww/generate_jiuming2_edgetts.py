#!/usr/bin/env python3
"""
用 edge-tts 生成"救命救命"正样本 + 对抗性负样本 + 测试集。
宿主机运行（python3.12）。
"""
import asyncio, os, subprocess, random, json, sys

# ── 正样本：救命救命 + 方言变体 ──
POSITIVE_PHRASES = [
    # 标准
    "救命救命", "救命救命啊", "救命救命呀", "快救命救命",
    "来人救命救命", "救命救命救命",
    # 声母变体
    "久名久名", "揪命揪命", "纠命纠命", "九命九命", "究命究命",
    # 韵母变体 (ming→min)
    "救民救民", "久民久民", "揪民揪民", "纠民纠民",
    # 混合（前后发音不一致）
    "救命久名", "久名救命", "救民救命", "救命救民",
    "揪命救命", "救命纠命", "九命救命", "救命九命",
    # 带语气
    "救命救命啊救命", "救命啊救命", "救命呀救命",
    # 酒命变体
    "酒命酒命", "酒名酒名",
]

# ── 负样本 ──
NEGATIVE_PHRASES = [
    # ★ 单次救命（最关键 — 只说一次不应触发）
    "救命", "救命啊", "快救命", "来人救命", "救命呀",
    "久名", "揪命", "纠命", "九命", "救民",
    # 含-ming重复但不是救命
    "说明说明", "文明文明", "聪明聪明", "革命革命",
    "生命生命", "光明光明", "证明证明", "要命要命",
    "拼命拼命", "玩命玩命",
    # 含-jiu重复
    "九九", "就是就是", "酒酒",
    # 部分匹配
    "救火救火", "救人救人", "救护车救护车",
    "救援救援", "救助救助",
    # 日常中文
    "你好", "谢谢", "再见", "对不起", "没关系",
    "打开灯", "关闭窗帘", "播放音乐", "设个闹钟",
    "今天天气怎么样", "几点了",
    "你好树实", "小爱同学", "天猫精灵", "你好小度",
    "你好百度", "你好谷歌",
    "着火了", "危险", "注意安全", "帮帮我", "快跑",
    "早上好", "晚上好", "吃饭了吗",
    "一二三四五", "六七八九十",
]

ZH_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural",
    "zh-CN-YunyangNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaochenNeural", "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural",
    "zh-CN-XiaomoNeural", "zh-CN-XiaoqiuNeural",
    "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural", "zh-CN-XiaoxuanNeural",
    "zh-CN-XiaoyanNeural", "zh-CN-XiaoyouNeural",
    "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural",
    "zh-CN-YunxiaNeural", "zh-CN-YunzeNeural",
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
    print(f"  [{prefix}] done: new={count}, fail={fails}, total={existing}")
    return existing


async def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/jiuming2_edgetts"

    print("=== 正样本 ===")
    await generate_batch(POSITIVE_PHRASES, ZH_VOICES, os.path.join(base_dir, "positive"), "pos")

    print("\n=== 负样本 ===")
    await generate_batch(NEGATIVE_PHRASES, ZH_VOICES, os.path.join(base_dir, "negative"), "neg")

    print("\n=== 测试集 ===")
    test_voices = ZH_VOICES[:5]
    test_pos_phrases = ["救命救命", "久名久名", "救民救命", "救命救命啊"]
    await generate_batch(test_pos_phrases, test_voices, os.path.join(base_dir, "test"), "tpos")

    test_neg_phrases = [
        "救命", "久名", "救民", "救火", "革命", "聪明",
        "说明说明", "要命要命", "救火救火",
        "你好", "打开灯", "你好百度",
    ]
    await generate_batch(test_neg_phrases, test_voices, os.path.join(base_dir, "test"), "tneg")

    # manifest
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
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\npos: {len([f for f in os.listdir(os.path.join(base_dir,'positive')) if f.endswith('.wav')])}")
    print(f"neg: {len([f for f in os.listdir(os.path.join(base_dir,'negative')) if f.endswith('.wav')])}")
    print(f"test: {len(manifest)}")


if __name__ == "__main__":
    asyncio.run(main())

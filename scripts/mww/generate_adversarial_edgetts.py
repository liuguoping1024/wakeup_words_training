#!/usr/bin/env python3
"""
用 edge-tts 生成对抗性负样本（在宿主机运行，不需要 GPU）。
重点：含 -ming/-min 韵母的词。
"""
import asyncio, os, subprocess, sys, random

# 含 -ming/-min 韵母（最高优先级）
MING_WORDS = [
    "说明", "证明", "文明", "光明", "聪明", "黎明", "透明",
    "生命", "革命", "拼命", "要命", "玩命", "小命", "性命", "使命", "宿命",
    "姓名", "有名", "出名", "著名", "知名", "报名", "签名",
    "人民", "市民", "居民", "农民", "移民", "难民",
    "明天", "明白", "明显", "明星", "发明", "说明书",
    "命令", "命运", "生命力", "革命家",
    "民主", "民族", "民间", "民生",
]

# 含 jiu- 声母
JIU_WORDS = [
    "救火", "救人", "救护车", "救援", "救助", "救灾",
    "九月", "九点", "九个", "九十", "九百",
    "就是", "就好", "就行", "就这样",
    "酒店", "酒吧", "喝酒", "白酒", "红酒",
    "旧的", "旧书", "依旧",
    "久等", "长久", "永久", "持久",
]

# 中文声音
VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural",
    "zh-CN-XiaomengNeural",
    "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural",
    "zh-CN-XiaoruiNeural",
    "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural",
    "zh-CN-XiaoyanNeural",
    "zh-CN-XiaoyouNeural",
    "zh-CN-YunfengNeural",
    "zh-CN-YunhaoNeural",
    "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]


async def generate_one(text, voice, output_path):
    """生成一个 edge-tts 音频"""
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
    except Exception as e:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return False


async def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/adversarial_edgetts"
    os.makedirs(output_dir, exist_ok=True)

    # 生成所有组合
    tasks = []
    for word_list, prefix in [(MING_WORDS, "ming"), (JIU_WORDS, "jiu")]:
        for text in word_list:
            for voice in VOICES:
                safe_text = text.replace(" ", "_")
                voice_short = voice.split("-")[-1].replace("Neural", "")
                fname = f"{prefix}_{safe_text}_{voice_short}.wav"
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
                    continue
                tasks.append((text, voice, fpath))

    random.shuffle(tasks)
    total = len(tasks)
    print(f"需要生成 {total} 个音频")

    count = 0
    fails = 0
    for text, voice, fpath in tasks:
        ok = await generate_one(text, voice, fpath)
        if ok:
            count += 1
        else:
            fails += 1
        if (count + fails) % 50 == 0:
            print(f"  进度: {count}/{total} 成功, {fails} 失败")

    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    print(f"\n完成: 新增 {count}, 失败 {fails}, 目录总计 {existing} 个 WAV")


if __name__ == "__main__":
    asyncio.run(main())

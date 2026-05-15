#!/usr/bin/env python3
"""
用 CosyVoice2 生成对抗性负样本：
- 含 "-ming/-min" 韵母的词（模型最容易误触发的）
- 含 "jiu-" 声母的词
- 其他易混淆词

多 GPU 并行：每个 GPU 跑一个进程，处理不同的说话人分片。
"""
import argparse, logging, os, random, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# ── 对抗性负样本词表 ──
# 第一优先级：含 -ming/-min 韵母（误触发最严重）
MING_WORDS = [
    "说明", "证明", "文明", "光明", "聪明", "黎明", "透明",
    "生命", "革命", "拼命", "要命", "玩命", "小命", "性命", "使命", "宿命",
    "姓名", "有名", "出名", "著名", "知名", "报名", "签名",
    "人民", "市民", "居民", "农民", "移民", "难民",
    "明天", "明白", "明显", "明星", "发明", "说明书",
    "命令", "命运", "生命力", "革命家",
    "民主", "民族", "民间", "民生",
]

# 第二优先级：含 "jiu-" 声母
JIU_WORDS = [
    "救火", "救人", "救护车", "救援", "救助", "救灾",
    "九月", "九点", "九个", "九十", "九百",
    "就是", "就好", "就行", "就这样",
    "酒店", "酒吧", "喝酒", "白酒", "红酒",
    "旧的", "旧书", "依旧",
    "久等", "长久", "永久", "持久",
]

# 第三优先级：日常中文（补充泛化）
DAILY_WORDS = [
    "你好", "谢谢", "再见", "对不起", "没关系",
    "打开灯", "关闭窗帘", "播放音乐", "设个闹钟",
    "今天天气怎么样", "几点了", "帮帮我",
    "你好树实", "小爱同学", "天猫精灵", "你好小度",
    "你好百度", "你好谷歌", "嘿Siri",
    "着火了", "危险", "注意安全", "快跑",
    "早上好", "晚上好", "吃饭了吗",
    "一二三四五", "六七八九十",
]

ALL_ADVERSARIAL = MING_WORDS + JIU_WORDS + DAILY_WORDS


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix, ref_wavs):
    """生成 TTS 音频"""
    import torch, torchaudio
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  已有 {existing}/{n_target}，跳过")
        return existing

    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)
    count, fails = existing, 0

    for i in range(existing, n_target):
        text = random.choice(texts)
        ref = str(random.choice(ref_wavs))
        wav_path = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")
        if os.path.exists(wav_path):
            count += 1
            continue
        try:
            for _, r in enumerate(cosyvoice.inference_cross_lingual(
                text, ref, stream=False
            )):
                torchaudio.save(wav_path, resampler(r['tts_speech']), 16000)
                count += 1
                break
        except Exception as e:
            fails += 1
            if fails <= 5:
                log.warning(f"  失败 {i}: {e}")
        if count % 100 == 0 and count > existing:
            log.info(f"  进度: {count}/{n_target} (失败: {fails})")

    log.info(f"  完成: {count}/{n_target} (失败: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refs-dir", default="/workspace/data/speaker_refs")
    parser.add_argument("--model-dir",
                        default="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--n-ming", type=int, default=3000,
                        help="含-ming韵母的对抗性负样本数")
    parser.add_argument("--n-jiu", type=int, default=1500,
                        help="含jiu-声母的对抗性负样本数")
    parser.add_argument("--n-daily", type=int, default=1500,
                        help="日常中文负样本数")
    parser.add_argument("--speakers-start", type=int, default=0)
    parser.add_argument("--speakers-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    from cosyvoice.cli.cosyvoice import AutoModel
    log.info("加载 CosyVoice2...")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    refs = sorted(Path(args.refs_dir).glob("*.wav"))
    end = args.speakers_end if args.speakers_end else len(refs)
    my_refs = refs[args.speakers_start:end]
    log.info(f"说话人 [{args.speakers_start}:{end}] 共 {len(my_refs)} 个")

    # 按说话人数量等比分配样本数
    ratio = len(my_refs) / len(refs)
    n_ming = max(100, int(args.n_ming * ratio))
    n_jiu = max(100, int(args.n_jiu * ratio))
    n_daily = max(100, int(args.n_daily * ratio))

    out = args.output_dir

    log.info(f"=== 含-ming韵母 {n_ming} 条 ===")
    generate_clips(cosyvoice, MING_WORDS,
                   os.path.join(out, "ming"), n_ming, "ming", my_refs)

    log.info(f"=== 含jiu-声母 {n_jiu} 条 ===")
    generate_clips(cosyvoice, JIU_WORDS,
                   os.path.join(out, "jiu"), n_jiu, "jiu", my_refs)

    log.info(f"=== 日常中文 {n_daily} 条 ===")
    generate_clips(cosyvoice, DAILY_WORDS,
                   os.path.join(out, "daily"), n_daily, "daily", my_refs)

    log.info("全部完成!")


if __name__ == "__main__":
    main()

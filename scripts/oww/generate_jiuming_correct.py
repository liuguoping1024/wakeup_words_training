#!/usr/bin/env python3
"""
按正确分类生成"救命"的正样本和负样本。

正样本（应触发）：救命及发音相近变体（方言宽容）
负样本（不应触发）：只含"救"或"命"的词 + 日常中文 + 其他唤醒词
"""
import argparse, logging, os, random, sys
from pathlib import Path
import torch, torchaudio

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# 正样本：救命及方言/口音变体（都应该触发）
POSITIVE_PHRASES = [
    "救命", "救命啊", "快救命", "救命救命",
    "救民", "久名", "揪命", "纠命", "究命",
    "九命", "酒命", "救命呀", "来人救命",
]

# 负样本：只含部分音节的词 + 日常中文
NEGATIVE_PHRASES = [
    # 只含"救"或"命"
    "救火", "救护车", "救人", "救援",
    "生命", "革命", "拼命", "要命", "玩命", "小命",
    "光明", "聪明", "文明", "说明", "证明",
    # 日常中文
    "你好", "谢谢", "再见", "对不起", "没关系",
    "今天天气怎么样", "打开灯", "关闭窗帘",
    "播放音乐", "设个闹钟", "几点了",
    "一二三四五", "六七八九十",
    "你好树实", "小爱同学", "天猫精灵", "你好小度",
    "吃饭了吗", "去哪里", "什么时候",
    "帮帮我", "快来", "注意安全", "危险", "着火了",
    "早上好", "晚上好", "你好吗", "好的",
    "我要去上班", "下班了", "回家吃饭",
]


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix, ref_wavs, is_single=False):
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  已有 {existing}/{n_target}，跳过")
        return
    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)
    count, fails = existing, 0
    for i in range(existing, n_target):
        text = texts if is_single else random.choice(texts)
        ref = str(random.choice(ref_wavs))
        wav_path = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")
        if os.path.exists(wav_path):
            count += 1
            continue
        try:
            for _, r in enumerate(cosyvoice.inference_cross_lingual(text, ref, stream=False)):
                torchaudio.save(wav_path, resampler(r['tts_speech']), 16000)
                count += 1
                break
        except Exception as e:
            fails += 1
            if fails <= 3: log.warning(f"  失败 {i}: {e}")
        if count % 200 == 0 and count > existing:
            log.info(f"  进度: {count}/{n_target} (失败: {fails})")
    log.info(f"  完成: {count}/{n_target} (失败: {fails})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refs-dir", default="/workspace/data/speaker_refs")
    parser.add_argument("--model-dir", default="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--n-pos", type=int, default=5000)
    parser.add_argument("--n-neg", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    # 分片参数（多 GPU 并行用）
    parser.add_argument("--task", choices=["pos", "neg", "both"], default="both")
    args = parser.parse_args()

    random.seed(args.seed)
    from cosyvoice.cli.cosyvoice import AutoModel
    log.info("加载 CosyVoice2...")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    refs = sorted(Path(args.refs_dir).glob("*.wav"))
    log.info(f"参考音频: {len(refs)} 个说话人")

    if args.task in ("pos", "both"):
        log.info(f"=== 正样本（救命变体）{args.n_pos} 条 ===")
        generate_clips(cosyvoice, POSITIVE_PHRASES,
                        os.path.join(args.output_dir, "positive"),
                        args.n_pos, "pos", refs)

    if args.task in ("neg", "both"):
        log.info(f"=== 负样本（无关中文）{args.n_neg} 条 ===")
        generate_clips(cosyvoice, NEGATIVE_PHRASES,
                        os.path.join(args.output_dir, "negative"),
                        args.n_neg, "neg", refs)

    log.info("完成!")


if __name__ == "__main__":
    main()

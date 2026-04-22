#!/usr/bin/env python3
"""用 CosyVoice2 生成"救命"正样本和对抗性负样本。"""
import argparse, logging, os, random, sys
from pathlib import Path
import numpy as np, torch, torchaudio

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

ADVERSARIAL_PHRASES = [
    "就命", "旧命", "九明", "酒名", "久鸣",
    "纠明", "揪命", "究命", "九命", "酒命",
    "救", "命", "救人", "命令",
    "救火", "救护车", "生命", "革命",
    "小命", "拼命", "要命", "玩命",
    "打开灯", "关闭窗帘", "今天天气",
    "小爱同学", "天猫精灵", "你好小度",
    "谢谢", "再见", "对不起", "早上好",
    "帮帮我", "快来", "危险", "着火了",
]

def generate_clips(cosyvoice, texts, output_dir, n_target, prefix, ref_wav, is_single_text=False):
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  已有 {existing}/{n_target}，跳过")
        return
    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)
    count, fails = existing, 0
    for i in range(existing, n_target):
        text = texts if is_single_text else random.choice(texts)
        wav_path = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")
        if os.path.exists(wav_path):
            count += 1
            continue
        try:
            for _, result in enumerate(cosyvoice.inference_cross_lingual(text, ref_wav, stream=False)):
                torchaudio.save(wav_path, resampler(result['tts_speech']), 16000)
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
    parser.add_argument("--output-dir", default="/workspace/outputs/cosyvoice_clips_jiuming")
    parser.add_argument("--model-dir", default="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--ref-wav", default="/workspace/CosyVoice/asset/zero_shot_prompt.wav")
    parser.add_argument("--n-pos", type=int, default=5000)
    parser.add_argument("--n-neg-train", type=int, default=5000)
    parser.add_argument("--n-neg-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    from cosyvoice.cli.cosyvoice import AutoModel
    log.info("加载 CosyVoice2...")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    log.info("=== 生成正样本 (救命) ===")
    generate_clips(cosyvoice, "救命", os.path.join(args.output_dir, "positive"),
                    args.n_pos, "pos", args.ref_wav, is_single_text=True)

    log.info("=== 生成负样本 (train) ===")
    generate_clips(cosyvoice, ADVERSARIAL_PHRASES, os.path.join(args.output_dir, "negative_train"),
                    args.n_neg_train, "neg", args.ref_wav)

    log.info("=== 生成负样本 (test) ===")
    generate_clips(cosyvoice, ADVERSARIAL_PHRASES, os.path.join(args.output_dir, "negative_test"),
                    args.n_neg_test, "neg", args.ref_wav)

    log.info("全部完成!")

if __name__ == "__main__":
    main()

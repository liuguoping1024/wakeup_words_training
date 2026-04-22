#!/usr/bin/env python3
"""
用 CosyVoice2 生成中文正样本和对抗性负样本。
在 Docker 内运行（cosyvoice 镜像），完全离线，GPU 推理。

CosyVoice2 支持 zero-shot voice cloning，用不同参考音频生成不同声音。
也支持 cross_lingual 模式直接生成中文语音。

用法（Docker 内）：
  python3 generate_cosyvoice_clips.py \
    --keyword "你好树实" \
    --output-dir /workspace/outputs/cosyvoice_clips \
    --n-pos 10000 --n-neg 10000
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

ADVERSARIAL_PHRASES = [
    "你好叔叔", "你好书式", "你好树枝", "你好属实",
    "你好舒适", "你好数十", "你好书市", "你好熟食",
    "你好竖石", "你好疏食", "你好殊实",
    "你好", "树实", "你好啊", "你好吗",
    "你好世界", "你好师傅", "你好同学", "你好朋友",
    "泥好树实", "拟好树实", "你浩树实", "你号树实",
    "小爱同学", "天猫精灵", "你好小度",
    "你好吃吗", "你好看吗", "你好厉害", "你好棒",
    "树上有鸟", "实在太好", "书本在哪", "数学题",
    "打开灯", "关闭窗帘", "今天天气", "播放音乐",
    "谢谢", "再见", "对不起", "没关系", "早上好",
]


def generate_clips(cosyvoice, text_list, output_dir, n_target, prefix,
                    prompt_wav, prompt_text, is_single_text=False):
    """用 CosyVoice2 生成语音片段。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  已有 {existing}/{n_target}，跳过")
        return existing

    # 16kHz resampler
    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)

    count = existing
    for i in range(existing, n_target):
        text = text_list if is_single_text else random.choice(text_list)
        wav_path = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")

        if os.path.exists(wav_path):
            count += 1
            continue

        try:
            # 用 zero-shot 模式生成（克隆参考音频的声音）
            for _, result in enumerate(cosyvoice.inference_zero_shot(
                text, prompt_text, prompt_wav, stream=False
            )):
                audio = result['tts_speech']
                # 重采样到 16kHz
                audio_16k = resampler(audio)
                torchaudio.save(wav_path, audio_16k, 16000)
                count += 1
                break  # 只取第一个结果

        except Exception as e:
            log.warning(f"  生成失败 {i}: {e}")

        if count % 200 == 0 and count > existing:
            log.info(f"  进度: {count}/{n_target}")

    log.info(f"  完成: {count}/{n_target}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="你好树实")
    parser.add_argument("--model-dir", default="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--output-dir", default="/workspace/outputs/cosyvoice_clips")
    parser.add_argument("--prompt-wav", default="/workspace/CosyVoice/asset/zero_shot_prompt.wav",
                        help="参考音频（用于 voice cloning）")
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。",
                        help="参考音频对应的文字")
    parser.add_argument("--n-pos", type=int, default=10000)
    parser.add_argument("--n-neg-train", type=int, default=10000)
    parser.add_argument("--n-neg-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, '/workspace/CosyVoice')
    sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')

    from cosyvoice.cli.cosyvoice import AutoModel

    log.info(f"加载 CosyVoice2 模型: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    log.info("模型加载完成")

    # 正样本
    log.info(f"=== 生成正样本 ({args.keyword}) ===")
    generate_clips(
        cosyvoice, args.keyword,
        os.path.join(args.output_dir, "positive"),
        args.n_pos, "pos",
        args.prompt_wav, args.prompt_text,
        is_single_text=True
    )

    # 负样本 train
    log.info("=== 生成负样本 (train) ===")
    generate_clips(
        cosyvoice, ADVERSARIAL_PHRASES,
        os.path.join(args.output_dir, "negative_train"),
        args.n_neg_train, "neg",
        args.prompt_wav, args.prompt_text
    )

    # 负样本 test
    log.info("=== 生成负样本 (test) ===")
    generate_clips(
        cosyvoice, ADVERSARIAL_PHRASES,
        os.path.join(args.output_dir, "negative_test"),
        args.n_neg_test, "neg",
        args.prompt_wav, args.prompt_text
    )

    log.info("全部完成!")


if __name__ == "__main__":
    main()

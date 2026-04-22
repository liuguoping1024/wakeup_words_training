#!/usr/bin/env python3
"""
用 CosyVoice2 批量生成中文正样本和对抗性负样本。
Docker 内运行，完全离线 GPU 推理。

策略：用 cross_lingual 模式 + 不同参考音频生成多种声音。

用法（Docker 内）：
  python3 generate_cosyvoice_batch.py \
    --keyword "你好树实" \
    --output-dir /workspace/outputs/cosyvoice_clips \
    --ref-dir /workspace/data/positive_raw/nihao_shushi \
    --n-pos 5000 --n-neg 5000
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
    "你好", "树实", "你好啊", "你好吗",
    "你好世界", "你好师傅", "你好同学", "你好朋友",
    "泥好树实", "拟好树实", "你浩树实", "你号树实",
    "小爱同学", "天猫精灵", "你好小度",
    "你好吃吗", "你好看吗", "你好厉害",
    "树上有鸟", "实在太好", "书本在哪",
    "打开灯", "关闭窗帘", "今天天气", "播放音乐",
    "谢谢", "再见", "对不起", "早上好",
]


def collect_ref_audios(ref_dir, n_refs=20, min_dur=1.0, seed=42):
    """从真实录音中选取不同说话人的参考音频。"""
    import wave
    wavs = sorted(Path(ref_dir).glob("*.wav"))
    rng = random.Random(seed)
    rng.shuffle(wavs)

    # 按说话人前缀分组（文件名格式: 001weishu_Android_00000.wav）
    speakers = {}
    for w in wavs:
        prefix = w.name.split("_")[0]  # e.g. "001weishu"
        if prefix not in speakers:
            speakers[prefix] = w

    refs = list(speakers.values())[:n_refs]
    log.info(f"选取 {len(refs)} 个不同说话人的参考音频")
    return refs


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix,
                    ref_audios, is_single_text=False):
    """批量生成语音。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  已有 {existing}/{n_target}，跳过")
        return existing

    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)
    count = existing
    fails = 0

    for i in range(existing, n_target):
        text = texts if is_single_text else random.choice(texts)
        ref_wav = str(random.choice(ref_audios))
        wav_path = os.path.join(output_dir, f"{prefix}_{i:06d}.wav")

        if os.path.exists(wav_path):
            count += 1
            continue

        try:
            for _, result in enumerate(cosyvoice.inference_cross_lingual(
                text, ref_wav, stream=False
            )):
                audio = result['tts_speech']
                audio_16k = resampler(audio)
                torchaudio.save(wav_path, audio_16k, 16000)
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
    parser.add_argument("--keyword", default="你好树实")
    parser.add_argument("--model-dir", default="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--output-dir", default="/workspace/outputs/cosyvoice_clips")
    parser.add_argument("--ref-dir", default="/workspace/data/positive_raw/nihao_shushi",
                        help="参考音频目录（用于 voice cloning）")
    parser.add_argument("--default-ref", default="/workspace/CosyVoice/asset/zero_shot_prompt.wav")
    parser.add_argument("--n-pos", type=int, default=5000)
    parser.add_argument("--n-neg-train", type=int, default=5000)
    parser.add_argument("--n-neg-test", type=int, default=1000)
    parser.add_argument("--n-refs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from cosyvoice.cli.cosyvoice import AutoModel

    log.info(f"加载 CosyVoice2: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    log.info(f"模型加载完成, sample_rate={cosyvoice.sample_rate}")

    # 收集参考音频
    if os.path.exists(args.ref_dir):
        ref_audios = collect_ref_audios(args.ref_dir, n_refs=args.n_refs, seed=args.seed)
    else:
        ref_audios = [args.default_ref]
        log.warning(f"参考音频目录不存在，使用默认: {args.default_ref}")

    # 正样本
    log.info(f"=== 生成正样本 ({args.keyword}) ===")
    generate_clips(cosyvoice, args.keyword,
                    os.path.join(args.output_dir, "positive"),
                    args.n_pos, "pos", ref_audios, is_single_text=True)

    # 负样本 train
    log.info("=== 生成负样本 (train) ===")
    generate_clips(cosyvoice, ADVERSARIAL_PHRASES,
                    os.path.join(args.output_dir, "negative_train"),
                    args.n_neg_train, "neg", ref_audios)

    # 负样本 test
    log.info("=== 生成负样本 (test) ===")
    generate_clips(cosyvoice, ADVERSARIAL_PHRASES,
                    os.path.join(args.output_dir, "negative_test"),
                    args.n_neg_test, "neg", ref_audios)

    log.info("全部完成!")


if __name__ == "__main__":
    main()

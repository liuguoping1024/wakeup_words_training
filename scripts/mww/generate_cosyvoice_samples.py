#!/usr/bin/env python3
"""
用 CosyVoice2 生成中文正样本和对抗性负样本。
在 Docker 内运行（cosyvoice 镜像），GPU 推理，完全离线。

多参考音频轮换，每个参考音频生成不同声音，最大化多样性。

用法（Docker 内）：
  python3 generate_cosyvoice_samples.py \
    --keyword "救命" --keyword-id jiuming \
    --pos-dir /workspace/data/tts_positive/jiuming \
    --neg-dir /workspace/data/tts_negative/jiuming \
    --n-pos 1000 --n-neg 1000 \
    --ref-dir /workspace/data/real_voices_jiuming
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

ADVERSARIAL = {
    "jiuming": [
        "旧命", "就命", "酒名", "九命", "久明", "纠明",
        "救民", "救星", "救人", "救火", "救护", "救急",
        "救", "命", "救了", "命令", "生命", "性命", "拼命",
        "揪命", "究命", "灸命",
        "帮帮我", "快来", "怎么办", "着火了",
        "报警", "快跑", "小心", "危险", "停下",
        "你好", "谢谢", "再见", "早上好",
    ],
    "nihao_shushi": [
        "你好叔叔", "你好书式", "你好树枝", "你好属实",
        "你好舒适", "你好数十", "你好书市", "你好熟食",
        "你好", "树实", "你好啊", "你好吗",
        "你好世界", "你好师傅", "你好同学", "你好朋友",
    ],
}

# 默认参考音频和文本（CosyVoice2 自带的）
DEFAULT_REFS = [
    {
        "wav": "/workspace/work/CosyVoice/asset/zero_shot_prompt.wav",
        "text": "希望你以后能够做的比我还好呦。",
    },
]


def collect_ref_audios(ref_dir, max_refs=10):
    """从真实录音目录中挑选参考音频，用于 voice cloning 多样性。"""
    refs = list(DEFAULT_REFS)
    if ref_dir and os.path.isdir(ref_dir):
        wavs = sorted(Path(ref_dir).glob("*.wav"))
        # 挑选间隔均匀的样本作为参考
        step = max(1, len(wavs) // max_refs)
        selected = wavs[::step][:max_refs]
        for w in selected:
            refs.append({
                "wav": str(w),
                # CosyVoice2 zero-shot 需要参考文本，用唤醒词本身
                "text": "救命",
            })
    log.info(f"参考音频数量: {len(refs)}")
    return refs


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix,
                   refs, is_single=False):
    """用 CosyVoice2 生成语音片段，轮换参考音频。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir)
                    if f.endswith(".wav") and f.startswith(prefix)])
    if existing >= n_target * 0.95:
        log.info(f"  {prefix}: 已有 {existing}/{n_target}，跳过")
        return existing

    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)
    count = existing
    fails = 0

    for i in range(existing, n_target):
        text = texts if is_single else random.choice(texts)
        ref = refs[i % len(refs)]
        wav_path = os.path.join(output_dir, f"{prefix}_{i:05d}.wav")
        if os.path.exists(wav_path):
            count += 1
            continue
        try:
            for _, result in enumerate(cosyvoice.inference_zero_shot(
                text, ref["text"], ref["wav"], stream=False
            )):
                audio = result['tts_speech']
                audio_16k = resampler(audio)
                torchaudio.save(wav_path, audio_16k, 16000)
                count += 1
                break
        except Exception as e:
            fails += 1
            if fails <= 5:
                log.warning(f"  生成失败 {i}: {e}")
        if count % 100 == 0 and count > existing:
            log.info(f"  {prefix} 进度: {count}/{n_target} (失败: {fails})")

    log.info(f"  {prefix}: 完成 {count} 条 (失败: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--keyword-id", required=True)
    parser.add_argument("--pos-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--n-pos", type=int, default=1000)
    parser.add_argument("--n-neg", type=int, default=1000)
    parser.add_argument("--ref-dir", default=None,
                        help="真实录音目录，用于多参考音频 voice cloning")
    parser.add_argument("--model-dir",
                        default="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, '/workspace/work/CosyVoice')
    sys.path.insert(0, '/workspace/work/CosyVoice/third_party/Matcha-TTS')
    from cosyvoice.cli.cosyvoice import AutoModel

    log.info(f"加载 CosyVoice2: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    log.info(f"模型加载完成, sample_rate={cosyvoice.sample_rate}")

    # 更新默认参考文本为当前唤醒词
    for r in DEFAULT_REFS:
        pass  # 保持默认参考文本不变，它是通用中文

    refs = collect_ref_audios(args.ref_dir)
    # 更新从真实录音来的参考文本
    for r in refs:
        if r not in DEFAULT_REFS:
            r["text"] = args.keyword

    neg_phrases = ADVERSARIAL.get(args.keyword_id, ADVERSARIAL.get("jiuming"))

    log.info(f"=== CosyVoice2 正样本 ({args.keyword}) ===")
    generate_clips(cosyvoice, args.keyword, args.pos_dir, args.n_pos,
                   "cosy_pos", refs, is_single=True)

    log.info(f"=== CosyVoice2 负样本 ({len(neg_phrases)} 短语) ===")
    generate_clips(cosyvoice, neg_phrases, args.neg_dir, args.n_neg,
                   "cosy_neg", refs)

    pos_count = len([f for f in os.listdir(args.pos_dir) if f.endswith(".wav")])
    neg_count = len([f for f in os.listdir(args.neg_dir) if f.endswith(".wav")])
    log.info(f"=== CosyVoice2 完成: 正样本 {pos_count}, 负样本 {neg_count} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
用 CosyVoice2 生成 help help 多说话人正样本 + 对抗性负样本。
在 cosyvoice Docker 内运行，支持 GPU 分片。
"""
import argparse, logging, os, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

POSITIVE_PHRASES = [
    "help help",
    "help help!",
    "help! help!",
    "help help please",
    "help help me",
    "somebody help help",
    "help help help",
]

NEGATIVE_PHRASES = [
    "help me", "help us", "help him", "help her",
    "can you help", "please help", "I need help",
    "help yourself", "helpful", "helpless",
    "health", "held", "yelp", "kelp", "self",
    "hello", "helicopter",
    "hip hop", "tick tock", "flip flop",
    "good morning", "thank you", "excuse me",
    "turn on the light", "play music",
    "hey siri", "ok google", "alexa",
    "one two three four five",
]


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix, ref_wavs):
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
            log.info(f"  [{prefix}] {count}/{n_target} (失败: {fails})")

    log.info(f"  [{prefix}] 完成: {count}/{n_target} (失败: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refs-dir", default="/workspace/data/speaker_refs")
    parser.add_argument("--model-dir",
                        default="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--n-pos", type=int, default=2000)
    parser.add_argument("--n-neg", type=int, default=1500)
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

    ratio = len(my_refs) / max(len(refs), 1)
    n_pos = max(100, int(args.n_pos * ratio))
    n_neg = max(100, int(args.n_neg * ratio))

    log.info(f"=== 正样本 {n_pos} 条 ===")
    generate_clips(cosyvoice, POSITIVE_PHRASES,
                   os.path.join(args.output_dir, "positive"),
                   n_pos, "pos", my_refs)

    log.info(f"=== 负样本 {n_neg} 条 ===")
    generate_clips(cosyvoice, NEGATIVE_PHRASES,
                   os.path.join(args.output_dir, "negative"),
                   n_neg, "neg", my_refs)

    log.info("完成!")


if __name__ == "__main__":
    main()

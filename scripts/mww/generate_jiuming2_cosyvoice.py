#!/usr/bin/env python3
"""
救命救命 CosyVoice 多说话人生成（v1/v2 通用）。
"""
import argparse, logging, os, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

POSITIVE_PHRASES = [
    "救命救命", "救命救命啊", "救命救命呀", "快救命救命",
    "来人救命救命", "救命救命救命",
    "久名久名", "揪命揪命", "纠命纠命", "九命九命", "究命究命",
    "救民救民", "久民久民", "揪民揪民",
    "救民救命", "救命救民", "久名救命", "救命久名",
    "九命救命", "救命九命", "纠命救命", "救命纠命",
    "酒名酒名", "酒命酒命",
]

NEGATIVE_PHRASES = [
    # 单次救命
    "救命", "救命啊", "快救命", "来人救命",
    "久名", "救民", "揪命", "九命",
    # 含-ming重复
    "说明说明", "文明文明", "聪明聪明", "革命革命",
    "光明光明", "生命生命", "要命要命", "拼命拼命",
    # 结构相似
    "救火救火", "救人救人", "玩命玩命",
    # 日常
    "你好", "谢谢", "再见", "打开灯", "播放音乐",
    "你好树实", "小爱同学", "天猫精灵",
    "今天天气怎么样", "早上好", "晚上好",
]


def generate_clips(cosyvoice, texts, output_dir, n_target, prefix, ref_wavs):
    import torch, torchaudio
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if existing >= n_target * 0.95:
        log.info(f"  have {existing}/{n_target}, skip")
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
            for _, r in enumerate(cosyvoice.inference_cross_lingual(text, ref, stream=False)):
                torchaudio.save(wav_path, resampler(r['tts_speech']), 16000)
                count += 1
                break
        except Exception as e:
            fails += 1
            if fails <= 5:
                log.warning(f"  fail {i}: {e}")
        if count % 100 == 0 and count > existing:
            log.info(f"  [{prefix}] {count}/{n_target} (fail: {fails})")
    log.info(f"  [{prefix}] done: {count}/{n_target} (fail: {fails})")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refs-dir", default="/workspace/data/speaker_refs")
    parser.add_argument("--model-dir",
                        default="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--n-pos", type=int, default=1250)
    parser.add_argument("--n-neg", type=int, default=1250)
    parser.add_argument("--speakers-start", type=int, default=0)
    parser.add_argument("--speakers-end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    from cosyvoice.cli.cosyvoice import AutoModel
    log.info("Loading CosyVoice2...")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    refs = sorted(Path(args.refs_dir).glob("*.wav"))
    end = args.speakers_end if args.speakers_end else len(refs)
    my_refs = refs[args.speakers_start:end]
    log.info(f"Speakers [{args.speakers_start}:{end}] = {len(my_refs)}")

    ratio = len(my_refs) / max(len(refs), 1)
    n_pos = max(100, int(args.n_pos * ratio))
    n_neg = max(100, int(args.n_neg * ratio))

    log.info(f"=== Positive {n_pos} ===")
    generate_clips(cosyvoice, POSITIVE_PHRASES,
                   os.path.join(args.output_dir, "positive"), n_pos, "pos", my_refs)

    log.info(f"=== Negative {n_neg} ===")
    generate_clips(cosyvoice, NEGATIVE_PHRASES,
                   os.path.join(args.output_dir, "negative"), n_neg, "neg", my_refs)

    log.info("Done!")


if __name__ == "__main__":
    main()

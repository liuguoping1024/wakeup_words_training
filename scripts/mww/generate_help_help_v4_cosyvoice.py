#!/usr/bin/env python3
"""
help help v4 CosyVoice — 严格正样本 + 扩展负样本。
按文件名区分 phrase（不用 random.choice）。
"""
import argparse, logging, os, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

POSITIVE_PHRASES = [
    "help help",
    "help help!",
    "help! help!",
]

NEGATIVE_PHRASES = [
    "help me", "help him", "help her", "help us", "help them",
    "help out", "help up",
    "helpful", "helpless", "helper",
    "helm", "helmet",
    "hello", "hell", "yelp", "kelp", "whelp",
    "shelf", "shell", "smell", "spell", "swell",
    "fell", "tell", "sell", "well", "yell",
    "smelly", "belly", "jelly",
    "held", "melt", "felt", "belt",
    "self", "myself", "yourself",
    "hip hop", "tick tock", "flip flop", "drip drop",
    "knock knock", "ping pong", "chit chat",
    "bye bye", "so so",
    "good morning", "good night", "thank you",
    "come on", "look out", "stand up", "sit down",
    "turn on", "turn off", "hey siri",
    "call mom", "fire fire",
    "one two", "three four", "yes no",
]


def generate_clips_per_phrase(cosyvoice, phrase, phrase_idx, output_dir,
                               n_per_phrase, prefix, ref_wavs):
    """每个 phrase 分别生成，文件名带 phrase index"""
    import torch, torchaudio
    os.makedirs(output_dir, exist_ok=True)
    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)

    existing = len([f for f in os.listdir(output_dir)
                    if f.startswith(f"{prefix}_p{phrase_idx:02d}_") and f.endswith(".wav")])
    if existing >= n_per_phrase * 0.95:
        log.info(f"  [{prefix} p{phrase_idx}] '{phrase}' have {existing}/{n_per_phrase}, skip")
        return existing

    count, fails = existing, 0
    for i in range(existing, n_per_phrase):
        ref = str(random.choice(ref_wavs))
        wav_path = os.path.join(output_dir, f"{prefix}_p{phrase_idx:02d}_{i:06d}.wav")
        if os.path.exists(wav_path):
            count += 1
            continue
        try:
            for _, r in enumerate(cosyvoice.inference_cross_lingual(phrase, ref, stream=False)):
                torchaudio.save(wav_path, resampler(r['tts_speech']), 16000)
                count += 1
                break
        except Exception as e:
            fails += 1
            if fails <= 5:
                log.warning(f"  fail {i}: {e}")
        if count % 100 == 0 and count > existing:
            log.info(f"  [{prefix} p{phrase_idx}] '{phrase}' {count}/{n_per_phrase}")
    log.info(f"  [{prefix} p{phrase_idx}] '{phrase}' done: {count}/{n_per_phrase}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refs-dir", default="/workspace/data/speaker_refs")
    parser.add_argument("--model-dir",
                        default="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--pos-per-phrase", type=int, default=800,
                        help="每个正样本 phrase 生成多少条")
    parser.add_argument("--neg-per-phrase", type=int, default=40,
                        help="每个负样本 phrase 生成多少条")
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
    n_pos_per = max(50, int(args.pos_per_phrase * ratio))
    n_neg_per = max(10, int(args.neg_per_phrase * ratio))

    log.info(f"=== 正样本（{len(POSITIVE_PHRASES)} 词 × {n_pos_per}）===")
    for idx, phrase in enumerate(POSITIVE_PHRASES):
        generate_clips_per_phrase(cosyvoice, phrase, idx,
                                   os.path.join(args.output_dir, "positive"),
                                   n_pos_per, "pos", my_refs)

    log.info(f"=== 负样本（{len(NEGATIVE_PHRASES)} 词 × {n_neg_per}）===")
    for idx, phrase in enumerate(NEGATIVE_PHRASES):
        generate_clips_per_phrase(cosyvoice, phrase, idx,
                                   os.path.join(args.output_dir, "negative"),
                                   n_neg_per, "neg", my_refs)

    log.info("Done!")


if __name__ == "__main__":
    main()

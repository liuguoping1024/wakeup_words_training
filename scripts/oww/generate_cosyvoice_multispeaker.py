#!/usr/bin/env python3
"""用多个真实说话人参考音频，CosyVoice2 批量生成多说话人的目标文本。

支持从命令行指定 speaker 分片（--speakers-start --speakers-end），
这样多 GPU/多进程并行时每个实例处理不同的说话人。
"""
import argparse, logging, os, random, sys
from pathlib import Path
import torch, torchaudio

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="救命", help="要生成的文本")
    parser.add_argument("--refs-dir", required=True, help="参考音频目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--n-per-speaker", type=int, default=125, help="每个说话人生成数")
    parser.add_argument("--speakers-start", type=int, default=0, help="说话人索引起始")
    parser.add_argument("--speakers-end", type=int, default=None, help="说话人索引结束（不含）")
    parser.add_argument("--model-dir", default="/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    from cosyvoice.cli.cosyvoice import AutoModel
    log.info(f"加载 CosyVoice2: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    # 获取所有说话人
    refs = sorted(Path(args.refs_dir).glob("*.wav"))
    end = args.speakers_end if args.speakers_end else len(refs)
    my_refs = refs[args.speakers_start:end]
    log.info(f"处理说话人 [{args.speakers_start}:{end}] 共 {len(my_refs)} 个")

    os.makedirs(args.output_dir, exist_ok=True)
    resampler = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)

    total_target = len(my_refs) * args.n_per_speaker
    count = 0
    fails = 0

    for ref_idx, ref_path in enumerate(my_refs):
        speaker = ref_path.stem
        for i in range(args.n_per_speaker):
            wav_path = os.path.join(args.output_dir, f"{speaker}_{i:04d}.wav")
            if os.path.exists(wav_path):
                count += 1
                continue

            try:
                for _, result in enumerate(cosyvoice.inference_cross_lingual(
                    args.text, str(ref_path), stream=False
                )):
                    torchaudio.save(wav_path, resampler(result['tts_speech']), 16000)
                    count += 1
                    break
            except Exception as e:
                fails += 1
                if fails <= 3:
                    log.warning(f"失败 {speaker}/{i}: {e}")

            if count % 100 == 0:
                log.info(f"  进度: {count}/{total_target} (失败: {fails})")

    log.info(f"完成: {count}/{total_target} (失败: {fails})")


if __name__ == "__main__":
    main()

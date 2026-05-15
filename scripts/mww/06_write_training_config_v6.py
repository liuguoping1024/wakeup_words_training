#!/usr/bin/env python3
"""
jiuming v6 训练配置。
改进：
- 真实录音 ×10 增加权重
- 对抗性负样本权重降低（v5 太激进）
- 训练步数增加到 25000
"""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-dir", default="trained_models/jiuming_v6")
    args = parser.parse_args()

    config = {
        "window_step_ms": 10,
        "train_dir": args.train_dir,
        "features": [
            # 正样本（真实录音 ×10 + TTS）
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            # 英文语音
            {
                "features_dir": "negative_datasets/speech/speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 英文聚会
            {
                "features_dir": "negative_datasets/dinner_party/dinner_party",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 无语音噪声
            {
                "features_dir": "negative_datasets/no_speech/no_speech",
                "sampling_weight": 3.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # AISHELL-1 中文语音
            {
                "features_dir": "negative_datasets/zh_chinese/zh_chinese",
                "sampling_weight": 10.0,
                "penalty_weight": 1.5,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 对抗性负样本（v6: 降低权重）
            {
                "features_dir": "negative_datasets/zh_adversarial/zh_adversarial",
                "sampling_weight": 8.0,
                "penalty_weight": 1.5,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 评估集
            {
                "features_dir": "negative_datasets/dinner_party_eval/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [args.steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[done] v6 训练配置: {output}")


if __name__ == "__main__":
    main()

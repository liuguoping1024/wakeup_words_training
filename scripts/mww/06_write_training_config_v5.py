#!/usr/bin/env python3
"""
生成 jiuming v5 训练配置 YAML。
改进：加入 zh_adversarial 对抗性负样本（含-ming韵母、jiu-声母）。
"""
import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-dir", default="trained_models/jiuming_v5")
    args = parser.parse_args()

    config = {
        "window_step_ms": 10,
        "train_dir": args.train_dir,
        "features": [
            # 正样本
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            # 英文语音负样本
            {
                "features_dir": "negative_datasets/speech/speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 英文聚会场景
            {
                "features_dir": "negative_datasets/dinner_party/dinner_party",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # 无语音环境噪声
            {
                "features_dir": "negative_datasets/no_speech/no_speech",
                "sampling_weight": 3.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # AISHELL-1 中文语音（通用中文负样本）
            {
                "features_dir": "negative_datasets/zh_chinese/zh_chinese",
                "sampling_weight": 10.0,
                "penalty_weight": 1.5,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            # ★ 对抗性负样本：含-ming韵母、jiu-声母（v5 新增）
            # 高权重 + 高惩罚，强制模型区分"救命"和"X命/X明"
            {
                "features_dir": "negative_datasets/zh_adversarial/zh_adversarial",
                "sampling_weight": 20.0,
                "penalty_weight": 3.0,
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

    print(f"[done] v5 训练配置已写入: {output}")


if __name__ == "__main__":
    main()

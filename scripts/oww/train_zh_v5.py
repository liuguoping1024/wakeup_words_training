#!/usr/bin/env python3
"""
OWW 中文唤醒词训练 v5：基于 v4 特征，优化负样本策略。

改进：
  1. max_negative_weight 100 → 500（加大负样本惩罚）
  2. 加回 ACAV100M 128/batch（通用音频背景）
  3. 正样本 batch 200，对抗性负样本 200，ACAV 128
     → 正样本占比 200/(200+200+128) = 37.9%

复用 v4 的特征文件（不重新准备数据和增强）。
"""
import argparse, logging, os, sys
from pathlib import Path
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/outputs/oww/nihao_shushi_v4")
    parser.add_argument("--output-dir", default="/workspace/outputs/oww")
    parser.add_argument("--acav-features", default="/workspace/data/oww/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    parser.add_argument("--fp-val-data", default="/workspace/data/oww/validation_set_features.npy")
    parser.add_argument("--max-neg-weight", type=int, default=500)
    parser.add_argument("--acav-batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=115000)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/work/openWakeWord")
    from openwakeword.train import Model, mmap_batch_generator

    pos_feat_train = os.path.join(args.model_dir, "positive_features_train.npy")
    pos_feat_test = os.path.join(args.model_dir, "positive_features_test.npy")
    neg_feat_train = os.path.join(args.model_dir, "negative_features_train.npy")
    neg_feat_test = os.path.join(args.model_dir, "negative_features_test.npy")

    for f in [pos_feat_train, pos_feat_test, neg_feat_train, neg_feat_test]:
        if not os.path.exists(f):
            log.error(f"缺少特征文件: {f}")
            sys.exit(1)
        log.info(f"  {Path(f).name}: {np.load(f, mmap_mode='r').shape}")

    input_shape = np.load(pos_feat_test, mmap_mode='r').shape[1:]

    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=64, n_blocks=3,
                seconds_per_example=1280 * input_shape[0] / 16000)

    def reshape_fn(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    feature_data_files = {
        "positive": pos_feat_train,
        "adversarial_negative": neg_feat_train,
    }
    batch_n_per_class = {
        "positive": 200,
        "adversarial_negative": 200,
    }
    data_transforms = {}
    label_transforms = {
        "positive": lambda x: [1 for _ in x],
        "adversarial_negative": lambda x: [0 for _ in x],
    }

    # 加回 ACAV100M 作为通用背景负样本（中等比例）
    if os.path.exists(args.acav_features):
        feature_data_files["ACAV100M"] = args.acav_features
        batch_n_per_class["ACAV100M"] = args.acav_batch
        data_transforms["ACAV100M"] = reshape_fn
        label_transforms["ACAV100M"] = lambda x: [0 for _ in x]
        log.info(f"ACAV100M: {np.load(args.acav_features, mmap_mode='r').shape}")

    total = sum(batch_n_per_class.values())
    log.info(f"batch_n_per_class: {batch_n_per_class}")
    log.info(f"正样本占比: {batch_n_per_class['positive']/total*100:.1f}%")

    batch_gen = mmap_batch_generator(feature_data_files, n_per_class=batch_n_per_class,
                                      data_transform_funcs=data_transforms,
                                      label_transform_funcs=label_transforms)

    class IterDS(torch.utils.data.IterableDataset):
        def __init__(self, g): self.g = g
        def __iter__(self): return self.g

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(IterDS(batch_gen), batch_size=None,
                                           num_workers=n_cpus, prefetch_factor=16)

    fp_val = np.load(args.fp_val_data)
    fp_val = np.array([fp_val[i:i+input_shape[0]] for i in range(0, fp_val.shape[0]-input_shape[0], 1)])
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_val),
                                         torch.from_numpy(np.zeros(fp_val.shape[0]).astype(np.float32))),
        batch_size=len(fp_val))

    X_pos = np.load(pos_feat_test)
    X_neg = np.load(neg_feat_test)
    labels = np.hstack((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0]))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((X_pos, X_neg))), torch.from_numpy(labels)),
        batch_size=len(labels))

    log.info(f"训练: {args.steps} steps, 64x3, penalty={args.max_neg_weight}")
    best_model = oww.auto_train(X_train=X_train, X_val=X_val, false_positive_val_data=X_val_fp,
                                 steps=args.steps, max_negative_weight=args.max_neg_weight,
                                 target_fp_per_hour=0.5)

    oww.export_model(model=best_model, model_name="nihao_shushi_v5", output_dir=args.output_dir)
    # 同时覆盖 nihao_shushi.onnx 作为最新模型
    import shutil
    src = os.path.join(args.output_dir, "nihao_shushi_v5.onnx")
    dst = os.path.join(args.output_dir, "nihao_shushi.onnx")
    if os.path.exists(src):
        shutil.copy2(src, dst)
    log.info(f"模型: {dst}")


if __name__ == "__main__":
    main()

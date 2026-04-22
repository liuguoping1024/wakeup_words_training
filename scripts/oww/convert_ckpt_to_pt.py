#!/usr/bin/env python3
"""
将 Piper .ckpt checkpoint 转换为 .pt 文件，供 generate_samples.py 使用。

参考：https://github.com/rhasspy/piper-sample-generator/issues/4#issuecomment-2428899252

用法（Docker 内）：
  python3 convert_ckpt_to_pt.py \
    --ckpt /workspace/work/piper-sample-generator-oww/models/zh_CN-huayan-medium.ckpt \
    --output /workspace/work/piper-sample-generator-oww/models/zh_CN-huayan-medium.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def convert_ckpt_to_pt(ckpt_path, output_path=None):
    """Convert a Piper .ckpt file to a .pt file containing just model_g."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "work" / "piper-sample-generator-oww"))
    from piper_train.vits.lightning import VitsModel

    log.info(f"Loading checkpoint: {ckpt_path}")
    model = VitsModel.load_from_checkpoint(ckpt_path)

    if output_path is None:
        output_path = str(ckpt_path).replace(".ckpt", ".pt")

    log.info(f"Saving model_g to: {output_path}")
    torch.save(model.model_g, output_path)

    # Also copy config.json if it exists alongside the ckpt
    config_src = Path(ckpt_path).parent / "config.json"
    config_dst = Path(output_path).with_suffix(".pt.json")
    if config_src.exists() and not config_dst.exists():
        import shutil
        shutil.copy2(str(config_src), str(config_dst))
        log.info(f"Copied config: {config_dst}")

    log.info("Done!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", default=None, help="Output .pt path (default: same name with .pt)")
    args = parser.parse_args()
    convert_ckpt_to_pt(args.ckpt, args.output)

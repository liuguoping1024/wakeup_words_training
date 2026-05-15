#!/usr/bin/env bash
# 4 GPU 并行生成对抗性负样本
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="outputs/adversarial_negatives"
COSYVOICE_IMAGE="cosyvoice:latest"
MODEL_DIR="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# 每个 GPU 处理 10 个说话人 (40 / 4 = 10)
# 总目标：3000 ming + 1500 jiu + 1500 daily = 6000 条
for GPU in 0 1 2 3; do
    START=$((GPU * 10))
    END=$(((GPU + 1) * 10))
    SEED=$((42 + GPU))

    echo "[GPU $GPU] 说话人 $START-$END, seed=$SEED"

    docker run -d --rm \
        --gpus "device=$GPU" \
        --name "adversarial_gpu${GPU}" \
        -v "${HOST_DIR}/data:/workspace/data" \
        -v "${HOST_DIR}/outputs:/workspace/outputs" \
        -v "${HOST_DIR}/scripts:/workspace/scripts" \
        -v "${HOST_DIR}/work:/workspace/work" \
        --shm-size=4g \
        ${COSYVOICE_IMAGE} \
        "python3 -u /workspace/scripts/mww/generate_adversarial_negatives.py \
            --output-dir /workspace/${OUTPUT_BASE}/gpu${GPU} \
            --refs-dir /workspace/data/speaker_refs \
            --model-dir ${MODEL_DIR} \
            --n-ming 3000 --n-jiu 1500 --n-daily 1500 \
            --speakers-start ${START} --speakers-end ${END} \
            --seed ${SEED}"
done

echo ""
echo "4 个 GPU 已启动，监控："
echo "  docker logs -f adversarial_gpu0"
echo "  docker logs -f adversarial_gpu1"
echo "  docker logs -f adversarial_gpu2"
echo "  docker logs -f adversarial_gpu3"

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# OWW 中文唤醒词训练 Pipeline（宿主机入口）
#
# 流程：
#   Step 0: 在宿主机用 edge-tts 生成对抗性负样本（需要网络）
#   Step 1: 启动 Docker 容器执行训练（prepare → augment → train → export）
#
# 用法：
#   bash scripts/oww/train_zh_pipeline.sh [--skip-tts] [--keyword 救命] [--id jiuming]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 参数 ──
KEYWORD="${KEYWORD:-你好树实}"
KEYWORD_ID="${KEYWORD_ID:-nihao_shushi}"
SKIP_TTS="${SKIP_TTS:-0}"
N_NEG_TRAIN="${N_NEG_TRAIN:-5000}"
N_NEG_TEST="${N_NEG_TEST:-1000}"
N_POS_TRAIN="${N_POS_TRAIN:-15000}"
N_POS_TEST="${N_POS_TEST:-3000}"
AUGMENT_ROUNDS="${AUGMENT_ROUNDS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-80000}"
LAYER_SIZE="${LAYER_SIZE:-64}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tts) SKIP_TTS=1; shift ;;
        --keyword) KEYWORD="$2"; shift 2 ;;
        --id) KEYWORD_ID="$2"; shift 2 ;;
        --steps) TRAIN_STEPS="$2"; shift 2 ;;
        --layer-size) LAYER_SIZE="$2"; shift 2 ;;
        --augment-rounds) AUGMENT_ROUNDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="${HOST_DIR}/outputs/oww/${KEYWORD_ID}"
NEG_TRAIN_DIR="${MODEL_DIR}/negative_train"
NEG_TEST_DIR="${MODEL_DIR}/negative_test"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

echo "═══════════════════════════════════════════════════"
echo "  OWW 中文唤醒词训练: ${KEYWORD} (${KEYWORD_ID})"
echo "  正样本: ${N_POS_TRAIN} train / ${N_POS_TEST} test"
echo "  负样本: ${N_NEG_TRAIN} train / ${N_NEG_TEST} test"
echo "  增强轮数: ${AUGMENT_ROUNDS}"
echo "  训练步数: ${TRAIN_STEPS}, layer_size: ${LAYER_SIZE}"
echo "═══════════════════════════════════════════════════"

# ── Step 0: 宿主机生成对抗性负样本 ──
if [[ "$SKIP_TTS" == "0" ]]; then
    echo ""
    echo ">>> Step 0: 生成对抗性负样本 (edge-tts, 宿主机)"
    python3 "${HOST_DIR}/scripts/oww/generate_zh_negatives.py" \
        --keyword "$KEYWORD" \
        --output-dir "$MODEL_DIR" \
        --n-train "$N_NEG_TRAIN" \
        --n-test "$N_NEG_TEST"
else
    echo ""
    echo ">>> Step 0: 跳过 TTS 生成 (--skip-tts)"
    EXISTING_NEG=$(ls "$NEG_TRAIN_DIR"/*.wav 2>/dev/null | wc -l)
    echo "    已有负样本: ${EXISTING_NEG} train"
fi

# ── Step 1: Docker 内训练 ──
echo ""
echo ">>> Step 1: Docker 内训练"

OWW_IMAGE="${OWW_IMAGE:-wakeword-oww:latest}"

docker run --rm --gpus all \
    -e PYTHONUNBUFFERED=1 \
    --shm-size=8g \
    -v "${HOST_DIR}/data:/workspace/data" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    -v "${HOST_DIR}/logs:/workspace/logs" \
    "$OWW_IMAGE" \
    "pip install -e /workspace/work/openWakeWord --no-deps -q && \
     python3 -u /workspace/scripts/oww/train_zh_oww.py \
       --keyword-id ${KEYWORD_ID} \
       --positive-dir /workspace/data/positive_raw/${KEYWORD_ID} \
       --n-pos-train ${N_POS_TRAIN} \
       --n-pos-test ${N_POS_TEST} \
       --augment-rounds ${AUGMENT_ROUNDS} \
       --steps ${TRAIN_STEPS} \
       --layer-size ${LAYER_SIZE} \
       --overwrite-features" \
    2>&1 | tee "${LOG_DIR}/oww_${KEYWORD_ID}_zh.log"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  训练完成! 模型: outputs/oww/${KEYWORD_ID}.onnx"
echo "═══════════════════════════════════════════════════"

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 救命救命 v2 完整训练管线
# 改进：充分混合 CosyVoice + edge-tts + Piper，均衡比例
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 救命救命 v2 训练管线 ==="

# ─── Step 1: edge-tts（大幅增加）─────────────────────────────
log "Step 1: edge-tts 生成..."
python3.12 "${HOST_DIR}/scripts/mww/generate_jiuming2_edgetts_v2.py" \
    "outputs/jiuming2_edgetts_v2" 2>&1 | tee "${LOG_DIR}/jiuming2_edgetts_v2.log"
log "edge-tts done"

# ─── Step 2: Piper（中文 huayan 模型）────────────────────────
log "Step 2: Piper 生成..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/data:/workspace/data" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    -v "${HOST_DIR}/work:/workspace/work" \
    wakeword-oww:latest \
    "pip install -e /workspace/work/piper-sample-generator-oww --no-deps -q 2>/dev/null; \
     bash /workspace/scripts/mww/generate_jiuming2_piper.sh" \
    > "${LOG_DIR}/jiuming2_piper.log" 2>&1 || log "Piper failed, continuing..."
log "Piper done"

# ─── Step 3: CosyVoice 4 GPU ─────────────────────────────────
log "Step 3: CosyVoice 4 GPU..."
COSYVOICE_IMAGE="cosyvoice:latest"
MODEL_DIR="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B"

for GPU in 0 1 2 3; do
    START=$((GPU * 10))
    END=$(((GPU + 1) * 10))
    log "  GPU $GPU (speakers $START-$END)"
    docker run --rm \
        --gpus "device=$GPU" \
        --name "jiuming2v2_cosy_gpu${GPU}" \
        -v "${HOST_DIR}/data:/workspace/data" \
        -v "${HOST_DIR}/outputs:/workspace/outputs" \
        -v "${HOST_DIR}/scripts:/workspace/scripts" \
        -v "${HOST_DIR}/work:/workspace/work" \
        --shm-size=4g \
        ${COSYVOICE_IMAGE} \
        "python3 -u /workspace/scripts/mww/generate_jiuming2_cosyvoice.py \
            --output-dir /workspace/outputs/jiuming2_cosyvoice_v2/gpu${GPU} \
            --refs-dir /workspace/data/speaker_refs \
            --model-dir ${MODEL_DIR} \
            --n-pos 1250 --n-neg 1250 \
            --speakers-start ${START} --speakers-end ${END} \
            --seed $((42 + GPU))" \
        > "${LOG_DIR}/jiuming2v2_cosy_gpu${GPU}.log" 2>&1 &
done
log "  Waiting for CosyVoice..."
wait
log "CosyVoice done"

# ─── Step 4: 合并数据 ─────────────────────────────────────────
log "Step 4: Merging data..."

POSDIR="data/positive_augmented/jiuming2_v2"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# edge-tts 正样本
if [ -d "outputs/jiuming2_edgetts_v2/positive" ]; then
    for f in outputs/jiuming2_edgetts_v2/positive/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
    done
fi
log "  edge-tts pos: $(find $POSDIR -name 'edge_*' | wc -l)"

# Piper 正样本
if [ -d "outputs/jiuming2_piper/positive" ]; then
    for f in outputs/jiuming2_piper/positive/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/piper_$(basename $f)" 2>/dev/null || true
    done
fi
log "  Piper pos: $(find $POSDIR -name 'piper_*' | wc -l)"

# CosyVoice 正样本
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v2/gpu${GPU}/positive"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  CosyVoice pos: $(find $POSDIR -name 'cosy*' | wc -l)"

POS_TOTAL=$(find "$POSDIR" -name "*.wav" | wc -l)
log "  Positive total: $POS_TOTAL"

# 对抗性负样本
NEGDIR="outputs/jiuming2_adversarial_v2"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

if [ -d "outputs/jiuming2_edgetts_v2/negative" ]; then
    for f in outputs/jiuming2_edgetts_v2/negative/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
    done
fi
if [ -d "outputs/jiuming2_piper/negative" ]; then
    for f in outputs/jiuming2_piper/negative/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/piper_$(basename $f)" 2>/dev/null || true
    done
fi
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v2/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
NEG_TOTAL=$(find "$NEGDIR" -name "*.wav" | wc -l)
log "  Adversarial neg total: $NEG_TOTAL"

# ─── Step 5: MWW 训练 ─────────────────────────────────────────
log "Step 5: MWW training..."
docker run --rm --gpus all \
    --name mww_jiuming2_v2_train \
    -e PYTHONUNBUFFERED=1 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "${HOST_DIR}/data:/workspace/data" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    -v "${HOST_DIR}/logs:/workspace/logs" \
    -v "${HOST_DIR}/inference:/workspace/inference" \
    wakeword-mww:latest \
    "chmod +x /workspace/scripts/mww/*.sh && bash /workspace/scripts/mww/run_jiuming2_v2.sh" \
    >> "${LOG_DIR}/mww_jiuming2_v2.log" 2>&1

log "=== 救命救命 v2 pipeline complete ==="

#!/usr/bin/env bash
# help help v4: 严格正样本 3 个 + 120 个双音节负样本
# 无真实录音，全 TTS
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help v4: 严格正样本 ==="

# Step 1: edge-tts
log "Step 1: edge-tts..."
python3.12 "${HOST_DIR}/scripts/mww/generate_help_help_v4_edgetts.py" \
    "outputs/help_help_edgetts_v4" 2>&1 | tee "${LOG_DIR}/help_help_v4_edgetts.log"
log "edge-tts done"

# Step 2: Piper
log "Step 2: Piper libritts..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    wakeword-oww:latest \
    "bash /workspace/scripts/mww/generate_help_help_v4_piper.sh" \
    > "${LOG_DIR}/help_help_v4_piper.log" 2>&1 || log "Piper failed, continuing..."
log "Piper done"

# Step 3: CosyVoice 4 GPU
log "Step 3: CosyVoice 4 GPU..."
COSYVOICE_IMAGE="cosyvoice:latest"
MODEL_DIR="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B"

for GPU in 0 1 2 3; do
    START=$((GPU * 10))
    END=$(((GPU + 1) * 10))
    log "  GPU $GPU (speakers $START-$END)"
    docker run --rm \
        --gpus "device=$GPU" \
        --name "helphelpv4_cosy_gpu${GPU}" \
        -v "${HOST_DIR}/data:/workspace/data" \
        -v "${HOST_DIR}/outputs:/workspace/outputs" \
        -v "${HOST_DIR}/scripts:/workspace/scripts" \
        -v "${HOST_DIR}/work:/workspace/work" \
        --shm-size=4g \
        ${COSYVOICE_IMAGE} \
        "python3 -u /workspace/scripts/mww/generate_help_help_v4_cosyvoice.py \
            --output-dir /workspace/outputs/help_help_cosyvoice_v4/gpu${GPU} \
            --refs-dir /workspace/data/speaker_refs \
            --model-dir ${MODEL_DIR} \
            --pos-per-phrase 800 --neg-per-phrase 40 \
            --speakers-start ${START} --speakers-end ${END} \
            --seed $((42 + GPU))" \
        > "${LOG_DIR}/helphelpv4_cosy_gpu${GPU}.log" 2>&1 &
done
log "  Waiting for CosyVoice..."
wait
log "CosyVoice done"

# Step 4: 合并
log "Step 4: Merging..."

POSDIR="data/positive_augmented/help_help_v4"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# edge-tts
for f in outputs/help_help_edgetts_v4/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
log "  edge: $(find $POSDIR -name 'edge_*' | wc -l)"

# Piper
find outputs/help_help_piper_v4/positive/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$POSDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
log "  piper: $(find $POSDIR -name 'piper_*' | wc -l)"

# CosyVoice
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice_v4/gpu${GPU}/positive"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  cosy: $(find $POSDIR -name 'cosy*' | wc -l)"
log "  === Positive total: $(find $POSDIR -name '*.wav' | wc -l) ==="

# 对抗性负样本
NEGDIR="outputs/help_help_adversarial_v4"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

for f in outputs/help_help_edgetts_v4/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done
find outputs/help_help_piper_v4/negative/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$NEGDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice_v4/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  === Adversarial neg total: $(find $NEGDIR -name '*.wav' | wc -l) ==="

# Step 5: MWW 训练
log "Step 5: MWW training..."
docker run --rm --gpus all \
    --name mww_help_help_v4_train \
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
    "bash /workspace/scripts/mww/run_help_help_v4.sh" \
    >> "${LOG_DIR}/mww_help_help_v4.log" 2>&1

log "=== help help v4 complete ==="

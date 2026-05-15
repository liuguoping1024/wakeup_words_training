#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# help help 完整训练管线
# 1. edge-tts 生成（宿主机，后台）
# 2. CosyVoice 4GPU 并行生成
# 3. Piper 生成
# 4. 合并数据 + 生成 mmap 特征
# 5. MWW 训练
# 6. 验证
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help 完整训练管线 ==="

# ─── Step 1: edge-tts（宿主机后台）─────────────────────────────
log "Step 1: edge-tts 生成..."
python3.12 "${HOST_DIR}/scripts/mww/generate_help_help_edgetts.py" \
    "outputs/help_help_edgetts" 2>&1 | tee "${LOG_DIR}/help_help_edgetts.log"
log "edge-tts 完成"

# ─── Step 2: CosyVoice 4 GPU 并行 ─────────────────────────────
log "Step 2: CosyVoice 4 GPU 并行生成..."
COSYVOICE_IMAGE="cosyvoice:latest"
MODEL_DIR="/workspace/work/CosyVoice/pretrained_models/CosyVoice2-0.5B"

for GPU in 0 1 2 3; do
    START=$((GPU * 10))
    END=$(((GPU + 1) * 10))
    SEED=$((42 + GPU))

    log "  启动 GPU $GPU (说话人 $START-$END)"
    docker run --rm \
        --gpus "device=$GPU" \
        --name "helphelp_cosy_gpu${GPU}" \
        -v "${HOST_DIR}/data:/workspace/data" \
        -v "${HOST_DIR}/outputs:/workspace/outputs" \
        -v "${HOST_DIR}/scripts:/workspace/scripts" \
        -v "${HOST_DIR}/work:/workspace/work" \
        --shm-size=4g \
        ${COSYVOICE_IMAGE} \
        "python3 -u /workspace/scripts/mww/generate_help_help_cosyvoice.py \
            --output-dir /workspace/outputs/help_help_cosyvoice/gpu${GPU} \
            --refs-dir /workspace/data/speaker_refs \
            --model-dir ${MODEL_DIR} \
            --n-pos 2000 --n-neg 1500 \
            --speakers-start ${START} --speakers-end ${END} \
            --seed ${SEED}" \
        > "${LOG_DIR}/help_help_cosy_gpu${GPU}.log" 2>&1 &
done

log "  等待 CosyVoice 完成..."
wait
log "CosyVoice 完成"

# ─── Step 3: Piper 生成 ───────────────────────────────────────
log "Step 3: Piper 生成..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/data:/workspace/data" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    -v "${HOST_DIR}/work:/workspace/work" \
    wakeword-oww:latest \
    "pip install -e /workspace/work/piper-sample-generator-oww --no-deps -q 2>/dev/null; \
     bash /workspace/scripts/mww/generate_help_help_piper.sh" \
    > "${LOG_DIR}/help_help_piper.log" 2>&1
log "Piper 完成"

# ─── Step 4: 合并数据 ─────────────────────────────────────────
log "Step 4: 合并数据..."

POSDIR="data/positive_augmented/help_help_v1"
mkdir -p "$POSDIR"

# 真实录音 × 5
log "  真实录音 × 5..."
for rep in $(seq 0 4); do
    for f in data/real_voices_help_help/*.wav; do
        [ -f "$f" ] || continue
        bn=$(basename "$f" .wav)
        out="$POSDIR/real_r${rep}_${bn}.wav"
        [ -f "$out" ] || sox "$f" "$out" gain -n -3 2>/dev/null
    done
done

# edge-tts 正样本
log "  edge-tts 正样本..."
if [ -d "outputs/help_help_edgetts/positive" ]; then
    for f in outputs/help_help_edgetts/positive/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
    done
fi

# CosyVoice 正样本
log "  CosyVoice 正样本..."
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice/gpu${GPU}/positive"
    if [ -d "$d" ]; then
        for f in "$d"/*.wav; do
            [ -f "$f" ] || continue
            cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
        done
    fi
done

# Piper 正样本
log "  Piper 正样本..."
if [ -d "outputs/help_help_piper/positive" ]; then
    for f in outputs/help_help_piper/positive/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/piper_$(basename $f)" 2>/dev/null || true
    done
fi

POS_TOTAL=$(find "$POSDIR" -name "*.wav" | wc -l)
log "  正样本总计: $POS_TOTAL"

# 合并对抗性负样本
NEGDIR="outputs/help_help_adversarial_merged"
mkdir -p "$NEGDIR"

log "  合并对抗性负样本..."
# edge-tts 负样本
if [ -d "outputs/help_help_edgetts/negative" ]; then
    for f in outputs/help_help_edgetts/negative/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
    done
fi

# CosyVoice 负样本
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice/gpu${GPU}/negative"
    if [ -d "$d" ]; then
        for f in "$d"/*.wav; do
            [ -f "$f" ] || continue
            cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
        done
    fi
done

# Piper 负样本
if [ -d "outputs/help_help_piper/negative" ]; then
    for f in outputs/help_help_piper/negative/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/piper_$(basename $f)" 2>/dev/null || true
    done
fi

NEG_TOTAL=$(find "$NEGDIR" -name "*.wav" | wc -l)
log "  对抗性负样本总计: $NEG_TOTAL"

# ─── Step 5: MWW 训练 ─────────────────────────────────────────
log "Step 5: MWW 训练..."
docker run --rm --gpus all \
    --name mww_help_help_train \
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
    "chmod +x /workspace/scripts/mww/*.sh && bash /workspace/scripts/mww/run_help_help_v1.sh" \
    >> "${LOG_DIR}/mww_help_help_v1.log" 2>&1

log "=== 全部完成 ==="

#!/usr/bin/env bash
# help help v2 完整训练管线
# 改进：+ Piper + 更多 edge-tts + 更多对抗性负样本 + 真实录音 ×10
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help v2 训练管线 ==="

# ─── Step 1: edge-tts（22 英文声音 × 扩展词表）─────────────────
log "Step 1: edge-tts..."
python3.12 "${HOST_DIR}/scripts/mww/generate_help_help_v2_edgetts.py" \
    "outputs/help_help_edgetts_v2" 2>&1 | tee "${LOG_DIR}/help_help_v2_edgetts.log"
log "edge-tts done"

# ─── Step 2: Piper libritts（多说话人英文）────────────────────
log "Step 2: Piper libritts..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    wakeword-oww:latest \
    "bash /workspace/scripts/mww/generate_help_help_v2_piper.sh" \
    > "${LOG_DIR}/help_help_v2_piper.log" 2>&1 || log "Piper failed, continuing..."
log "Piper done"

# ─── Step 3: CosyVoice（复用 v1 的 2000 条）───────────────────
log "Step 3: CosyVoice (reuse v1)"
V1_COSY=$(find outputs/help_help_cosyvoice -path '*/positive/*.wav' 2>/dev/null | wc -l)
log "  CosyVoice v1 available: $V1_COSY"

# ─── Step 4: 合并 ─────────────────────────────────────────────
log "Step 4: Merging..."

POSDIR="data/positive_augmented/help_help_v2"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# 真实录音 × 10（v1 是 ×5）
log "  Real voices × 10..."
for rep in $(seq 0 9); do
    for f in data/real_voices_help_help/*.wav; do
        [ -f "$f" ] || continue
        bn=$(basename "$f" .wav)
        out="$POSDIR/real_r${rep}_${bn}.wav"
        [ -f "$out" ] || sox "$f" "$out" gain -n -3 2>/dev/null
    done
done
log "    real: $(find $POSDIR -name 'real_*' | wc -l)"

# edge-tts v2
for f in outputs/help_help_edgetts_v2/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
log "    edge: $(find $POSDIR -name 'edge_*' | wc -l)"

# Piper v2（递归展平）
find outputs/help_help_piper_v2/positive/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$POSDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
log "    piper: $(find $POSDIR -name 'piper_*' | wc -l)"

# CosyVoice（复用 v1）
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice/gpu${GPU}/positive"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "    cosy: $(find $POSDIR -name 'cosy*' | wc -l)"
log "  === Positive total: $(find $POSDIR -name '*.wav' | wc -l) ==="

# 对抗性负样本
NEGDIR="outputs/help_help_adversarial_v2"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

for f in outputs/help_help_edgetts_v2/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done
find outputs/help_help_piper_v2/negative/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$NEGDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
# CosyVoice 负样本（v1 的）
for GPU in 0 1 2 3; do
    d="outputs/help_help_cosyvoice/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  === Adversarial neg total: $(find $NEGDIR -name '*.wav' | wc -l) ==="

# ─── Step 5: MWW 训练 ─────────────────────────────────────────
log "Step 5: MWW training..."
docker run --rm --gpus all \
    --name mww_help_help_v2_train \
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
    "bash /workspace/scripts/mww/run_help_help_v2.sh" \
    >> "${LOG_DIR}/mww_help_help_v2.log" 2>&1

log "=== help help v2 complete ==="

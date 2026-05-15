#!/usr/bin/env bash
# 救命救命 v3: CosyVoice 完成后，合并数据 + 训练
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 救命救命 v3: 等待 CosyVoice 完成 ==="

# 等待 4 个 CosyVoice 容器完成
while docker ps --format "{{.Names}}" | grep -q "jiuming2v3_cosy"; do
    sleep 30
done
log "CosyVoice all done"

# 检查生成量
for GPU in 0 1 2 3; do
    pos=$(find outputs/jiuming2_cosyvoice_v3/gpu${GPU}/positive -name "*.wav" 2>/dev/null | wc -l)
    neg=$(find outputs/jiuming2_cosyvoice_v3/gpu${GPU}/negative -name "*.wav" 2>/dev/null | wc -l)
    log "  GPU $GPU: pos=$pos neg=$neg"
done

# 合并数据
log "Merging data..."
POSDIR="data/positive_augmented/jiuming2_v3"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# edge-tts 正样本（复用 v2 的）
for f in outputs/jiuming2_edgetts_v2/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
log "  edge-tts pos: $(find $POSDIR -name 'edge_*' | wc -l)"

# CosyVoice 正样本
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v3/gpu${GPU}/positive"
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
NEGDIR="outputs/jiuming2_adversarial_v3"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

for f in outputs/jiuming2_edgetts_v2/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v3/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
NEG_TOTAL=$(find "$NEGDIR" -name "*.wav" | wc -l)
log "  Adversarial neg total: $NEG_TOTAL"

# MWW 训练
log "Starting MWW training..."
docker run --rm --gpus all \
    --name mww_jiuming2_v3_train \
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
    "chmod +x /workspace/scripts/mww/*.sh && bash /workspace/scripts/mww/run_jiuming2_v3.sh" \
    >> "${LOG_DIR}/mww_jiuming2_v3.log" 2>&1

log "=== 救命救命 v3 complete ==="

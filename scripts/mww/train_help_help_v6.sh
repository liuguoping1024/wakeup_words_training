#!/usr/bin/env bash
# help help v6: 慢速 + 严格正样本 + 无 CosyVoice 英文
# edge-tts (慢速) + Piper (length_scale 2.0-3.0)
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help v6: 慢速，无 CosyVoice ==="

# Step 1: edge-tts
log "Step 1: edge-tts (慢速 -15%)..."
python3.12 "${HOST_DIR}/scripts/mww/generate_help_help_v6_edgetts.py" \
    "outputs/help_help_edgetts_v6" 2>&1 | tee "${LOG_DIR}/help_help_v6_edgetts.log"
log "edge-tts done"

# Step 2: Piper libritts 慢速
log "Step 2: Piper libritts (慢速 length_scales 2.0-3.0)..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    -v "${HOST_DIR}/scripts:/workspace/scripts" \
    wakeword-oww:latest \
    "bash /workspace/scripts/mww/generate_help_help_v6_piper.sh" \
    > "${LOG_DIR}/help_help_v6_piper.log" 2>&1 || log "Piper had issues"
log "Piper done"

# Step 3: 合并
log "Step 3: Merging..."

POSDIR="data/positive_augmented/help_help_v6"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# edge-tts 正样本
for f in outputs/help_help_edgetts_v6/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
log "  edge: $(find $POSDIR -name 'edge_*' | wc -l)"

# Piper 正样本
for f in outputs/help_help_piper_v6/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/piper_$(basename $f)" 2>/dev/null || true
done
log "  piper: $(find $POSDIR -name 'piper_*' | wc -l)"

log "  === Positive total: $(find $POSDIR -name '*.wav' | wc -l) ==="

# 对抗性负样本
NEGDIR="outputs/help_help_adversarial_v6"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

for f in outputs/help_help_edgetts_v6/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done
find outputs/help_help_piper_v6/negative/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$NEGDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
log "  === Adversarial neg total: $(find $NEGDIR -name '*.wav' | wc -l) ==="

# Step 4: 检查正样本时长分布
log "Step 4: 正样本时长分布（应该 > 1 秒）..."
python3 -c "
import os, wave
d = 'data/positive_augmented/help_help_v6'
durs = []
for f in os.listdir(d):
    if not f.endswith('.wav'): continue
    path = os.path.join(d, f)
    try:
        with wave.open(path, 'rb') as w:
            durs.append(w.getnframes()/w.getframerate())
    except: pass
if durs:
    durs.sort()
    print(f'  total: {len(durs)}')
    print(f'  min:  {durs[0]:.2f}s')
    print(f'  p25:  {durs[len(durs)//4]:.2f}s')
    print(f'  p50:  {durs[len(durs)//2]:.2f}s')
    print(f'  p75:  {durs[3*len(durs)//4]:.2f}s')
    print(f'  max:  {durs[-1]:.2f}s')
    print(f'  < 1s: {sum(1 for d in durs if d < 1.0)} ({sum(1 for d in durs if d < 1.0)*100/len(durs):.1f}%)')
"

# Step 5: MWW 训练
log "Step 5: MWW training..."
docker run --rm --gpus all \
    --name mww_help_help_v6_train \
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
    "bash /workspace/scripts/mww/run_help_help_v6.sh" \
    >> "${LOG_DIR}/mww_help_help_v6.log" 2>&1

log "=== help help v6 complete ==="

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 救命救命 v4: 全部 TTS 都用上
# CosyVoice (复用v3) + Piper huayan + edge-tts 14声音
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 救命救命 v4: 全部 TTS ==="

# ─── Step 1: Piper huayan 正样本 + 负样本 ─────────────────────
log "Step 1: Piper huayan..."
docker run --rm --gpus all \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    wakeword-oww:latest \
    "cd /workspace/work/piper-sample-generator-oww && \
     echo '=== Piper positive ===' && \
     for phrase in '救命救命' '救命救命啊' '快救命救命' '久名久名' '救民救民' '揪命揪命' '九命九命' '救命救命呀' '纠命纠命' '救民救命' '救命救民' '久名救命' '救命久名'; do \
       safe=\$(echo \"\$phrase\" | md5sum | head -c 8); \
       outdir=/workspace/outputs/jiuming2_piper_v4/positive/\${safe}; \
       mkdir -p \$outdir; \
       existing=\$(find \$outdir -name '*.wav' 2>/dev/null | wc -l); \
       if [ \$existing -ge 180 ]; then echo \"  [skip] \$phrase: \$existing\"; continue; fi; \
       echo \"  gen: \$phrase (200)\"; \
       python3 generate_samples.py \"\$phrase\" \
         --model models/zh_CN-huayan-medium.pt \
         --output-dir \$outdir \
         --max-samples 200 --batch-size 10 2>&1 | tail -1; \
     done && \
     echo '=== Piper negative ===' && \
     for phrase in '救命' '救命啊' '快救命' '救火' '救火救火' '革命' '聪明' '说明' '你好' '谢谢' '打开灯' '生命' '要命' '拼命' '光明'; do \
       safe=\$(echo \"\$phrase\" | md5sum | head -c 8); \
       outdir=/workspace/outputs/jiuming2_piper_v4/negative/\${safe}; \
       mkdir -p \$outdir; \
       existing=\$(find \$outdir -name '*.wav' 2>/dev/null | wc -l); \
       if [ \$existing -ge 80 ]; then echo \"  [skip] \$phrase: \$existing\"; continue; fi; \
       echo \"  gen: \$phrase (100)\"; \
       python3 generate_samples.py \"\$phrase\" \
         --model models/zh_CN-huayan-medium.pt \
         --output-dir \$outdir \
         --max-samples 100 --batch-size 10 2>&1 | tail -1; \
     done && \
     echo '=== Piper summary ===' && \
     echo \"pos: \$(find /workspace/outputs/jiuming2_piper_v4/positive -name '*.wav' | wc -l)\" && \
     echo \"neg: \$(find /workspace/outputs/jiuming2_piper_v4/negative -name '*.wav' | wc -l)\"" \
    > "${LOG_DIR}/jiuming2_piper_v4.log" 2>&1
log "Piper done"
cat "${LOG_DIR}/jiuming2_piper_v4.log" | grep -E "pos:|neg:|skip|summary" | tail -5

# ─── Step 2: edge-tts 全部 14 中文声音 ────────────────────────
log "Step 2: edge-tts 14 Chinese voices..."
# 复用 v2 的 edge-tts（已有 216 正 + 301 负），不重新生成
EDGE_POS=$(find outputs/jiuming2_edgetts_v2/positive -name "*.wav" 2>/dev/null | wc -l)
EDGE_NEG=$(find outputs/jiuming2_edgetts_v2/negative -name "*.wav" 2>/dev/null | wc -l)
log "  edge-tts (reuse v2): pos=$EDGE_POS neg=$EDGE_NEG"

# ─── Step 3: 合并全部数据 ─────────────────────────────────────
log "Step 3: Merging ALL TTS data..."

POSDIR="data/positive_augmented/jiuming2_v4"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# CosyVoice（复用 v3 的 12000 条）
log "  CosyVoice..."
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v3/gpu${GPU}/positive"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
COSY_POS=$(find "$POSDIR" -name "cosy*" | wc -l)
log "    cosy: $COSY_POS"

# Piper
log "  Piper..."
find outputs/jiuming2_piper_v4/positive -name "*.wav" 2>/dev/null | while read f; do
    cp -n "$f" "$POSDIR/piper_$(basename $f)" 2>/dev/null || true
done
PIPER_POS=$(find "$POSDIR" -name "piper*" | wc -l)
log "    piper: $PIPER_POS"

# edge-tts
log "  edge-tts..."
for f in outputs/jiuming2_edgetts_v2/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
EDGE_POS=$(find "$POSDIR" -name "edge*" | wc -l)
log "    edge: $EDGE_POS"

POS_TOTAL=$(find "$POSDIR" -name "*.wav" | wc -l)
log "  === Positive total: $POS_TOTAL ==="

# 对抗性负样本
NEGDIR="outputs/jiuming2_adversarial_v4"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

# CosyVoice 负样本（复用 v3）
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v3/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done

# Piper 负样本
find outputs/jiuming2_piper_v4/negative -name "*.wav" 2>/dev/null | while read f; do
    cp -n "$f" "$NEGDIR/piper_$(basename $f)" 2>/dev/null || true
done

# edge-tts 负样本
for f in outputs/jiuming2_edgetts_v2/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done

NEG_TOTAL=$(find "$NEGDIR" -name "*.wav" | wc -l)
log "  === Adversarial neg total: $NEG_TOTAL ==="

# ─── Step 4: MWW 训练 ─────────────────────────────────────────
log "Step 4: MWW training..."
docker run --rm --gpus all \
    --name mww_jiuming2_v4_train \
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
    "chmod +x /workspace/scripts/mww/*.sh && bash /workspace/scripts/mww/run_jiuming2_v4.sh" \
    >> "${LOG_DIR}/mww_jiuming2_v4.log" 2>&1

log "=== 救命救命 v4 complete ==="

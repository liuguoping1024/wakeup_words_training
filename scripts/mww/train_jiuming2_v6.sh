#!/usr/bin/env bash
# 救命救命 v6: 严格叠词 + 全部中文TTS
set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${HOST_DIR}/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== 救命救命 v6: 严格叠词 + 全部TTS ==="

# ─── Step 1: edge-tts (14 中文声音) ──────────────────────────
log "Step 1: edge-tts..."
python3.12 "${HOST_DIR}/scripts/mww/generate_jiuming2_v6_edgetts.py" \
    "outputs/jiuming2_edgetts_v6" 2>&1 | tee "${LOG_DIR}/jiuming2_v6_edgetts.log"
log "edge-tts done"

# ─── Step 2: Piper huayan ────────────────────────────────────
log "Step 2: Piper huayan..."
PIPER_POS_PHRASES=("救命救命" "救命救命啊" "久名久名" "救民救民" "揪命揪命" "九命九命" "纠命纠命" "酒命酒命")
PIPER_NEG_PHRASES=("救命" "救命啊" "救火" "救火救火" "革命" "聪明" "说明" "你好" "谢谢" "打开灯" "生命" "要命" "拼命" "光明" "播放音乐")

docker run --rm --gpus all \
    -v "${HOST_DIR}/work:/workspace/work" \
    -v "${HOST_DIR}/outputs:/workspace/outputs" \
    wakeword-oww:latest \
    "cd /workspace/work/piper-sample-generator-oww && \
     mkdir -p /workspace/outputs/jiuming2_piper_v6/positive /workspace/outputs/jiuming2_piper_v6/negative && \
     for phrase in '救命救命' '救命救命啊' '久名久名' '救民救民' '揪命揪命' '九命九命' '纠命纠命' '酒命酒命'; do \
       safe=\$(echo \"\$phrase\" | md5sum | head -c 8); \
       outdir=/workspace/outputs/jiuming2_piper_v6/positive/\${safe}; \
       mkdir -p \$outdir; \
       cnt=\$(find \$outdir -name '*.wav' 2>/dev/null | wc -l); \
       if [ \$cnt -ge 180 ]; then echo \"[skip] \$phrase: \$cnt\"; continue; fi; \
       echo \"[pos] \$phrase (200)\"; \
       python3 generate_samples.py \"\$phrase\" --model models/zh_CN-huayan-medium.pt --output-dir \$outdir --max-samples 200 --batch-size 10 2>&1 | tail -1; \
     done && \
     for phrase in '救命' '救命啊' '救火' '救火救火' '革命' '聪明' '说明' '你好' '谢谢' '打开灯' '生命' '要命' '拼命' '光明' '播放音乐'; do \
       safe=\$(echo \"\$phrase\" | md5sum | head -c 8); \
       outdir=/workspace/outputs/jiuming2_piper_v6/negative/\${safe}; \
       mkdir -p \$outdir; \
       cnt=\$(find \$outdir -name '*.wav' 2>/dev/null | wc -l); \
       if [ \$cnt -ge 80 ]; then echo \"[skip] \$phrase: \$cnt\"; continue; fi; \
       echo \"[neg] \$phrase (100)\"; \
       python3 generate_samples.py \"\$phrase\" --model models/zh_CN-huayan-medium.pt --output-dir \$outdir --max-samples 100 --batch-size 10 2>&1 | tail -1; \
     done && \
     echo \"Piper pos: \$(find /workspace/outputs/jiuming2_piper_v6/positive -name '*.wav' | wc -l)\" && \
     echo \"Piper neg: \$(find /workspace/outputs/jiuming2_piper_v6/negative -name '*.wav' | wc -l)\"" \
    > "${LOG_DIR}/jiuming2_v6_piper.log" 2>&1
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
        --name "jiuming2v6_cosy_gpu${GPU}" \
        -v "${HOST_DIR}/data:/workspace/data" \
        -v "${HOST_DIR}/outputs:/workspace/outputs" \
        -v "${HOST_DIR}/scripts:/workspace/scripts" \
        -v "${HOST_DIR}/work:/workspace/work" \
        --shm-size=4g \
        ${COSYVOICE_IMAGE} \
        "python3 -u /workspace/scripts/mww/generate_jiuming2_v6_cosyvoice.py \
            --output-dir /workspace/outputs/jiuming2_cosyvoice_v6/gpu${GPU} \
            --refs-dir /workspace/data/speaker_refs \
            --model-dir ${MODEL_DIR} \
            --n-pos 12000 --n-neg 8000 \
            --speakers-start ${START} --speakers-end ${END} \
            --seed $((42 + GPU))" \
        > "${LOG_DIR}/jiuming2v6_cosy_gpu${GPU}.log" 2>&1 &
done
log "  Waiting for CosyVoice..."
wait
log "CosyVoice done"

# ─── Step 4: 合并 ─────────────────────────────────────────────
log "Step 4: Merging..."

POSDIR="data/positive_augmented/jiuming2_v6"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# edge-tts
for f in outputs/jiuming2_edgetts_v6/positive/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/edge_$(basename $f)" 2>/dev/null || true
done
log "  edge: $(find $POSDIR -name 'edge_*' | wc -l)"

# Piper（递归展平）
find outputs/jiuming2_piper_v6/positive/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$POSDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
log "  piper: $(find $POSDIR -name 'piper_*' | wc -l)"

# CosyVoice
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v6/gpu${GPU}/positive"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$POSDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  cosy: $(find $POSDIR -name 'cosy*' | wc -l)"
log "  === Positive total: $(find $POSDIR -name '*.wav' | wc -l) ==="

# 负样本
NEGDIR="outputs/jiuming2_adversarial_v6"
sudo mkdir -p "$NEGDIR"
sudo chown -R $(whoami):$(whoami) "$NEGDIR"

for f in outputs/jiuming2_edgetts_v6/negative/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$NEGDIR/edge_$(basename $f)" 2>/dev/null || true
done
find outputs/jiuming2_piper_v6/negative/ -name "*.wav" 2>/dev/null | while read f; do
    bn=$(basename "$f")
    dir=$(basename "$(dirname "$f")")
    cp -n "$f" "$NEGDIR/piper_${dir}_${bn}" 2>/dev/null || true
done
for GPU in 0 1 2 3; do
    d="outputs/jiuming2_cosyvoice_v6/gpu${GPU}/negative"
    [ -d "$d" ] || continue
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$NEGDIR/cosy${GPU}_$(basename $f)" 2>/dev/null || true
    done
done
log "  === Adversarial neg total: $(find $NEGDIR -name '*.wav' | wc -l) ==="

# ─── Step 5: MWW 训练 ─────────────────────────────────────────
log "Step 5: MWW training..."
# 复用 run_jiuming2_v5.sh 的结构，改路径
sed 's/jiuming2_v5/jiuming2_v6/g; s/jiuming2_adv_v5/jiuming2_adv_v6/g' \
    "${HOST_DIR}/scripts/mww/run_jiuming2_v5.sh" > "${HOST_DIR}/scripts/mww/run_jiuming2_v6.sh"
chmod +x "${HOST_DIR}/scripts/mww/run_jiuming2_v6.sh"

# 更新测试集路径
sed -i 's|jiuming2_edgetts_v2|jiuming2_edgetts_v6|g' "${HOST_DIR}/scripts/mww/run_jiuming2_v6.sh"

docker run --rm --gpus all \
    --name mww_jiuming2_v6_train \
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
    "chmod +x /workspace/scripts/mww/*.sh && bash /workspace/scripts/mww/run_jiuming2_v6.sh" \
    >> "${LOG_DIR}/mww_jiuming2_v6.log" 2>&1

log "=== 救命救命 v6 complete ==="

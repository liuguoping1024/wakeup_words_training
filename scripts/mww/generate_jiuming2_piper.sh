#!/usr/bin/env bash
# Piper 生成救命救命正样本和负样本（中文 huayan 模型）
set -euo pipefail

PIPER_DIR="/workspace/work/piper-sample-generator-oww"
OUTPUT_POS="/workspace/outputs/jiuming2_piper/positive"
OUTPUT_NEG="/workspace/outputs/jiuming2_piper/negative"
MODEL="/workspace/work/piper-sample-generator-oww/models/zh_CN-huayan-medium.pt"

if [ ! -f "$MODEL" ]; then
    echo "[error] Piper 中文模型不存在: $MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_POS" "$OUTPUT_NEG"
cd "$PIPER_DIR"

echo "=== Piper 正样本 ==="
for phrase in "救命救命" "救命救命啊" "快救命救命" "久名久名" "救民救民" "揪命揪命" "九命九命" "救命救命呀"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_POS}/${safe}"
    existing=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$existing" -ge 150 ]; then
        echo "  [skip] ${phrase}: have ${existing}"
        continue
    fi
    echo "  generating: ${phrase} (target 200)"
    python3 -u generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 200 \
        --batch-size 10 2>&1 | tail -3 || echo "  [warn] failed: $phrase"
done

# 整理到根目录
for d in "$OUTPUT_POS"/*/; do
    [ -d "$d" ] || continue
    prefix=$(basename "$d")
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        mv -n "$f" "$OUTPUT_POS/piper_${prefix}_$(basename $f)" 2>/dev/null || true
    done
    rmdir "$d" 2>/dev/null || true
done

echo ""
echo "=== Piper 负样本 ==="
for phrase in "救命" "救命啊" "救火" "救火救火" "革命" "聪明" "说明" "你好" "谢谢" "打开灯"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_NEG}/${safe}"
    existing=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$existing" -ge 80 ]; then
        echo "  [skip] ${phrase}: have ${existing}"
        continue
    fi
    echo "  generating: ${phrase} (target 100)"
    python3 -u generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 100 \
        --batch-size 10 2>&1 | tail -3 || echo "  [warn] failed: $phrase"
done

for d in "$OUTPUT_NEG"/*/; do
    [ -d "$d" ] || continue
    prefix=$(basename "$d")
    for f in "$d"/*.wav; do
        [ -f "$f" ] || continue
        mv -n "$f" "$OUTPUT_NEG/piper_${prefix}_$(basename $f)" 2>/dev/null || true
    done
    rmdir "$d" 2>/dev/null || true
done

echo ""
echo "=== done ==="
echo "pos: $(find $OUTPUT_POS -name '*.wav' | wc -l)"
echo "neg: $(find $OUTPUT_NEG -name '*.wav' | wc -l)"

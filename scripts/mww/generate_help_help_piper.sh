#!/usr/bin/env bash
# 用 Piper 生成 help help 正样本和负样本
# 在 wakeword-oww Docker 内运行
set -euo pipefail

PIPER_DIR="/workspace/work/piper-sample-generator-oww"
OUTPUT_POS="/workspace/outputs/help_help_piper/positive"
OUTPUT_NEG="/workspace/outputs/help_help_piper/negative"
MODEL="/workspace/work/piper-sample-generator-oww/models/en_US-libritts_r-medium.pt"

# 如果 medium 模型不存在，用 high
if [ ! -f "$MODEL" ]; then
    MODEL="/workspace/work/piper-sample-generator-oww/models/en-us-libritts-high.pt"
fi

echo "Using model: $MODEL"

mkdir -p "$OUTPUT_POS" "$OUTPUT_NEG"

cd "$PIPER_DIR"

echo "=== Piper 正样本 ==="
for phrase in "help help" "help help please" "help help me" "somebody help help" "help help help"; do
    safe=$(echo "$phrase" | tr ' !' '_' | tr -d '.')
    outdir="${OUTPUT_POS}/${safe}"
    existing=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$existing" -ge 150 ]; then
        echo "  [skip] ${phrase}: 已有 ${existing}"
        continue
    fi
    echo "  生成: ${phrase} (目标 200)"
    python3 -u generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 200 \
        --batch-size 10 2>&1 | tail -3
done

# 把子目录的 wav 都移到 positive 根目录
echo "  整理文件..."
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
for phrase in "help me" "help us" "please help" "hello" "health" "held" "yelp" "hip hop" "good morning" "thank you"; do
    safe=$(echo "$phrase" | tr ' !' '_' | tr -d '.')
    outdir="${OUTPUT_NEG}/${safe}"
    existing=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$existing" -ge 80 ]; then
        echo "  [skip] ${phrase}: 已有 ${existing}"
        continue
    fi
    echo "  生成: ${phrase} (目标 100)"
    python3 -u generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 100 \
        --batch-size 10 2>&1 | tail -3
done

# 整理负样本
echo "  整理文件..."
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
echo "=== 完成 ==="
echo "正样本: $(find $OUTPUT_POS -name '*.wav' | wc -l)"
echo "负样本: $(find $OUTPUT_NEG -name '*.wav' | wc -l)"

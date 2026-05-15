#!/usr/bin/env bash
# help help v2 Piper 生成（修复 v1 的参数错误）
set -euo pipefail

PIPER_DIR="/workspace/work/piper-sample-generator-oww"
OUTPUT_POS="/workspace/outputs/help_help_piper_v2/positive"
OUTPUT_NEG="/workspace/outputs/help_help_piper_v2/negative"
MODEL="/workspace/work/piper-sample-generator-oww/models/en-us-libritts-high.pt"

mkdir -p "$OUTPUT_POS" "$OUTPUT_NEG"
cd "$PIPER_DIR"

echo "=== Piper positive ==="
for phrase in "help help" "help help please" "help help me" "somebody help help" "help help help" "help help help help"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_POS}/${safe}"
    mkdir -p "$outdir"
    cnt=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$cnt" -ge 280 ]; then
        echo "  [skip] $phrase: $cnt"
        continue
    fi
    echo "  gen: $phrase (300)"
    python3 generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 300 \
        --batch-size 10 2>&1 | tail -1 || echo "  [warn] failed"
done

echo ""
echo "=== Piper negative ==="
for phrase in "help me" "help us" "please help" "hello" "health" "held" "yelp" "hip hop" "good morning" "thank you" "helm" "shell" "helmet" "healthy"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_NEG}/${safe}"
    mkdir -p "$outdir"
    cnt=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$cnt" -ge 100 ]; then
        echo "  [skip] $phrase: $cnt"
        continue
    fi
    echo "  gen: $phrase (120)"
    python3 generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 120 \
        --batch-size 10 2>&1 | tail -1 || echo "  [warn] failed"
done

echo ""
echo "=== summary ==="
echo "pos: $(find $OUTPUT_POS -name '*.wav' | wc -l)"
echo "neg: $(find $OUTPUT_NEG -name '*.wav' | wc -l)"

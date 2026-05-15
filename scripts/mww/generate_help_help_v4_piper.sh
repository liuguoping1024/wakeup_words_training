#!/usr/bin/env bash
# help help v4 Piper libritts（多说话人英文）
set -euo pipefail

PIPER_DIR="/workspace/work/piper-sample-generator-oww"
OUTPUT_POS="/workspace/outputs/help_help_piper_v4/positive"
OUTPUT_NEG="/workspace/outputs/help_help_piper_v4/negative"
MODEL="/workspace/work/piper-sample-generator-oww/models/en-us-libritts-high.pt"

mkdir -p "$OUTPUT_POS" "$OUTPUT_NEG"
cd "$PIPER_DIR"

echo "=== Piper positive（严格 3 个正样本）==="
# 只生成 help help（含感叹号后 ffmpeg 会去掉，所以 3 个 phrase 实际声学只有 2 种，全部用 help help 效果一样）
for phrase in "help help"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_POS}/${safe}"
    mkdir -p "$outdir"
    cnt=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$cnt" -ge 900 ]; then
        echo "  [skip] $phrase: $cnt"
        continue
    fi
    echo "  gen: $phrase (target 1000)"
    python3 generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 1000 \
        --batch-size 10 2>&1 | tail -1 || echo "  [warn] failed"
done

echo ""
echo "=== Piper negative ==="
NEG_WORDS=(
    "help me" "help us" "helpful" "helpless" "helm" "helmet"
    "hello" "yelp" "kelp" "whelp"
    "shelf" "shell" "smell" "spell" "swell"
    "fell" "tell" "sell" "well" "yell"
    "held" "melt" "felt" "belt" "self" "myself"
    "hip hop" "tick tock" "flip flop" "knock knock" "ping pong"
    "good morning" "good night" "thank you"
    "come on" "look out" "stand up" "sit down"
    "turn on" "turn off" "hey siri"
    "fire fire" "emergency"
)
for phrase in "${NEG_WORDS[@]}"; do
    safe=$(echo "$phrase" | md5sum | head -c 8)
    outdir="${OUTPUT_NEG}/${safe}"
    mkdir -p "$outdir"
    cnt=$(find "$outdir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$cnt" -ge 80 ]; then
        echo "  [skip] $phrase: $cnt"
        continue
    fi
    echo "  gen: $phrase (target 100)"
    python3 generate_samples.py \
        "$phrase" \
        --model "$MODEL" \
        --output-dir "$outdir" \
        --max-samples 100 \
        --batch-size 10 2>&1 | tail -1 || echo "  [warn] failed: $phrase"
done

echo ""
echo "=== summary ==="
echo "pos: $(find $OUTPUT_POS -name '*.wav' | wc -l)"
echo "neg: $(find $OUTPUT_NEG -name '*.wav' | wc -l)"

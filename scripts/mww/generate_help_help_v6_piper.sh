#!/usr/bin/env bash
# help help v6 Piper libritts — 慢速 length_scales 2.0/2.5/3.0
set -euo pipefail

PIPER_DIR="/workspace/work/piper-sample-generator-oww"
OUTPUT_POS="/workspace/outputs/help_help_piper_v6/positive"
OUTPUT_NEG="/workspace/outputs/help_help_piper_v6/negative"
MODEL="/workspace/work/piper-sample-generator-oww/models/en-us-libritts-high.pt"

mkdir -p "$OUTPUT_POS" "$OUTPUT_NEG"
cd "$PIPER_DIR"

echo "=== Piper positive (length_scales 2.0 2.5 3.0)==="
cnt=$(find "$OUTPUT_POS" -name "*.wav" 2>/dev/null | wc -l)
if [ "$cnt" -lt 1800 ]; then
    echo "  gen: help help (target 2000, slow)"
    python3 generate_samples.py \
        "help help" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_POS" \
        --max-samples 2000 \
        --batch-size 10 \
        --length-scales 2.0 2.5 3.0 2>&1 | tail -3 || echo "  [warn] failed"
fi

echo ""
echo "=== Piper negative (normal speed 1.0 1.2 1.4)==="
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
        --batch-size 10 \
        --length-scales 1.0 1.2 1.4 2>&1 | tail -1 || echo "  [warn] failed: $phrase"
done

echo ""
echo "=== summary ==="
echo "pos: $(find $OUTPUT_POS -name '*.wav' | wc -l)"
echo "neg: $(find $OUTPUT_NEG -name '*.wav' | wc -l)"
echo ""
echo "=== 正样本时长抽样 ==="
for f in $(find "$OUTPUT_POS" -name "*.wav" | head -5); do
    dur=$(python3 -c "import wave; w=wave.open('$f','rb'); print(f'{w.getnframes()/w.getframerate():.2f}')")
    echo "  $(basename $f): ${dur}s"
done

#!/usr/bin/env bash
# 准备 jiuming v6 正样本
# 策略：增加真实录音权重（复制多份），保持 TTS 样本
set -euo pipefail

echo "=== 准备 jiuming v6 正样本 ==="

POSDIR="data/positive_augmented/jiuming_v6"
mkdir -p "$POSDIR"

# 1. 真实录音 × 10 份（原始100条，不做裁剪，只做峰值归一化）
echo "复制真实录音 × 10..."
for rep in $(seq 0 9); do
  for f in data/real_voices_jiuming/*.wav; do
    bn=$(basename "$f" .wav)
    out="$POSDIR/real_r${rep}_${bn}.wav"
    if [ ! -f "$out" ]; then
      sox "$f" "$out" gain -n -3 2>/dev/null
    fi
  done
done
REAL_COUNT=$(find "$POSDIR" -name "real_r*" | wc -l)
echo "  真实录音: $REAL_COUNT"

# 2. CosyVoice 正样本（正确分类的）
echo "复制 CosyVoice 正样本..."
for d in outputs/jiuming_correct/positive outputs/jiuming_correct_gpu1/positive; do
  if [ -d "$d" ]; then
    for f in "$d"/*.wav; do
      [ -f "$f" ] || continue
      bn=$(basename "$f")
      cp -n "$f" "$POSDIR/cosy_${bn}" 2>/dev/null || true
    done
  fi
done
COSY_COUNT=$(find "$POSDIR" -name "cosy_*" | wc -l)
echo "  CosyVoice: $COSY_COUNT"

# 3. CosyVoice 多说话人
echo "复制 CosyVoice 多说话人..."
if [ -d "outputs/cosyvoice_jiuming_multispk" ]; then
  for f in outputs/cosyvoice_jiuming_multispk/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/multispk_$(basename $f)" 2>/dev/null || true
  done
fi
MULTI_COUNT=$(find "$POSDIR" -name "multispk_*" | wc -l)
echo "  多说话人: $MULTI_COUNT"

# 4. Piper
echo "复制 Piper..."
if [ -d "outputs/piper_jiuming" ]; then
  for f in outputs/piper_jiuming/*.wav; do
    [ -f "$f" ] || continue
    cp -n "$f" "$POSDIR/piper_$(basename $f)" 2>/dev/null || true
  done
fi
PIPER_COUNT=$(find "$POSDIR" -name "piper_*" | wc -l)
echo "  Piper: $PIPER_COUNT"

TOTAL=$(find "$POSDIR" -name "*.wav" | wc -l)
echo ""
echo "正样本总计: $TOTAL"
echo "  真实录音占比: $(echo "scale=1; $REAL_COUNT * 100 / $TOTAL" | bc)%"

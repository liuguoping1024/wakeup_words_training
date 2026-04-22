#!/usr/bin/env bash
# 准备 MWW "你好树实" v2 训练数据
# 合并：真实录音 + CosyVoice TTS + edge-tts
set -euo pipefail

SRC_REAL="data/positive_raw/nihao_shushi"
SRC_COSYVOICE="outputs/cosyvoice_clips/positive"
SRC_EDGE="outputs/oww/nihao_shushi_v3/edge_tts_positive"
DST="data/positive_raw/nihao_shushi_v2"

echo "准备 MWW 'nihao_shushi_v2' 混合数据集"
mkdir -p "$DST"

EXISTING=$(find "$DST" -name "*.wav" 2>/dev/null | wc -l)
echo "已有: $EXISTING 条"

if [ "$EXISTING" -gt 100000 ]; then
  echo "已准备好，跳过"
  exit 0
fi

echo "复制真实录音..."
count=0
find "$SRC_REAL" -name "*.wav" | while read f; do
  name=$(basename "$f")
  cp "$f" "$DST/real_$name"
done

echo "复制 CosyVoice..."
i=0
find "$SRC_COSYVOICE" -name "*.wav" | while read f; do
  cp "$f" "$DST/cosy_$(basename $f)"
done

echo "复制 edge-tts..."
find "$SRC_EDGE" -name "*.wav" | while read f; do
  cp "$f" "$DST/edge_$(basename $f)"
done

TOTAL=$(find "$DST" -name "*.wav" | wc -l)
echo "完成: $TOTAL 条"

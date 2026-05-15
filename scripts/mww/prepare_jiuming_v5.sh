#!/usr/bin/env bash
# 准备"救命" v5 训练数据
# 改进：加入对抗性负样本（含-ming韵母、jiu-声母的词）
set -euo pipefail

echo "=== 准备 v5 对抗性负样本 ==="

# 合并 4 GPU 的 CosyVoice 对抗性负样本
ADVERSARIAL_DIR="outputs/adversarial_merged"
mkdir -p "$ADVERSARIAL_DIR"

echo "合并 CosyVoice 对抗性负样本..."
for GPU in 0 1 2 3; do
    SRC="outputs/adversarial_negatives/gpu${GPU}"
    for subdir in ming jiu daily; do
        if [ -d "$SRC/$subdir" ]; then
            for f in "$SRC/$subdir"/*.wav; do
                [ -f "$f" ] || continue
                cp -n "$f" "$ADVERSARIAL_DIR/cosy_gpu${GPU}_${subdir}_$(basename $f)"
            done
        fi
    done
done
COSY_COUNT=$(find "$ADVERSARIAL_DIR" -name "cosy_*.wav" | wc -l)
echo "  CosyVoice 对抗性: $COSY_COUNT"

# 合并 edge-tts 对抗性负样本
EDGETTS_DIR="outputs/adversarial_edgetts"
if [ -d "$EDGETTS_DIR" ]; then
    for f in "$EDGETTS_DIR"/*.wav; do
        [ -f "$f" ] || continue
        cp -n "$f" "$ADVERSARIAL_DIR/edge_$(basename $f)"
    done
fi
EDGE_COUNT=$(find "$ADVERSARIAL_DIR" -name "edge_*.wav" | wc -l)
echo "  edge-tts 对抗性: $EDGE_COUNT"

TOTAL=$(find "$ADVERSARIAL_DIR" -name "*.wav" | wc -l)
echo "  对抗性负样本总计: $TOTAL"

echo ""
echo "=== 正样本（复用 v4）==="
V4_POS="data/positive_augmented/jiuming_v4"
V4_COUNT=$(find "$V4_POS" -name "*.wav" 2>/dev/null | wc -l)
echo "  v4 正样本: $V4_COUNT"

echo ""
echo "=== 完成 ==="
echo "对抗性负样本: $ADVERSARIAL_DIR ($TOTAL 条)"
echo "正样本: $V4_POS ($V4_COUNT 条)"
echo ""
echo "下一步: 在 Docker 内生成对抗性负样本的 mmap 特征，然后训练"

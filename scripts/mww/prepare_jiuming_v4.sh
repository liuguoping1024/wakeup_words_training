#!/usr/bin/env bash
# 准备"救命" v4 训练数据
# 正样本：真实录音 + CosyVoice 多说话人变体 + Piper
# 负样本：AISHELL-1 中文语音 + CosyVoice 无关中文 + MUSAN + RIRS
set -euo pipefail

echo "=== 准备正样本 ==="
POSDIR="data/positive_raw/jiuming_v4"
sudo mkdir -p "$POSDIR"
sudo chown -R $(whoami):$(whoami) "$POSDIR"

# 真实录音
echo "复制真实录音..."
for f in data/positive_raw/jiuming/*.wav; do
  cp "$f" "$POSDIR/real_$(basename $f)"
done
echo "  真实: $(find $POSDIR -name 'real_*' | wc -l)"

# CosyVoice 多说话人（正确分类的正样本）
echo "复制 CosyVoice 正样本..."
for f in outputs/jiuming_correct/positive/*.wav; do
  cp "$f" "$POSDIR/cosy0_$(basename $f)"
done
for f in outputs/jiuming_correct_gpu1/positive/*.wav; do
  cp "$f" "$POSDIR/cosy1_$(basename $f)"
done
echo "  CosyVoice: $(find $POSDIR -name 'cosy*' | wc -l)"

# CosyVoice 多说话人参考音频生成的
echo "复制 CosyVoice 多说话人..."
for f in outputs/cosyvoice_jiuming_multispk/*.wav; do
  cp "$f" "$POSDIR/multispk_$(basename $f)"
done
echo "  多说话人: $(find $POSDIR -name 'multispk_*' | wc -l)"

# Piper
echo "复制 Piper..."
for f in outputs/piper_jiuming/*.wav; do
  cp "$f" "$POSDIR/piper_$(basename $f)"
done
echo "  Piper: $(find $POSDIR -name 'piper_*' | wc -l)"

TOTAL=$(find "$POSDIR" -name "*.wav" | wc -l)
echo "正样本总计: $TOTAL"

echo ""
echo "=== 准备 augmented 目录 ==="
AUGDIR="data/positive_augmented/jiuming_v4"
sudo mkdir -p "$AUGDIR"
sudo chown -R $(whoami):$(whoami) "$AUGDIR"

# 随机抽取（如果总数 > 15000 就抽样，否则全用）
if [ "$TOTAL" -gt 15000 ]; then
  echo "抽取 15000 条..."
  find "$POSDIR" -name "*.wav" | shuf -n 15000 | while read f; do
    cp "$f" "$AUGDIR/$(basename $f)"
  done
else
  echo "全部复制..."
  find "$POSDIR" -name "*.wav" | while read f; do
    cp "$f" "$AUGDIR/$(basename $f)"
  done
fi
echo "augmented: $(find $AUGDIR -name '*.wav' | wc -l)"

echo ""
echo "=== 完成 ==="

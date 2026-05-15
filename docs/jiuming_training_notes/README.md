# 救命 (jiuming) 唤醒词训练记录

## 模型版本对比

使用正确推理管线 (`inference/runtime.py`, 维护模型状态) 的评估结果。

### cutoff=0.88, window=3

| 版本 | 100条真实录音 | 12条ARM测试 | TTS误触发率 | 说明 |
|------|-------------|------------|------------|------|
| v3 | — | 66.7% (8/12) | — | 无中文负样本 |
| v4 | 57.0% (57/100) | 91.7% (11/12) | 33.3% | + AISHELL-1 中文负样本 (weight=15/2.0) |
| v5 | 25.0% (25/100) | 16.7% (2/12) | 8.6% | + 对抗性负样本 (weight=20/3.0) → 过拟合 |
| **v6** | **90.0% (90/100)** | **83.3% (10/12)** | **9.9%** | 真实录音×10 + 对抗性负样本 (weight=8/1.5) |

### 关键发现

1. **推理管线必须维护模型状态** — MWW 是流式模型，`_state_ins`/`_state_outs` 必须在帧间传递。之前的评估脚本忽略了状态，给出虚假的高 recall。
2. **真实录音权重很重要** — v4 只有 100 条真实录音被 12000 条 TTS 淹没，v6 复制 10 份后 recall 从 57% 升到 90%。
3. **对抗性负样本权重要适度** — v5 的 weight=20/3.0 太激进导致 recall 崩溃，v6 降到 8/1.5 效果好。
4. **音量归一化/静音裁剪对 MWW 流式模型无帮助** — 模型需要前面的静音来预热状态。

## 数据组成

### v6 正样本 (13000条)
- 真实录音 × 10: 1000 条 (峰值归一化到 -3dB)
- CosyVoice 正确分类: 5000 条 (救命/救命啊/救民/久名 等)
- CosyVoice 多说话人: 5000 条 (40 说话人 voice cloning)
- Piper huayan: 2000 条

### v6 负样本
- AISHELL-1 中文: ~30k (sampling_weight=10, penalty=1.5)
- 对抗性负样本: 6490 条 (sampling_weight=8, penalty=1.5)
  - 含 -ming 韵母: 3000 (说明/革命/聪明/生命...)
  - 含 jiu- 声母: 1500 (救火/救人/九月/就是...)
  - 日常中文: 1500
- 英文 speech/dinner_party/no_speech: 原有数据集

### 训练参数
- steps: 25000
- batch_size: 128
- learning_rate: 0.001
- model: mixednet 64,64,64,64 / stride=3

## 后续调校方向

- 录制更多真实录音（不同人、不同距离、不同音量）
- 调整 cutoff (建议 0.75-0.85) 和 window (建议 2-3)
- 对抗性负样本中 "聪明" 仍有误触发，可针对性增加
- 考虑增加训练步数到 30000-40000

## 文件位置

- 模型: `outputs/jiuming_v6.tflite`, `outputs/jiuming_v6.json`
- 训练日志: `logs/mww_jiuming_v6.log`
- 训练配置: `scripts/mww/06_write_training_config_v6.py`
- 训练脚本: `scripts/mww/run_jiuming_v6.sh`
- 正样本: `data/positive_augmented/jiuming_v6/`
- 对抗性负样本: `outputs/adversarial_merged/`, `outputs/adversarial_edgetts/`
- 验证脚本 (正确推理): `scripts/mww/infer_verify_v3.py`
- 预处理脚本: `scripts/mww/preprocess_real_voices.py`
- 对抗性生成: `scripts/mww/generate_adversarial_negatives.py`

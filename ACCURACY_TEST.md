# 离线准确度测试说明

本文档说明本项目里我是如何做唤醒词离线准确度测试的，便于复现。

## 1. 测试目标

- 评估模型在真实录音（`data/real_voices/*.wav`）上的唤醒词检出率（召回）。
- 每个 wav 只要触发次数 `> 0`，就记为该条样本“检测到”。

## 2. 测试环境

- 设备：Linux aarch64
- 模型：`inference/help_me.tflite`
- 测试集：`data/real_voices/real_0050.wav` ~ `real_0099.wav`（50 条）
- 推理脚本：`inference/detect.py`

## 3. 评估命令

先进入推理目录并激活虚拟环境：

```bash
cd /root/wakeup_words_training/inference
source .venv/bin/activate
```

单次评估（示例参数）：

```bash
python3 detect.py \
  --model help_me.tflite \
  --wav-dir /root/wakeup_words_training/data/real_voices \
  --cutoff 0.10 \
  --window 5
```

脚本末尾会输出：

```text
结果: X/50 检测到  (Y%)
```

其中 `Y%` 就是“每条样本至少触发一次”的检出率。

## 4. 我这次跑出来的结果

- `cutoff=0.10, window=5` -> `18/50 (36.0%)`
- `cutoff=0.08, window=5` -> `18/50 (36.0%)`
- `cutoff=0.06, window=5` -> `19/50 (38.0%)`

说明：当前模型在这批 Rapoo 真实录音上的召回率偏低，单纯降低阈值改善有限。

## 5. 注意事项

- `detect.py` 的默认参数是 `cutoff=0.10`、`window=5`。
- 但 `help_me.json` 里写的是：
  - `probability_cutoff = 0.97`
  - `sliding_window_size = 5`
- 因此测试时请明确你想对齐哪个阈值口径（命令行参数 vs json 配置）。

## 6. 建议的下一步验证

- 用 `help_me.json` 的阈值再跑一轮离线测试。
- 增加负样本目录，统计误触发率（FAR）。
- 做 `cutoff x window` 网格测试，找最优运行点。


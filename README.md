# Wake Word Training

训练自定义唤醒词模型，支持两个框架：
- **MWW** (micro-wake-word) — TensorFlow，输出量化 TFLite，适合 ESPHome / Home Assistant 边缘设备
- **OWW** (openWakeWord) — PyTorch，输出 ONNX，适合通用 Linux 设备

两个唤醒词：
- **help me** — 英文，TTS 合成正样本
- **你好树实** — 中文，40 位说话人真实录音（99,454 条）

## 环境

| 项目 | 配置 |
|------|------|
| GPU | GTX 1080 Ti × 4 (sm_61, 11GB) |
| Driver | 545.23, CUDA 12.3 |
| OS | Ubuntu 20.04 |
| Docker | NVIDIA Container Toolkit |

全部训练在 Docker 容器内执行，无需宿主机 Python 环境。

## 快速开始

```bash
# 构建镜像
make build-all

# 训练 MWW help_me
make mww-help-me

# 训练 MWW 你好树实
make mww-nihao-shushi

# 训练 OWW help_me
make oww-help-me

# 评估
make eval-all

# 全部训练
make train-all
```

## 项目结构

```
├── docker/
│   ├── mww.Dockerfile          # TF 2.16.2 + cuDNN 8
│   └── oww.Dockerfile          # PyTorch 2.5.1 + CUDA 12.1
├── scripts/
│   ├── mww/                    # MWW 流水线脚本
│   │   ├── run_pipeline_tts.sh     # help_me (TTS)
│   │   ├── run_pipeline_real.sh    # 你好树实 (真实语音)
│   │   └── ...
│   ├── oww/                    # OWW 训练脚本
│   │   ├── train_oww.py
│   │   └── train_nihao_oww.py
│   └── eval_model.py           # 通用评估脚本
├── data/                       # 训练数据 (gitignore)
├── work/                       # 源码仓库 (gitignore)
│   ├── micro-wake-word/
│   ├── piper-sample-generator/
│   ├── piper-sample-generator-oww/
│   └── openWakeWord/
├── outputs/                    # 模型输出
│   ├── help_me.tflite + .json
│   ├── nihao_shushi.tflite + .json
│   └── oww/
│       ├── help_me.onnx
│       └── nihao_shushi.onnx
├── inference/                  # 推理运行时
├── logs/                       # 训练日志
└── docs/                       # 文档
```

## 最新训练结果 (2026-04-11)

### MWW 模型

| 模型 | 召回率 | 平均概率 | 测试样本 |
|------|--------|---------|---------|
| help_me | **98.0%** (49/50) | 0.891 | 50 条真实录音 |
| nihao_shushi | **99.5%** (199/200) | 0.792 | 200 条切分录音 |

### OWW 模型

| 模型 | Accuracy | Recall | FP/Hour |
|------|----------|--------|---------|
| help_me | 70.2% | 40.6% | 1.15 |
| nihao_shushi | N/A | N/A | 中文不适配 OWW 英文 pipeline |

## Docker 镜像

| 镜像 | 基础 | 用途 |
|------|------|------|
| wakeword-mww | cuda:12.2.2-cudnn8 + TF 2.16.2 | MWW 训练 |
| wakeword-oww | cuda:12.2.2-cudnn8 + PyTorch 2.5.1+cu121 | OWW 训练 |

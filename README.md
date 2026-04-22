# Wake Word Training

训练自定义唤醒词模型，支持两个框架：
- **MWW** (micro-wake-word) — TensorFlow，输出量化 TFLite，适合 ESPHome / Home Assistant 边缘设备
- **OWW** (openWakeWord) — PyTorch，输出 ONNX，适合通用 Linux / Android 设备

三个唤醒词：
- **help me** — 英文，TTS 合成正样本
- **你好树实** — 中文，40 位说话人真实录音（99,454 条）+ TTS 增强
- **救命** — 中文，真实录音 + 多说话人 CosyVoice2 TTS + Piper TTS

## 环境

| 项目 | 配置 |
|------|------|
| GPU | GTX 1080 Ti × 4 (sm_61, 11GB) |
| Driver | 545.23, CUDA 12.3 |
| OS | Ubuntu 20.04 |
| Docker | NVIDIA Container Toolkit |

全部训练在 Docker 容器内执行，无需宿主机 Python 环境。

## 中文 TTS 引擎

三个离线/在线 TTS 引擎用于生成中文合成语音训练数据：

| 引擎 | 类型 | 声音数 | 速度 | 用途 |
|------|------|--------|------|------|
| CosyVoice2 (0.5B) | 离线 GPU | 多声音（voice cloning） | ~18 条/分钟/GPU | 多说话人正样本 + 对抗性负样本 |
| Piper huayan (zh_CN) | 离线 GPU | 1 声音 | ~75 条/分钟/GPU | 快速批量正样本 |
| edge-tts (Microsoft) | 在线 | 19 种中文声音 | ~6 条/分钟 | 高质量多声音正样本 |

支持 4 GPU × 3 进程/GPU = 12 进程并行 TTS 生成。

## 快速开始

```bash
# 构建镜像
make build-all

# 训练 MWW help_me
make mww-help-me

# 训练 MWW 你好树实（真实录音）
make mww-nihao-shushi

# 训练 MWW 救命（真实录音）
make mww-jiuming

# 评估
make eval-all

# 全部训练
make train-all
```

## 项目结构

```
├── docker/
│   ├── mww.Dockerfile          # TF 2.16.2 + cuDNN 8
│   ├── oww.Dockerfile          # PyTorch 2.5.1 + CUDA 12.1
│   └── cosyvoice.Dockerfile    # CosyVoice2 TTS 生成
├── scripts/
│   ├── mww/                    # MWW 流水线脚本
│   │   ├── run_pipeline_tts.sh     # help_me (TTS)
│   │   ├── run_pipeline_real.sh    # 中文唤醒词 (真实语音 + TTS)
│   │   └── ...
│   ├── oww/                    # OWW 训练 + TTS 生成脚本
│   │   ├── generate_cosyvoice_multispeaker.py  # 多说话人 CosyVoice 生成
│   │   ├── generate_all_zh_clips.py            # edge-tts 中文生成
│   │   ├── eval_oww_model.py                   # OWW 模型评估
│   │   ├── train_zh_v4.py                      # 全中文数据训练
│   │   └── ...
│   └── eval_model.py           # MWW 通用评估脚本
├── data/                       # 训练数据 (gitignore)
│   ├── positive_raw/           # 正样本原始录音
│   ├── speaker_refs/           # 说话人参考音频（CosyVoice voice cloning 用）
│   └── augmentation/           # 增强数据（RIR、背景噪声）
├── work/                       # 源码仓库 (gitignore)
│   ├── micro-wake-word/
│   ├── piper-sample-generator-oww/
│   ├── openWakeWord/
│   └── CosyVoice/
├── outputs/                    # 模型输出 (gitignore)
├── inference/                  # 推理运行时
├── logs/                       # 训练日志 (gitignore)
└── docs/                       # 文档
```


## 最新训练结果 (2026-04-21)

### MWW 模型（推荐）

| 模型 | 版本 | 正样本来源 | 召回率 | 平均概率 | 测试样本 |
|------|------|-----------|--------|---------|---------|
| help_me | v1 | TTS 合成 | **98.0%** | 0.891 | 50 条真实录音 |
| nihao_shushi | v1 | 真实录音 99k | **99.6%** | 0.788 | 500 条真实录音 |
| nihao_shushi | v2 | 真实 99k + CosyVoice 5k + edge-tts 1.2k | **97.6%** | 0.905 | 500 条真实录音 |
| jiuming | v1 | 真实录音 1k（单人） | **100%** | 0.974 | 100 条真实录音 |
| jiuming | v3 | 真实 1k + 多说话人 CosyVoice 5k + Piper 2k | **96.0%** | 0.806 | 100 条真实录音 |

**模型选择建议：**
- 你好树实：v1（99.6%）适合已知说话人群体，v2（97.6%）泛化更好
- 救命：v1（100%）仅识别训练者声音，v3（96.0%）支持多说话人

### OWW 模型（实验性）

| 模型 | 版本 | 召回率 | 误触发率 | 备注 |
|------|------|--------|---------|------|
| help_me | v1 | 70.2% | — | 英文 TTS |
| nihao_shushi | v4 | 100% | 44.8% | 全中文数据，penalty=100 |
| nihao_shushi | v5 | 79.6% | 0.6% | +ACAV100M，penalty=500 |
| nihao_shushi | v6 | 91.6% | 6.6% | penalty=250，最佳平衡 |

### 模型格式

| 框架 | 格式 | 输入 shape | 特征 | 适用场景 |
|------|------|-----------|------|---------|
| MWW | TFLite (int8) | [1, 3, 40] | 40-ch mel filterbank, PCAN, 30ms window, 10ms step, stride=3 | ESP32, 边缘设备 |
| OWW | ONNX (float32) | [1, 16, 96] | Google speech_embedding | Linux, Android, PC |

## Docker 镜像

| 镜像 | 大小 | 用途 |
|------|------|------|
| wakeword-mww | ~15.7GB | MWW 训练（TF 2.16.2 + cuDNN 8） |
| wakeword-oww | ~16.5GB | OWW 训练（PyTorch 2.5.1+cu121） |
| cosyvoice | ~16.6GB | CosyVoice2 中文 TTS 生成 |

## 多说话人 TTS 生成（方案 5）

从"你好树实"99k 真实录音中提取 40 个说话人参考音频，用 CosyVoice2 voice cloning 生成目标唤醒词的多说话人版本：

```bash
# 提取参考音频
python3 scripts/extract_speaker_refs.py

# 4 GPU × 3 进程并行生成
# 见 scripts/oww/generate_cosyvoice_multispeaker.py
```

这种方法解决了"只有少量真实录音"的问题，通过 voice cloning 将已有说话人的声音特征迁移到新唤醒词上。

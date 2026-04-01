# 「你好树实」唤醒词训练 Wiki

本文档说明如何使用**真实人声录音**训练唤醒词 **「你好树实」**（输出前缀 `nihao_shushi`），对应 `scripts-real/` 流水线。

---

## 1. 主要原理

与 `help me` 训练流程的核心框架一致（microWakeWord + TensorFlow + MixedNet），区别在于：

- **正样本来源**：不使用 TTS 合成，而是从 **40 位说话人** 的真实录音中切分唤醒词片段
- **数据规模**：每位说话人在多个平台（智能音箱、Android、iOS、PC）上录制约 12 分钟长音频，共 823 个 WAV 文件，切分后得到 **~99,000 条** 唤醒词样本
- **增强策略**：原始样本数量充足时跳过增强；不足时通过增益、时间拉伸、白噪声、简单混响扩充到目标数量

---

## 2. 数据准备

### 2.1 录音目录结构

将录音文件放入 `data/sounds/`，按如下结构组织：

```
data/sounds/
├── 001weishu/
│   ├── Android/          # 手机录音
│   │   └── 001weishu.wav
│   ├── Apple/            # 苹果设备录音
│   │   └── 001weishu.wav
│   ├── IOS/              # iOS 录音
│   │   └── 001weishu.wav
│   └── Smartspeaker/     # 智能音箱录音（多距离多通道）
│       ├── 001weishu-27-female-1m.wav
│       ├── 001weishu-27-female-1moutput1.wav
│       ├── 001weishu-27-female-1moutput2.wav
│       ├── 001weishu-27-female-3m.wav
│       └── ...
├── 002jiakai/
│   └── ...
└── 040sunzong/
    └── ...
```

**注意**：平台子目录名不需要严格统一，脚本会自动扫描每个说话人下的所有子目录。支持 `Smartspeaker`、`SmartSpeaker`、`smartspeaker`、`WIN10`、`win10` 等任意命名。

### 2.2 录音要求

- 每个 WAV 文件约 **10-12 分钟**，包含 **~100-150 次** 重复说出「你好树实」
- 采样率不限，脚本会自动重采样到 **16 kHz mono**
- 智能音箱录音包含不同距离（1m/3m/5m/7m/9m/11m）和多通道（output1/output2）

---

## 3. 流水线步骤

整体由 `scripts-real/run_pipeline.sh` 串起，`Makefile` 的 `train-real` 目标执行。

### 3.1 阶段概览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 0 | `scripts/01_prepare_repos.sh` + `00_patch_mww.sh` | 准备 micro-wake-word 仓库和环境 |
| 1 | `scripts/02_download_datasets.sh` | 下载增强数据和负样本 |
| 2 | `scripts/03_prepare_audio.py` | 准备增强音频（MIT RIR / AudioSet / FMA → 16kHz） |
| 3 | `scripts-real/01_split_real_voices.py` | 基于能量 VAD 切分真实语音为单条样本 |
| 4 | `scripts-real/02_augment_positives.py` | 数据增强扩充（如原始样本已足够则跳过） |
| 5 | `scripts-real/03_generate_features.py` | 生成频谱特征（RaggedMmap） |
| 6 | `scripts-real/05_train_and_export.sh` | 训练 MixedNet 模型并导出 TFLite |

### 3.2 智能跳过

流水线会自动跳过已完成的步骤：

- **切分**：若 `data/positive_raw/nihao_shushi/` 已有 WAV 文件，跳过
- **增强**：若 `data/positive_augmented/nihao_shushi/` 已有足够样本，跳过
- **特征**：每次重新生成（先清理旧特征目录）

---

## 4. 操作命令

### 4.1 完整训练（一键）

```bash
make train-real
```

首次运行约需 **数小时**（切分 ~1 小时，复制样本 ~50 分钟，特征生成 ~2 小时，训练 ~2 小时）。
再次运行会跳过切分和增强，主要时间在特征生成和训练。

### 4.2 仅切分语音（不训练）

```bash
make split-real
```

用于检查切分效果，结果在 `data/positive_raw/nihao_shushi/`。

### 4.3 评估模型

```bash
make eval-real
```

从切分的正样本中测试召回率，默认 `cutoff=0.10`、`window=3`。

### 4.4 调整参数

```bash
# 增加训练步数
make train-real REAL_TRAIN_STEPS=20000

# 调整目标正样本数
make train-real REAL_TARGET_POSITIVE=8000

# 同时调整多个参数
make train-real REAL_TRAIN_STEPS=20000 REAL_TARGET_POSITIVE=8000
```

### 4.5 强制重新训练（保留切分数据）

```bash
rm -rf data/generated_augmented_features work/micro-wake-word/trained_models/nihao_shushi
make train-real
```

### 4.6 完全从头开始

```bash
rm -rf data/positive_raw/nihao_shushi data/positive_augmented/nihao_shushi data/generated_augmented_features
make train-real
```

---

## 5. 可配置参数

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `REAL_KEYWORD_PHRASE` | `你好树实` | 唤醒词文本 |
| `REAL_KEYWORD_ID` | `nihao_shushi` | 输出文件名前缀 |
| `REAL_TRAIN_STEPS` | `15000` | 训练步数 |
| `REAL_TARGET_POSITIVE` | `5000` | 目标正样本数（原始样本不足时才增强） |

---

## 6. 训练产物

| 文件 | 说明 |
|------|------|
| `outputs/nihao_shushi.tflite` | 量化流式 TFLite 模型（~60KB） |
| `outputs/nihao_shushi.json` | ESPHome 配置（cutoff=0.97, window=5） |
| `work/micro-wake-word/trained_models/nihao_shushi/` | 训练 checkpoint、权重、日志 |

---

## 7. 本次训练记录

- **日期**：2026-03-31
- **数据**：40 位说话人，823 个 WAV 文件，切分得到 99,454 条唤醒词样本
- **平台覆盖**：智能音箱（多距离多通道）、Android、iOS/Apple、PC（Win10）
- **训练配置**：15,000 步，batch_size=128，MixedNet 架构
- **模型大小**：60,960 bytes（TFLite 量化）

---

## 8. 与 help me 训练的区别

| 对比项 | help me | 你好树实 |
|--------|---------|----------|
| 正样本来源 | TTS 合成（Piper / espeak-ng） | 真实人声录音 |
| 说话人数量 | TTS 音色数（~2-5） | 40 位真人 |
| 正样本数量 | ~800 条 | ~99,000 条 |
| 录音平台 | 无 | 智能音箱/手机/PC 多平台 |
| 脚本目录 | `scripts/` | `scripts-real/` |
| Makefile 目标 | `make train` | `make train-real` |

---

## 相关文件

| 路径 | 说明 |
|------|------|
| `scripts-real/` | 真实语音训练流水线脚本 |
| `scripts-real/common.sh` | 公共变量和工具函数 |
| `Makefile` | `train-real` / `split-real` / `eval-real` 目标 |
| `docs/WIKI_HELP_ME.md` | help me 唤醒词训练文档 |

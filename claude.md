# 项目总结：「help me」唤醒词训练系统

## 项目目标

为 ThirdReality 的智能家居设备训练一个自定义唤醒词 **「help me」**，输出量化 TFLite 模型，部署到 ESPHome / Home Assistant 生态的边缘设备上。

---

## 硬件与环境约束

| 项目 | 实际情况 | 带来的问题与应对 |
|------|---------|----------------|
| GPU | GTX 1080 Ti × 4（sm_61，11GB VRAM） | 算力足够，但属于较老架构 |
| OS | Ubuntu 20.04.6 LTS | 系统自带 Python 3.8，无法直接跑新版 TF；通过 Docker 容器（Ubuntu 22.04）绕过 |
| CUDA | 驱动 545.23，CUDA 12.3 | 需要 cuDNN 8.x 配合 TF 2.16；通过 pip 安装 `nvidia-cudnn-cu12==8.9.7.29` 解决 |
| Docker 基础镜像 | `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` | 提供 `ptxas` 编译器和 CUDA 工具链，容器内环境与宿主机 GPU 驱动兼容 |
| 推理设备 | Orange Pi 5B（aarch64，无 GPU） | 需要纯 CPU 的轻量 TFLite 推理运行时 |
| 麦克风 | USB PnP Sound Device（Rapoo），通过 Home Assistant 的 PulseAudio socket 访问 | 录音质量受限，真实样本召回率偏低 |

### 宿主机 Python 版本情况

宿主机 `/usr/bin/` 下安装了多个 Python 版本：

| 版本 | 路径 | 角色 |
|------|------|------|
| Python 2.7.18 | `/usr/bin/python2.7`（`python` / `python2` 均指向此） | 系统遗留，本项目不使用 |
| Python 3.8.10 | `/usr/bin/python3.8`（`python3` 默认指向此） | Ubuntu 20.04 系统自带，版本过低，不满足 TF 2.16 要求 |
| Python 3.9.22 | `/usr/bin/python3.9` | 后装版本，未在本项目中使用 |
| Python 3.10.17 | `/usr/bin/python3.10` | **训练实际使用的版本** — Docker 容器内 Ubuntu 22.04 自带 Python 3.10，训练 venv（`work/micro-wake-word/.venv/bin/python3.10`）即基于此版本创建 |
| Python 3.12.10 | `/usr/bin/python3.12` | 后装版本，未在本项目中使用（TF 2.16 对 3.12 兼容性未验证） |

关键决策：虽然宿主机有 Python 3.10 和 3.12，但训练流水线**全部在 Docker 容器内执行**，使用容器内 Ubuntu 22.04 提供的 Python 3.10。这样做的原因是：
1. 宿主机 `python3` 默认指向 3.8，直接改系统符号链接可能破坏 apt 等系统工具
2. Docker 容器提供了干净隔离的 Python 3.10 + CUDA 工具链环境
3. 容器内 venv 路径为 `work/micro-wake-word/.venv/`，通过卷挂载持久化到宿主机，二次运行无需重建

---

## 已完成的工作（按 git 历史时间线）

### 第一阶段：核心训练流水线（b41d29e，3月26日）

用 Cursor 辅助搭建了完整的端到端训练流水线，全部容器化：

1. **Docker 化构建**：`Dockerfile` + `Makefile`，一键 `make build && make train`
2. **7 步流水线脚本**（`scripts/run_pipeline.sh` 串联）：
   - `01_prepare_repos.sh`：克隆 micro-wake-word 和 piper-sample-generator，创建 venv，安装所有依赖
   - `00_patch_mww.sh`：**关键 patch** — TF 2.16 下 `model.evaluate()` 返回的是 `np.ndarray` 而非 Tensor，`.numpy()` 会报错，用 `sed` 把 `result["fp"].numpy()` 替换为 `np.asarray(result["fp"])`
   - `02_download_datasets.sh`：下载增强数据（AudioSet、FMA、MIT RIR）和预生成负样本特征 ZIP，带多 URL 回退和损坏检测重下逻辑
   - `03_prepare_audio.py`：用 `soundfile` + `resampy` 重采样到 16kHz，**刻意避开 `torchcodec` 依赖**（该库在此环境下安装困难）
   - `04_generate_positive_samples.sh`：优先用 Piper TTS 生成正样本，失败则回退到 `espeak-ng`（变速/变调增加多样性）
   - `05_generate_features.py`：数据增强（EQ、失真、变调、带阻、彩色噪声、背景噪声、混响）+ 生成 RaggedMmap 频谱特征
   - `07_train_and_export.sh`：生成训练配置 YAML → 训练 MixedNet → 量化导出 `stream_state_internal_quant.tflite` + ESPHome JSON 清单
3. **推理运行时**（`inference/`）：
   - `runtime.py`：流式检测器，自动选择 `ai-edge-litert` / `tflite-runtime` / `tensorflow` 后端
   - `detect.py`：支持单文件测试、目录批量测试、实时麦克风检测（自动连接 HA PulseAudio socket）
   - `setup_venv.sh`：在边缘设备上一键搭建推理环境

### 第二阶段：依赖兼容性修复

- **numpy 版本冲突**：`numpy-minmax` 和 `numpy-rms` 声明需要 `numpy>=2`，但 TF 2.16 需要 `numpy==1.26.4`，通过 `--no-deps` 安装绕过
- **datasets 版本锁定**：固定 `datasets==2.19.2`，避免新版 API 变更导致 AudioSet 下载失败
- **Makefile 维护命令**：`fix-datasets` 和 `fix-tf` 用于在已有容器工作区内修正依赖版本，不用重跑全流程

### 第三阶段：真实人声与评估体系（ed398b1 → 97f854c，3月26-27日）

1. **`prepare_real_voices.py`**：真实录音预处理 — 静音裁剪、16kHz 统一、数据增强扩充（增益、时间拉伸、白噪声、简单混响），从少量原始录音扩充到目标数量
2. **`eval_model.py`**：批量评估脚本 — 正样本召回率 + 可选负样本误触发率，输出概率分布统计
3. **`runtime.py` 重大修复**：
   - 新增纯 Python `AudioFrontend` 作为 `pymicro_features` 的回退方案（aarch64 上 pymicro 可能编译失败）
   - 修复 int8 量化公式（`quantized = real / scale + zero_point`），之前公式有误导致概率输出不正确
   - 对齐训练参数：30ms 窗口、10ms 步长、40 维 mel、PCAN 自动增益
4. **`collect_samples.py`**：通过 `parec` + PulseAudio socket 在 HA 设备上交互式采集真实人声，带音量检测和自动重录
5. **`test_audio.sh`**：录音回放测试脚本，验证麦克风链路是否正常
6. **`test_real_voices.sh`**：一键跑真实人声评估的 wrapper 脚本
7. **`02_download_datasets.sh` 增强**：增加 ZIP 提取缓存标记（sha256 签名），避免重复解压

---

## 关键 Patch 与 Workaround 汇总

| 问题 | 原因 | 解决方式 | 涉及文件 |
|------|------|---------|---------|
| TF 2.16 `.numpy()` 报错 | `model.evaluate()` 在 TF 2.16 返回 ndarray 而非 Tensor | `sed` 替换为 `np.asarray()` | `00_patch_mww.sh` |
| cuDNN 版本不匹配 | TF 2.16 pip wheel 需要 cuDNN 8.9.x | pip 安装 `nvidia-cudnn-cu12==8.9.7.29` | `01_prepare_repos.sh` |
| numpy 2.x 依赖冲突 | `numpy-minmax`/`numpy-rms` 要求 numpy≥2，TF 2.16 要求 <2 | `pip install --no-deps` | `01_prepare_repos.sh` |
| torchcodec 安装失败 | 该库在此环境下编译困难 | 用 `soundfile` + `resampy` 替代，完全绕开 | `03_prepare_audio.py` |
| AudioSet 下载 404 | HuggingFace 数据集转 parquet 后旧 tar 链接失效 | 多 URL 回退 + HF datasets streaming API 兜底 | `02_download_datasets.sh`, `03_prepare_audio.py` |
| pymicro_features 编译失败（aarch64） | C++ 编译环境不完整 | 纯 Python `AudioFrontend` 回退 | `inference/runtime.py` |
| Ubuntu 20.04 Python 版本过低 | 宿主机默认 `python3` 指向 3.8，不满足 TF 2.16 要求（宿主机虽有 3.9/3.10/3.12，但不改系统默认） | Docker 容器内使用 Ubuntu 22.04 自带的 Python 3.10，venv 通过卷挂载持久化 | `Dockerfile`, `01_prepare_repos.sh` |
| ZIP 重复解压浪费时间 | 每次跑流水线都重新解压负样本 | sha256 签名标记文件，跳过已解压的 | `02_download_datasets.sh` |

---

## 当前模型表现

在 50 条 Rapoo USB 麦克风真实录音上的离线测试结果：

| cutoff | window | 召回率 |
|--------|--------|--------|
| 0.10 | 5 | 18/50 (36.0%) |
| 0.08 | 5 | 18/50 (36.0%) |
| 0.06 | 5 | 19/50 (38.0%) |

召回率偏低，单纯降低阈值改善有限。主要瓶颈在于：
- TTS 合成的正样本与真实人声分布差距大
- Rapoo USB 麦克风录音质量有限
- 真实样本数量不足（仅 50 条用于测试，训练时混入的也有限）

---

## 项目文件结构

```
├── Dockerfile                    # CUDA 12.2 + cuDNN 8 + Ubuntu 22.04 基础镜像
├── Makefile                      # build / train / fix-datasets / fix-tf
├── README.md                     # 快速开始文档
├── ACCURACY_TEST.md              # 离线准确度测试记录
├── docs/WIKI_HELP_ME.md          # 详细 Wiki 文档
├── collect_samples.py            # HA 设备上的人声采集工具
├── test_audio.sh                 # 麦克风录音回放测试
├── scripts/
│   ├── run_pipeline.sh           # 流水线入口
│   ├── common.sh                 # 公共变量与工具函数
│   ├── 00_patch_mww.sh           # TF 2.16 兼容性 patch
│   ├── 01_prepare_repos.sh       # 克隆仓库 + 安装依赖
│   ├── 02_download_datasets.sh   # 下载增强/负样本数据
│   ├── 03_prepare_audio.py       # 音频重采样到 16kHz
│   ├── 04_generate_positive_samples.sh  # TTS 正样本生成
│   ├── 05_generate_features.py   # 数据增强 + 频谱特征生成
│   ├── 06_write_training_config.py  # 训练配置 YAML 生成
│   ├── 07_train_and_export.sh    # 训练 + 量化导出
│   ├── eval_model.py             # 批量准确率评估
│   ├── prepare_real_voices.py    # 真实人声预处理与增强
│   └── test_real_voices.sh       # 真实人声评估 wrapper
├── inference/
│   ├── detect.py                 # 唤醒词检测入口（WAV/麦克风）
│   ├── runtime.py                # TFLite 流式推理运行时
│   ├── setup_venv.sh             # 边缘设备推理环境搭建
│   └── requirements.txt          # 推理依赖
├── data/                         # 训练数据（gitignore）
├── work/                         # 克隆的源码与 venv（gitignore）
└── outputs/                      # 训练产物 .tflite + .json（gitignore）
```

---

## 待改进方向（来自 ACCURACY_TEST.md 和实际测试）

1. 采集更多、更多样的真实人声样本（不同人、不同距离、不同环境噪声）
2. 对 `cutoff × window` 做网格搜索，找到召回率与误触发率的最优平衡点
3. 增加负样本集统计误唤醒率（FAR）
4. 用 `help_me.json` 中的阈值（`probability_cutoff=0.97`）重新跑离线测试，对齐 ESPHome 部署行为
5. 考虑调整训练数据增强策略，使合成样本更接近真实录音的声学特征

# GTX 1080 Ti 环境兼容性指南

本文档记录在 GTX 1080 Ti (sm_61) 上训练唤醒词模型时，各软件版本的兼容性约束和踩坑记录。
**核心原则：1080 Ti 是 Pascal 架构 (sm_61)，很多新版本软件已不再支持，版本选择必须谨慎。**

---

## 硬件信息

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce GTX 1080 Ti × 4 |
| 架构 | Pascal (sm_61, Compute Capability 6.1) |
| VRAM | 11 GB × 4 |
| 驱动 | 545.23.08 |
| CUDA (驱动内置) | 12.3 |
| OS | Ubuntu 20.04.6 LTS |
| RAM | 251 GB |

---

## NVIDIA 驱动 & CUDA

| 组件 | 当前版本 | 约束 |
|------|---------|------|
| 驱动 | 545.23 | 支持 CUDA ≤12.3 的应用 |
| CUDA Toolkit | 12.3 (系统安装) | `/usr/local/cuda-12.3` |
| cuDNN | 通过 pip 安装，版本取决于框架 | 见下方 TF/PyTorch 章节 |

**注意事项：**
- 驱动 545 最高支持 CUDA 12.3 运行时。PyTorch cu124+ 的 wheel 可能要求更新的驱动
- 不要升级到 PyTorch 的 cu130 wheel（需要驱动 ≥550+）
- Docker 容器内的 CUDA 版本必须 ≤ 宿主机驱动支持的版本

---

## Python

| 版本 | 路径 | 状态 |
|------|------|------|
| 2.7.18 | `/usr/bin/python2.7` | 系统遗留，不使用 |
| 3.8.10 | `/usr/bin/python3.8` | 系统默认 `python3`，版本过低 |
| 3.9.22 | `/usr/bin/python3.9` | 可用，有 `-dev` 头文件 |
| **3.10.17** | `/usr/bin/python3.10` | **推荐** — TF 2.16 和 OWW 都需要 ≥3.10 |
| 3.12.10 | `/usr/bin/python3.12` | 可用但 TF 2.16 兼容性未验证 |

**注意事项：**
- python3.10 没有 `-dev` 包（`/usr/include/python3.10/` 为空），C 扩展编译会失败
- 解决方案：用预编译 wheel（`manylinux`）或在 Docker 容器内编译
- Docker 容器 (Ubuntu 22.04) 自带 Python 3.10 + dev 头文件，无此问题

---

## TensorFlow (MWW 训练用)

| 组件 | 版本 | 说明 |
|------|------|------|
| **TensorFlow** | **2.16.2** | 最后一个支持 cuDNN 8 的版本 |
| cuDNN (pip) | **8.9.7.29** (`nvidia-cudnn-cu12`) | TF 2.16 需要 cuDNN 8.x |
| numpy | **1.26.4** | TF 2.16 需要 numpy <2.0 |
| Python | 3.10 | TF 2.16 需要 ≥3.9 |

**关键约束：**
- TF 2.17+ 需要 cuDNN 9，但 cuDNN 9 的 pip 包与 PyTorch 的 cuDNN 版本冲突
- **不要升级 TF 到 2.17+**，除非你不需要在同一环境跑 PyTorch
- TF 2.16 的 `model.evaluate()` 返回 `np.ndarray` 而非 Tensor，需要 patch `microwakeword/train.py`（`00_patch_mww.sh`）
- `numpy-minmax` 和 `numpy-rms` 声明需要 numpy≥2，但运行时兼容 1.26，用 `--no-deps` 安装

**TF 版本选择决策树：**
```
需要 GPU 训练？
  ├─ 是 → TF 2.16.2 + cuDNN 8.9.7.29 (pip)
  │       ├─ 需要同环境跑 PyTorch？→ 用 Docker 隔离
  │       └─ 不需要 → 可以直接装
  └─ 否 → tensorflow-cpu 任意版本
```

---

## PyTorch (OWW 训练用)

| 组件 | 版本 | 说明 |
|------|------|------|
| **PyTorch** | **2.5.1+cu121** | 最后一个稳定支持 CUDA 12.1 的版本 |
| torchaudio | **2.5.1+cu121** | 与 PyTorch 版本匹配 |
| CUDA wheel | **cu121** | 必须用 cu121，不能用 cu124/cu130 |

**关键约束：**
- PyTorch 2.6+ 的 cu121 wheel 可能不再提供，需要测试
- **绝对不要用 cu130 wheel**（需要驱动 ≥550，我们是 545）
- **绝对不要用 cu124 wheel**（需要驱动 ≥550）
- 安装时必须用 `--index-url`（不是 `--extra-index-url`），否则 pip 会从 PyPI 拉到 cu130 版本：
  ```bash
  pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  ```

**PyTorch CUDA 版本对照：**
```
驱动 545 (CUDA 12.3) → 最高用 cu121 wheel
驱动 550+ (CUDA 12.4) → 可以用 cu124
驱动 560+ (CUDA 12.6) → 可以用 cu126
```

---

## Docker

| 组件 | 版本/配置 |
|------|----------|
| Docker Engine | 需要 NVIDIA Container Toolkit |
| 基础镜像 | `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` |
| `--gpus all` | 必须，否则容器看不到 GPU |
| `--shm-size=8g` | PyTorch DataLoader 多进程需要，否则报 bus error |

**镜像选择：**
- `cudnn8-devel`：包含 cuDNN 8 头文件和 `ptxas` 编译器，TF 训练需要
- 不要用 `cudnn9` 镜像给 TF 2.16
- `devel` 比 `runtime` 大但包含编译工具，训练必须用 `devel`

**当前镜像：**
| 镜像 | 大小 | 用途 |
|------|------|------|
| `wakeword-mww` | ~15.7GB | TF 2.16.2 + cuDNN 8 + PyTorch CPU |
| `wakeword-oww` | ~16.5GB | PyTorch 2.5.1+cu121 |
| `cosyvoice` | ~17.9GB | CosyVoice2 TTS 生成（PyTorch 2.5.1+cu121） |

---

## pip 依赖版本锁定

### MWW 镜像关键依赖

```
tensorflow==2.16.2
nvidia-cudnn-cu12==8.9.7.29
numpy==1.26.4
numpy-minmax==0.4.0  # --no-deps
numpy-rms==0.5.0     # --no-deps
datasets==2.19.2
setuptools<81        # ≥81 移除了 pkg_resources
torch (CPU)          # 仅用于 piper-sample-generator
piper-phonemize-cross==1.2.1
pymicro-features
```

### OWW 镜像关键依赖

```
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
onnxruntime-gpu
speechbrain>=0.5.14
audiomentations>=0.30.0
deep-phonemizer==0.0.19
edge-tts              # 中文 TTS
setuptools<81
```

### CosyVoice 镜像关键依赖

```
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
numpy==1.26.4
modelscope==1.20.0
transformers==4.51.3
librosa==0.10.2
soundfile==0.12.1
setuptools<81
# 过滤掉: openai-whisper, tensorrt-cu12*, deepspeed, torch/torchaudio（避免覆盖 cu121）
```

---

## 已知问题 & Workaround

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| TF `.numpy()` 报错 | TF 2.16 `evaluate()` 返回 ndarray | `sed` patch 为 `np.asarray()` |
| numpy 2.x 冲突 | numpy-minmax 要求 ≥2，TF 要求 <2 | `pip install --no-deps` |
| cuDNN 版本冲突 | TF 要 cuDNN 8，PyTorch 要 cuDNN 9 | Docker 隔离 |
| PyTorch cu130 不工作 | 驱动 545 太旧 | 锁定 cu121 wheel |
| `pkg_resources` 缺失 | setuptools ≥81 移除了 | `setuptools<81` |
| Docker shm 不足 | 默认 64MB | `--shm-size=8g` |
| python3.10 无 dev 头文件 | 系统包不完整 | Docker 内编译或用预编译 wheel |
| AudioSet tar 404 | HuggingFace 转 parquet | 多 URL 回退 + streaming API |
| `torchcodec` 安装失败 | 编译环境不完整 | 用 `soundfile` + `resampy` 替代 |
| CosyVoice tensorrt-cu12 | 拉入 CUDA 12.4+ 依赖 | Dockerfile 中过滤 tensorrt-cu12* |
| CosyVoice torch 版本覆盖 | requirements.txt 里 torch==2.3.1 | Dockerfile 中过滤 torch/torchaudio，预装 cu121 |
| CosyVoice deepspeed 编译 | sm_61 CUDA kernel 编译可能失败 | Dockerfile 中过滤 deepspeed（推理不需要） |

---

## 升级建议

**安全升级（不会破坏现有环境）：**
- Docker 基础镜像小版本更新（如 12.2.2 → 12.2.x）
- pip 包的 patch 版本更新
- datasets、soundfile 等纯 Python 包

**危险升级（可能破坏环境）：**
- ❌ TensorFlow 2.16 → 2.17+（cuDNN 版本变化）
- ❌ PyTorch cu121 → cu124/cu130（驱动不支持）
- ❌ numpy 1.26 → 2.x（TF 2.16 不兼容）
- ❌ setuptools ≥81（移除 pkg_resources）
- ❌ NVIDIA 驱动升级（可能影响所有 CUDA 应用）

**如果必须升级驱动：**
1. 升级到 550+ 后可以用 PyTorch cu124
2. 升级到 560+ 后可以用 PyTorch cu126
3. TF 2.17+ 需要 cuDNN 9，升级驱动后可以考虑
4. 升级前务必备份当前 Docker 镜像：`docker save wakeword-mww > mww-backup.tar`

---

## 快速恢复

如果环境搞坏了：

```bash
# Docker 镜像是自包含的，重建即可
make build-all

# 数据在宿主机 data/ 目录，不受影响
# 源码在 work/ 目录，不受影响
# 模型在 outputs/ 目录，不受影响
```

最坏情况下，只要 `docker/`、`scripts/`、`Makefile` 还在，整个训练环境可以从零重建。

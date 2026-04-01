# 「help me」唤醒词训练 Wiki

本文档说明本仓库中如何训练默认唤醒词 **「help me」**（输出前缀 `help_me`），对应流水线、环境、测试与部署用法。

---

## 1. 主要原理

- **框架**：使用 [microWakeWord](https://github.com/OHF-Voice/micro-wake-word)（基于 TensorFlow）训练**流式**小模型，最终导出 **量化 TFLite**（`stream_state_internal_quant`），适合 ESPHome / Home Assistant 等边缘设备。
- **任务本质**：在 16 kHz 单声道音频上，对短时频谱特征做二分类——**是否为目标短语**；负样本来自大量「非唤醒」语音/环境声特征，正样本来自 **TTS 合成**（可选混入 **真实人声**）。
- **数据流**：原始 WAV → 统一到 16 kHz → **正样本**（Piper TTS 或 espeak-ng 回退）+ **数据增强**（RIR、AudioSet、FMA 等，见 `scripts/02_download_datasets.sh`）→ 生成 **mmap 特征** → 训练 **MixedNet** 类网络 → 量化并导出 `.tflite`，同时写 **ESPHome 用的 JSON 清单**（阈值、滑窗等）。
- **默认短语与产物**：`KEYWORD_PHRASE=help me`，`KEYWORD_ID=help_me`，输出 `outputs/help_me.tflite` 与 `help_me.json`。

---

## 2. 准备工作

### 2.1 硬件与软件（参考仓库根目录 `README.md` 兼容性表）

| 项目 | 说明 |
|------|------|
| **GPU** | 建议 NVIDIA，算力 sm_35+；本项目在 GTX 1080 Ti 等环境验证 |
| **驱动 / CUDA** | 与 TensorFlow 2.16 + CUDA 12.x 配套（如驱动 ≥520，CUDA 12.3） |
| **Docker** | 需安装 **NVIDIA Container Toolkit**，训练命令使用 `--gpus all` |
| **磁盘 / 内存** | 数据与缓存约数十 GB 量级；内存用于特征加载（README 中示例约 10–20 GB 量级） |

### 2.2 宿主机需具备

- Linux（如 Ubuntu 20.04）+ Docker  
- 能访问外网以下载数据集与克隆 `micro-wake-word`、`piper-sample-generator`  
- 可选：提前录制 **16 kHz 单声道 WAV** 放入 `data/real_voices/`，以提升真实场景表现  

### 2.3 仓库内关键目录

- `data/`：原始数据、负样本、生成特征等（挂载进容器）  
- `work/`：克隆的源码与 venv（持久化可加速二次运行）  
- `outputs/`：训练产物 `.tflite` + `.json`  
- `scripts/`：流水线脚本（宿主机挂载，便于改脚本不重打镜像）  

---

## 3. 详细步骤

整体由 `scripts/run_pipeline.sh` 串起，根目录 `Makefile` 的 `train` 目标会构建镜像并以 `bash -lc` 执行该脚本。

### 3.1 阶段概览

1. **`01_prepare_repos.sh`**  
   - 克隆 `micro-wake-word`、`piper-sample-generator`（若已存在则跳过）  
   - 在 `micro-wake-word` 下创建 venv，安装项目、`tensorflow==2.16.2`、`nvidia-cudnn-cu12`、datasets、音频与重采样依赖、Torch、Piper 相关等  

2. **`00_patch_mww.sh`**  
   - 修补 `microwakeword/train.py`：将 `evaluate` 返回值的 `.numpy()` 改为 `np.asarray(...)`，以兼容 TF 2.16  

3. **`02_download_datasets.sh`**  
   - 下载并准备增强与负样本相关数据（MIT RIR、AudioSet、FMA、负特征 ZIP 等），带多 URL 回传与损坏包重下逻辑  

4. **`03_prepare_audio.py`**  
   - 将原始音频重采样到 **16 kHz**（`soundfile` + `resampy`，不依赖 torchcodec）  

5. **`04_generate_positive_samples.sh`**  
   - 若 `data/positive_raw/<KEYWORD_ID>/` 下尚无 WAV：优先用 **piper-sample-generator** 按 `KEYWORD_PHRASE` 生成 `POSITIVE_SAMPLES` 条；失败则 **espeak-ng + ffmpeg** 合成（速度/音高等有变化以增加多样性）  
   - 若已有 WAV，则整步跳过  

6. **`05_generate_features.py`**  
   - 对正样本目录做增强并生成 **spectrogram mmap 特征**；若存在 `data/real_voices/*.wav`，会**混入**真实人声参与特征生成  

7. **`07_train_and_export.sh`**  
   - 用 `06_write_training_config.py` 生成 `training_parameters.yaml`（步数、正负特征路径、采样权重、学习率等）  
   - 调用 `python -m microwakeword.model_train_eval ... mixednet`（具体卷积结构在脚本中写死）  
   - 开启 **streaming quantized TFLite** 测试项，用 `best_weights`  
   - 拷贝生成的 `stream_state_internal_quant.tflite` 为 `outputs/<KEYWORD_ID>.tflite`  
   - 写入 `outputs/<KEYWORD_ID>.json`（含 `probability_cutoff`、`sliding_window_size` 等 micro 字段）  

### 3.2 一键命令（宿主机）

```bash
make build    # 构建镜像 wakeword-trainer:latest
make train    # 全流程（可改 Makefile 变量或 docker -e）
```

### 3.3 常用环境变量（`scripts/common.sh` / `Makefile`）

| 变量 | 默认 | 含义 |
|------|------|------|
| `KEYWORD_PHRASE` | `help me` | 要训练的短语文本 |
| `KEYWORD_ID` | `help_me` | 输出文件名前缀、正样本子目录名 |
| `PIPER_VOICES` | `en_US-lessac-medium,en_US-amy-medium` | Piper 相关音色（若走 Piper） |
| `POSITIVE_SAMPLES` | `800` | TTS 正样本条数 |
| `TRAIN_STEPS` | `12000` | 训练步数 |

---

## 4. 测试方式

### 4.1 推理目录离线测试（`inference/detect.py`）

- 在设备上：`cd inference && bash setup_venv.sh && source .venv/bin/activate`  
- **单文件**：`python detect.py --model help_me.tflite --wav test.wav --cutoff 0.10 --verbose`  
- **目录批量**：`--wav-dir` 指向一批 wav，可看汇总检出率  
- **实时麦克风**：不传 `--wav` 时从麦克风读（Home Assistant OS 上可自动连 `hassio_audio` 的 Pulse socket）  

### 4.2 准确度说明（`ACCURACY_TEST.md`）

- 目标：在 **真实录音** 上统计「每条 wav 是否至少触发一次」的**召回**  
- 示例：对 `real_0050.wav`～`real_0099.wav` 等跑 `detect.py`，对比不同 `cutoff`、`window`  
- **注意**：`detect.py` 默认 `cutoff=0.10` 与 `help_me.json` 里的 `probability_cutoff: 0.97` **不是同一套口径**，对比 ESPHome 行为时要明确对齐哪一侧  

### 4.3 脚本级评估（`scripts/eval_model.py`）

- 可对 **正样本目录 + 可选负样本目录** 批量算召回与误触发，参数与 `inference` 里检测器一致（`--cutoff`、`--window` 等）  

### 4.4 建议的后续测试

- 用 **JSON 中的阈值** 再跑离线，对齐部署  
- 增加负样本集统计 **误唤醒率（FAR）**  
- 对 `cutoff × window` 做网格搜索，折中漏检与误报  

---

## 5. 使用方法

### 5.1 训练产物

- `outputs/help_me.tflite`：量化流式模型  
- `outputs/help_me.json`：元数据与 micro 参数（含 `probability_cutoff`、`sliding_window_size`、`minimum_esphome_version`）  

### 5.2 ESPHome / Home Assistant

- 将上述两个文件放到 **可被设备 URL 访问** 的位置，在 ESPHome 配置中引用该模型（与官方 micro 唤醒词流程一致）  
- 根据环境噪声与误报情况，在 JSON 中微调 `probability_cutoff` 和 `sliding_window_size`  

### 5.3 采集真实「help me」样本（可选，`collect_samples.py`）

- 在 **PulseAudio** 环境（如文档中的 Home Assistant `pulse.sock`）下，用 `parec` 录制多段 WAV，默认 **16 kHz 单声道**，可设条数、时长、输出目录  
- 将采集结果放入 `data/real_voices/` 后**重新跑流水线**，使特征与真实发音分布更一致  

---

## 6. 其他

- **二次运行加速**：`work/` 里已有克隆与 venv 时，`01` 会跳过重复克隆；`04` 若已有正样本 WAV 会跳过生成。  
- **维护命令**：根目录 `Makefile` 提供 `fix-datasets`、`fix-tf` 等，用于在已有容器工作区内修正 `datasets` 或 TensorFlow/cuDNN 版本。  
- **已知局限（来自 `ACCURACY_TEST.md`）**：某批真实 Rapoo 录音上召回约 36%–38%，单纯降低 `cutoff` 改善有限，通常需要 **更多/更广域真实数据** 或调整训练数据与增强策略。  
- **Dockerfile**：基础镜像 `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`，预装 `espeak-ng`、`git-lfs` 等；入口默认可指向 `run_pipeline.sh`，而 `make train` 用 `bash -lc` 显式执行同一脚本并挂载卷。  

---

## 相关文件

| 路径 | 说明 |
|------|------|
| 根目录 `README.md` | 快速开始、环境变量、流水线编号说明 |
| `ACCURACY_TEST.md` | 离线准确度测试步骤与示例结果 |
| `Makefile` | `build` / `train` 与镜像名、挂载 |

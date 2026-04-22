# ═══════════════════════════════════════════════════════════════════════
# CosyVoice2 TTS 生成镜像
# ═══════════════════════════════════════════════════════════════════════
#
# 硬件约束：
#   GPU:    GTX 1080 Ti (Pascal, sm_61, Compute Capability 6.1)
#   驱动:   545.23 → 最高支持 CUDA 12.3
#   OS:     Ubuntu 20.04 (宿主机) / Ubuntu 22.04 (容器内)
#
# 版本选择理由：
#   基础镜像:  cuda:12.2.2-cudnn8  — 驱动 545 兼容 ≤12.3，cuDNN 8 与 onnxruntime-gpu 1.18 匹配
#   PyTorch:   2.5.1+cu121         — 最后一个稳定支持 cu121 的版本，cu124/cu130 需要驱动 ≥550
#   numpy:     1.26.4              — CosyVoice 和 TF 2.16 都需要 <2.0
#   setuptools: <81                — ≥81 移除了 pkg_resources，modelscope/whisper 等需要
#
# 过滤掉的依赖（不兼容或不需要）：
#   - tensorrt-cu12*    → 会拉入 nvidia-cuda-runtime-cu12 ≥12.4，与驱动 545 不兼容
#   - torch/torchaudio  → requirements.txt 里是 2.3.1（默认 cu124），必须用预装的 2.5.1+cu121
#   - deepspeed         → 编译 CUDA kernel 对 sm_61 可能有问题，TTS 推理不需要
#   - openai-whisper    → 从 requirements.txt 过滤，单独安装（需要先锁 setuptools）
#   - gradio/fastapi*   → Web UI，批量生成不需要，减小镜像
#   - --extra-index-url → 避免从非官方源拉到 cu124 版本的包
#
# pynini/WeTextProcessing 不是必须的，CosyVoice 会自动 fallback
# ═══════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget ffmpeg sox libsox-dev \
    python3 python3-pip python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── 锁定 setuptools<81（pkg_resources 依赖）──
RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install "setuptools<81"

# ── PyTorch 2.5.1+cu121 ──
# 必须用 --index-url（不是 --extra-index-url），否则 pip 会从 PyPI 拉到 cu130 版本
# 绝对不能用 cu124（需要驱动 ≥550）或 cu130（需要驱动 ≥560）
RUN python3 -m pip install \
    "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# ── CosyVoice 依赖（过滤不兼容项）──
COPY work/CosyVoice/requirements.txt /tmp/cosyvoice_req.txt
RUN grep -v -E \
    'openai-whisper|tensorrt|^torch==|^torchaudio==|deepspeed|^--extra-index-url|gradio|fastapi|uvicorn|fastapi-cli' \
    /tmp/cosyvoice_req.txt > /tmp/req_filtered.txt && \
    python3 -m pip install -r /tmp/req_filtered.txt || true

# ── openai-whisper（CosyVoice frontend.py 用了 whisper.log_mel_spectrogram）──
# --no-deps 避免 whisper 拉入 torch==2.3.1 覆盖我们的 2.5.1+cu121
# --no-build-isolation 使用系统 setuptools<81（隔离环境会拉 ≥81，缺 pkg_resources）
# whisper 的运行时依赖（torch, numpy, tiktoken 等）已经由前面的步骤安装
RUN python3 -m pip install "setuptools<81" tiktoken && \
    python3 -m pip install openai-whisper==20231117 --no-deps --no-build-isolation

# ── 锁定 numpy<2.0 + setuptools<81（防止被依赖链升级）──
RUN python3 -m pip install "numpy==1.26.4" "setuptools<81"

# CosyVoice 源码通过 volume mount 挂载到 /workspace/work/CosyVoice
ENV PYTHONPATH="/workspace/work/CosyVoice:/workspace/work/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

ENTRYPOINT ["/bin/bash", "-lc"]

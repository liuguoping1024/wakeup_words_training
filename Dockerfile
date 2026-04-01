FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    wget \
    curl \
    unzip \
    tar \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /workspace

COPY scripts /workspace/scripts
COPY scripts-real /workspace/scripts-real
RUN chmod +x /workspace/scripts/*.sh /workspace/scripts-real/*.sh

ENTRYPOINT ["/workspace/scripts/run_pipeline.sh"]

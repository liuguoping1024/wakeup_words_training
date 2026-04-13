# MWW (micro-wake-word) training image
# TF 2.16.2 + cuDNN 8 + Python 3.10
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg wget curl unzip tar ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential espeak-ng sox \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

# Step 1: Install TF and cuDNN first
RUN python3 -m pip install --upgrade pip "setuptools<81" wheel && \
    python3 -m pip install \
      "tensorflow==2.16.2" \
      "nvidia-cudnn-cu12==8.9.7.29"

# Step 2: Install PyTorch CPU-only (for piper-sample-generator, no GPU needed)
RUN python3 -m pip install \
      torch torchaudio \
      --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install other deps
RUN python3 -m pip install \
      "datasets==2.19.2" soundfile librosa scipy tqdm resampy \
      huggingface_hub audiomentations audio_metadata mmap_ninja \
      pyyaml webrtcvad-wheels ai-edge-litert \
      piper-phonemize-cross==1.2.1 pymicro-features

# Step 4: Pin numpy LAST (TF needs 1.26.x, torch CPU doesn't care)
RUN python3 -m pip install "numpy==1.26.4" && \
    python3 -m pip install "numpy-minmax==0.4.0" "numpy-rms==0.5.0" --no-deps

# Verify
RUN python3 -c "import tensorflow; print('TF', tensorflow.__version__); import numpy; print('numpy', numpy.__version__)"

ENTRYPOINT ["/bin/bash", "-lc"]

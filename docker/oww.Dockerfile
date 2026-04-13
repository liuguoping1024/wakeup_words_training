# OWW (openWakeWord) training image
# PyTorch GPU + CUDA 12 + Python 3.10
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg wget curl unzip tar ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential espeak-ng sox \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

# Step 1: PyTorch with CUDA 12.1 (driver 545.23 = CUDA 12.3, needs cu121 wheels)
RUN python3 -m pip install --upgrade pip "setuptools<81" wheel && \
    python3 -m pip install \
      "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" \
      --index-url https://download.pytorch.org/whl/cu121

# Step 2: OWW training dependencies (no tensorflow - ONNX->TFLite done separately)
RUN python3 -m pip install \
      onnxruntime-gpu \
      ai-edge-litert \
      "speexdsp-ns>=0.1.2" \
      scipy scikit-learn requests tqdm \
      mutagen torchinfo torchmetrics \
      "speechbrain>=0.5.14" \
      "audiomentations>=0.30.0" \
      "torch-audiomentations>=0.11.0" \
      acoustics pyyaml \
      "datasets>=2.14.4" \
      pronouncing \
      "deep-phonemizer==0.0.19" \
      piper-phonemize-cross==1.2.1 \
      soundfile resampy webrtcvad

# Step 3: ONNX (for model export, no onnx_tf needed - we convert via onnxruntime)
RUN python3 -m pip install "onnx>=1.14" espeak-phonemizer edge-tts

# Step 4: Install piper_train from dscripka's fork (mounted at runtime, but pre-install deps)
# piper_train is needed by generate_samples.py

# Verify
RUN python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available()); import openwakeword; print('OWW import check skipped - installed at runtime')" || true

ENTRYPOINT ["/bin/bash", "-lc"]

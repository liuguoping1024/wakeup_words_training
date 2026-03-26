# Wake Word Training: "help me"

A fully automated, containerized pipeline to train a custom wake word model using [microWakeWord](https://github.com/OHF-Voice/micro-wake-word), exportable as `.tflite` for ESPHome / Home Assistant.

---

## Hardware & Compatibility

| Layer | This Setup | Required | Status |
|-------|-----------|----------|--------|
| GPU | GTX 1080 Ti (sm_61) × 4 | sm_35+ | ✅ |
| Driver | 545.23 | ≥520 for CUDA 12.x | ✅ |
| CUDA runtime | 12.3 (driver built-in) | 12.3 | ✅ |
| cuDNN | pip 8.9.7.29 | 8.9.x | ✅ |
| TensorFlow | 2.16.2 | ≥2.16 | ✅ |
| Docker base image | `cuda:12.2.2-cudnn8-devel` | includes `ptxas` | ✅ |
| VRAM | 11 GB × 4 | model ~100 KB, training <1 GB | ✅ very comfortable |
| RAM | 251 GB | data loading ~10–20 GB | ✅ |
| Disk | 1.3 TB available | ~20–50 GB | ✅ |

**OS**: Ubuntu 20.04.6 LTS
**Docker**: with NVIDIA Container Toolkit (`--gpus all`)

---

## Outputs

After training, the following files are written to `outputs/`:

- `help_me.tflite` — quantized streaming TFLite model
- `help_me.json` — ESPHome manifest (cutoff, window size, etc.)

---

## Quick Start

### 1. Build the Docker image

```bash
make build
# or:
docker build -t wakeword-trainer:latest .
```

### 2. Run the full pipeline

```bash
make train
# or manually:
docker run --rm --gpus all \
  -e KEYWORD_PHRASE="help me" \
  -e KEYWORD_ID="help_me" \
  -e POSITIVE_SAMPLES="800" \
  -e TRAIN_STEPS="12000" \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -v "$(pwd)/work:/workspace/work" \
  -v "$(pwd)/scripts:/workspace/scripts" \
  wakeword-trainer:latest
```

### 3. (Optional) Add real voice samples

For better real-world accuracy, record real voice samples and place them in `data/real_voices/` before training:

```bash
mkdir -p data/real_voices
# copy *.wav files recorded at 16kHz mono into data/real_voices/
```

The pipeline will automatically detect and merge them with TTS-generated positive samples.

---

## Configurable Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KEYWORD_PHRASE` | `help me` | Wake word text |
| `KEYWORD_ID` | `help_me` | Output file name prefix |
| `POSITIVE_SAMPLES` | `800` | Number of TTS positive samples to generate |
| `TRAIN_STEPS` | `12000` | Training steps |

---

## Pipeline Stages

`scripts/run_pipeline.sh` orchestrates:

1. **`01`** Clone & install `micro-wake-word` and `piper-sample-generator`, pin TF 2.16.2 + cuDNN 8.9.7.29
2. **`00`** Patch `microwakeword/train.py` for TF 2.16.2 compatibility (`.numpy()` → `np.asarray()`)
3. **`02`** Download augmentation data (MIT RIR / AudioSet / FMA) and negative feature ZIPs with multi-URL fallback
4. **`03`** Resample raw audio to 16 kHz using `soundfile` + `resampy` (no `torchcodec` required)
5. **`04`** Generate positive TTS samples via Piper (fallback: `espeak-ng`)
6. **`05`** Augment & generate spectrogram features (Ragged Mmap); merges real voice samples if present
7. **`07`** Train, quantize, and export `tflite` + `json`

---

## Inference (Embedded / Edge Devices)

The `inference/` directory contains a lightweight runtime for running the model on any Linux device (tested on Orange Pi 5B, aarch64, no GPU):

```bash
cd inference/
bash setup_venv.sh          # create venv, install ai-edge-litert / pymicro-features / pyaudio
source .venv/bin/activate

# Test with a WAV file
python detect.py --model help_me.tflite --wav test.wav --cutoff 0.10 --verbose

# Real-time microphone detection
python detect.py --model help_me.tflite --cutoff 0.10 --verbose
```

For Home Assistant OS devices, the script auto-connects to the `hassio_audio` PulseAudio socket at `/var/lib/homeassistant/audio/external/pulse.sock`.

---

## ESPHome Integration

Copy `outputs/help_me.tflite` and `outputs/help_me.json` to a web-accessible location, then reference the model in your ESPHome config.

> Tip: tune `probability_cutoff` and `sliding_window_size` in the JSON based on your environment's false-positive / false-negative rate.

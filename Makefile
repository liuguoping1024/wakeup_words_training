IMAGE ?= wakeword-trainer:latest
HOST_DIR := $(CURDIR)
KEYWORD_PHRASE ?= help me
KEYWORD_ID ?= help_me
POSITIVE_SAMPLES ?= 800
TRAIN_STEPS ?= 12000

.PHONY: build train fix-datasets fix-tf

build:
	docker build -t $(IMAGE) .

fix-datasets:
	docker run --rm \
		-v "$(HOST_DIR)/work:/workspace/work" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "/workspace/work/micro-wake-word/.venv/bin/pip install 'datasets==2.19.2'"

fix-tf:
	docker run --rm \
		-v "$(HOST_DIR)/work:/workspace/work" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "\
		  PIP=/workspace/work/micro-wake-word/.venv/bin/pip && \
		  \$$PIP install 'tensorflow==2.16.2' 'nvidia-cudnn-cu12==8.9.7.29' && \
		  \$$PIP install 'numpy==1.26.4' && \
		  \$$PIP install 'numpy-minmax==0.4.0' 'numpy-rms==0.5.0' --no-deps \
		"

train: build
	docker run --rm --gpus all \
		-e KEYWORD_PHRASE="$(KEYWORD_PHRASE)" \
		-e KEYWORD_ID="$(KEYWORD_ID)" \
		-e POSITIVE_SAMPLES="$(POSITIVE_SAMPLES)" \
		-e TRAIN_STEPS="$(TRAIN_STEPS)" \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		-e XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
		-e TF_CPP_MIN_LOG_LEVEL=2 \
		-v "$(HOST_DIR)/data:/workspace/data" \
		-v "$(HOST_DIR)/outputs:/workspace/outputs" \
		-v "$(HOST_DIR)/work:/workspace/work" \
		-v "$(HOST_DIR)/scripts:/workspace/scripts" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "chmod +x /workspace/scripts/*.sh && /workspace/scripts/run_pipeline.sh"

# ── 「你好树实」真实语音训练 ──────────────────────────────────────────────
REAL_KEYWORD_PHRASE ?= 你好树实
REAL_KEYWORD_ID ?= nihao_shushi
REAL_TRAIN_STEPS ?= 15000
REAL_TARGET_POSITIVE ?= 5000

.PHONY: train-real split-real eval-real

split-real:
	docker run --rm \
		-e KEYWORD_PHRASE="$(REAL_KEYWORD_PHRASE)" \
		-e KEYWORD_ID="$(REAL_KEYWORD_ID)" \
		-v "$(HOST_DIR)/data:/workspace/data" \
		-v "$(HOST_DIR)/work:/workspace/work" \
		-v "$(HOST_DIR)/scripts-real:/workspace/scripts-real" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "\
		  source /workspace/work/micro-wake-word/.venv/bin/activate && \
		  python /workspace/scripts-real/01_split_real_voices.py \
		    --sounds-dir /workspace/data/sounds \
		    --output-dir /workspace/data/positive_raw/$(REAL_KEYWORD_ID)"

train-real: build
	docker run --rm --gpus all \
		-e KEYWORD_PHRASE="$(REAL_KEYWORD_PHRASE)" \
		-e KEYWORD_ID="$(REAL_KEYWORD_ID)" \
		-e TRAIN_STEPS="$(REAL_TRAIN_STEPS)" \
		-e TARGET_POSITIVE="$(REAL_TARGET_POSITIVE)" \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		-e XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
		-e TF_CPP_MIN_LOG_LEVEL=2 \
		-v "$(HOST_DIR)/data:/workspace/data" \
		-v "$(HOST_DIR)/outputs:/workspace/outputs" \
		-v "$(HOST_DIR)/work:/workspace/work" \
		-v "$(HOST_DIR)/scripts:/workspace/scripts" \
		-v "$(HOST_DIR)/scripts-real:/workspace/scripts-real" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "chmod +x /workspace/scripts/*.sh /workspace/scripts-real/*.sh && /workspace/scripts-real/run_pipeline.sh"

eval-real:
	docker run --rm \
		-v "$(HOST_DIR)/data:/workspace/data" \
		-v "$(HOST_DIR)/outputs:/workspace/outputs" \
		-v "$(HOST_DIR)/work:/workspace/work" \
		-v "$(HOST_DIR)/inference:/workspace/inference" \
		-v "$(HOST_DIR)/scripts-real:/workspace/scripts-real" \
		--entrypoint /bin/bash \
		$(IMAGE) -lc "\
		  source /workspace/work/micro-wake-word/.venv/bin/activate && \
		  python /workspace/scripts-real/eval_model.py \
		    --model /workspace/outputs/$(REAL_KEYWORD_ID).tflite \
		    --pos /workspace/data/positive_raw/$(REAL_KEYWORD_ID) \
		    --cutoff 0.10 --window 3"

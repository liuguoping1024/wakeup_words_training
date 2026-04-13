# ═══════════════════════════════════════════════════════════════════════
# Wake Word Training - Unified Makefile
# MWW = micro-wake-word (TF 2.16, Docker)
# OWW = openWakeWord (PyTorch, Docker)
# ═══════════════════════════════════════════════════════════════════════

HOST_DIR := $(CURDIR)
LOG_DIR  := $(HOST_DIR)/logs
MWW_IMAGE ?= wakeword-mww:latest
OWW_IMAGE ?= wakeword-oww:latest

# ── Common Docker flags ──────────────────────────────────────────────
DOCKER_COMMON = --rm --gpus all \
	-e PYTHONUNBUFFERED=1 \
	-e TF_FORCE_GPU_ALLOW_GROWTH=true \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e STDBUF_CMD="" \
	-v "$(HOST_DIR)/data:/workspace/data" \
	-v "$(HOST_DIR)/outputs:/workspace/outputs" \
	-v "$(HOST_DIR)/work:/workspace/work" \
	-v "$(HOST_DIR)/scripts:/workspace/scripts" \
	-v "$(HOST_DIR)/inference:/workspace/inference" \
	-v "$(HOST_DIR)/logs:/workspace/logs"

DOCKER_OWW_EXTRA = --shm-size=8g

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# ═══════════════════════════════════════════════════════════════════════
# Docker Images
# ═══════════════════════════════════════════════════════════════════════

.PHONY: build-mww build-oww build-all

build-mww:
	docker build -t $(MWW_IMAGE) -f docker/mww.Dockerfile .

build-oww:
	docker build -t $(OWW_IMAGE) -f docker/oww.Dockerfile .

build-all: build-mww build-oww

# ═══════════════════════════════════════════════════════════════════════
# MWW: "help me" (TTS positive samples)
# ═══════════════════════════════════════════════════════════════════════

MWW_HELP_PHRASE ?= help me
MWW_HELP_ID ?= help_me
MWW_HELP_SAMPLES ?= 800
MWW_HELP_STEPS ?= 12000

.PHONY: mww-help-me

mww-help-me: build-mww $(LOG_DIR)
	docker run $(DOCKER_COMMON) \
		-e KEYWORD_PHRASE="$(MWW_HELP_PHRASE)" \
		-e KEYWORD_ID="$(MWW_HELP_ID)" \
		-e POSITIVE_SAMPLES="$(MWW_HELP_SAMPLES)" \
		-e TRAIN_STEPS="$(MWW_HELP_STEPS)" \
		$(MWW_IMAGE) \
		"chmod +x /workspace/scripts/mww/*.sh && /workspace/scripts/mww/run_pipeline_tts.sh" \
		2>&1 | tee $(LOG_DIR)/mww_help_me.log

# ═══════════════════════════════════════════════════════════════════════
# MWW: "help help"
# ═══════════════════════════════════════════════════════════════════════

MWW_HELPHELP_PHRASE ?= help help
MWW_HELPHELP_ID ?= help_help
MWW_HELPHELP_SAMPLES ?= 800
MWW_HELPHELP_STEPS ?= 12000

.PHONY: mww-help-help

mww-help-help: build-mww $(LOG_DIR)
	docker run $(DOCKER_COMMON) \
		-e KEYWORD_PHRASE="$(MWW_HELPHELP_PHRASE)" \
		-e KEYWORD_ID="$(MWW_HELPHELP_ID)" \
		-e POSITIVE_SAMPLES="$(MWW_HELPHELP_SAMPLES)" \
		-e TRAIN_STEPS="$(MWW_HELPHELP_STEPS)" \
		$(MWW_IMAGE) \
		"chmod +x /workspace/scripts/mww/*.sh && /workspace/scripts/mww/run_pipeline_tts.sh" \
		2>&1 | tee $(LOG_DIR)/mww_help_help.log

# ═══════════════════════════════════════════════════════════════════════
# MWW: "你好树实" (real voice positive samples)
# ═══════════════════════════════════════════════════════════════════════

MWW_NIHAO_PHRASE ?= 你好树实
MWW_NIHAO_ID ?= nihao_shushi
MWW_NIHAO_STEPS ?= 15000
MWW_NIHAO_TARGET ?= 5000

.PHONY: mww-nihao-shushi

mww-nihao-shushi: build-mww $(LOG_DIR)
	docker run $(DOCKER_COMMON) \
		-e KEYWORD_PHRASE="$(MWW_NIHAO_PHRASE)" \
		-e KEYWORD_ID="$(MWW_NIHAO_ID)" \
		-e TRAIN_STEPS="$(MWW_NIHAO_STEPS)" \
		-e TARGET_POSITIVE="$(MWW_NIHAO_TARGET)" \
		$(MWW_IMAGE) \
		"chmod +x /workspace/scripts/mww/*.sh && /workspace/scripts/mww/run_pipeline_real.sh" \
		2>&1 | tee $(LOG_DIR)/mww_nihao_shushi.log

# ═══════════════════════════════════════════════════════════════════════
# OWW: "help me" (TTS)
# ═══════════════════════════════════════════════════════════════════════

OWW_HELP_STEPS ?= 50000
OWW_HELP_SAMPLES ?= 10000

.PHONY: oww-help-me

oww-help-me: build-oww $(LOG_DIR)
	docker run $(DOCKER_COMMON) $(DOCKER_OWW_EXTRA) \
		$(OWW_IMAGE) \
		"pip install -e /workspace/work/openWakeWord --no-deps -q && \
		 pip install espeak-phonemizer -q && \
		 export PYTHONPATH=/workspace/work/piper-sample-generator-oww:\$$PYTHONPATH && \
		 python3 -u /workspace/work/openWakeWord/openwakeword/train.py \
		   --training_config /workspace/data/oww/help_me_config.yaml \
		   --generate_clips && \
		 python3 -u /workspace/work/openWakeWord/openwakeword/train.py \
		   --training_config /workspace/data/oww/help_me_config.yaml \
		   --augment_clips && \
		 python3 -u /workspace/work/openWakeWord/openwakeword/train.py \
		   --training_config /workspace/data/oww/help_me_config.yaml \
		   --train_model" \
		2>&1 | tee $(LOG_DIR)/oww_help_me.log

# ═══════════════════════════════════════════════════════════════════════
# OWW: "你好树实" (real voice)
# ═══════════════════════════════════════════════════════════════════════

OWW_NIHAO_STEPS ?= 50000

.PHONY: oww-nihao-shushi

oww-nihao-shushi: build-oww $(LOG_DIR)
	docker run $(DOCKER_COMMON) $(DOCKER_OWW_EXTRA) \
		$(OWW_IMAGE) \
		"pip install -e /workspace/work/openWakeWord --no-deps -q && \
		 python3 -u /workspace/scripts/oww/train_nihao_oww.py \
		   --steps $(OWW_NIHAO_STEPS)" \
		2>&1 | tee $(LOG_DIR)/oww_nihao_shushi.log

# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

.PHONY: eval-mww-help eval-mww-helphelp eval-mww-nihao eval-all

eval-mww-help:
	docker run $(DOCKER_COMMON) \
		$(MWW_IMAGE) \
		"pip install -e /workspace/work/micro-wake-word --no-deps -q && \
		 python3 /workspace/scripts/eval_model.py \
		  --model /workspace/outputs/$(MWW_HELP_ID).tflite \
		  --pos /workspace/data/real_voices \
		  --cutoff 0.10 --window 3"

eval-mww-helphelp:
	docker run $(DOCKER_COMMON) \
		$(MWW_IMAGE) \
		"pip install -e /workspace/work/micro-wake-word --no-deps -q && \
		 python3 /workspace/scripts/eval_model.py \
		  --model /workspace/outputs/$(MWW_HELPHELP_ID).tflite \
		  --pos /workspace/data/real_voices_help_help \
		  --cutoff 0.10 --window 3"

eval-mww-nihao:
	docker run $(DOCKER_COMMON) \
		$(MWW_IMAGE) \
		"pip install -e /workspace/work/micro-wake-word --no-deps -q && \
		 python3 /workspace/scripts/eval_model.py \
		  --model /workspace/outputs/$(MWW_NIHAO_ID).tflite \
		  --pos /workspace/data/positive_raw/$(MWW_NIHAO_ID) \
		  --cutoff 0.10 --window 3"

eval-all: eval-mww-help eval-mww-helphelp eval-mww-nihao

# ═══════════════════════════════════════════════════════════════════════
# Train ALL
# ═══════════════════════════════════════════════════════════════════════

.PHONY: train-all train-mww train-oww

train-mww: mww-help-me mww-help-help mww-nihao-shushi
train-oww: oww-help-me oww-nihao-shushi
train-all: train-mww train-oww
	@echo "All training complete."

# ═══════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════

.PHONY: clean-features clean-models clean-logs

clean-features:
	rm -rf data/generated_augmented_features

clean-models:
	rm -rf outputs/*.tflite outputs/*.json outputs/oww/*.onnx

clean-logs:
	rm -rf logs/

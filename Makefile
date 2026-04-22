# ═══════════════════════════════════════════════════════════════════════
# Wake Word Training - Unified Makefile
# MWW = micro-wake-word (TF 2.16, Docker)
# OWW = openWakeWord (PyTorch, Docker)
# ═══════════════════════════════════════════════════════════════════════

HOST_DIR := $(CURDIR)
LOG_DIR  := $(HOST_DIR)/logs
MWW_IMAGE ?= wakeword-mww:latest
OWW_IMAGE ?= wakeword-oww:latest
COSYVOICE_IMAGE ?= cosyvoice:latest

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

.PHONY: build-mww build-oww build-cosyvoice build-all

build-mww:
	docker build -t $(MWW_IMAGE) -f docker/mww.Dockerfile .

build-oww:
	docker build -t $(OWW_IMAGE) -f docker/oww.Dockerfile .

build-cosyvoice:
	docker build -t $(COSYVOICE_IMAGE) -f docker/cosyvoice.Dockerfile .

build-all: build-mww build-oww build-cosyvoice

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
# MWW: "救命" (Chinese, multi-TTS + real voice)
#
# 完整流程（三步）：
#   1. make jiuming-tts-host    — 宿主机生成 edge-tts + Piper 样本（需网络）
#   2. make jiuming-tts-cosy    — Docker 内生成 CosyVoice2 样本（需 GPU）
#   3. make mww-jiuming         — 合并样本 + MWW 训练
#
# 一键执行：
#   make jiuming-full           — 依次执行 1→2→3
#   make mww-jiuming            — 跳过 TTS 生成，直接用已有样本训练
# ═══════════════════════════════════════════════════════════════════════

MWW_JIUMING_PHRASE ?= 救命
MWW_JIUMING_ID ?= jiuming
MWW_JIUMING_STEPS ?= 15000

# TTS 样本数量
JIUMING_N_POS_EDGE ?= 1500
JIUMING_N_NEG_EDGE ?= 2000
JIUMING_N_POS_PIPER ?= 200
JIUMING_N_NEG_PIPER ?= 300
JIUMING_N_POS_COSY ?= 1000
JIUMING_N_NEG_COSY ?= 1000

.PHONY: jiuming-tts-host jiuming-tts-cosy jiuming-merge mww-jiuming jiuming-full

# Step 1: 宿主机生成 edge-tts + Piper 样本
jiuming-tts-host: $(LOG_DIR)
	python3 scripts/mww/generate_zh_tts_all.py \
		--keyword "$(MWW_JIUMING_PHRASE)" \
		--keyword-id "$(MWW_JIUMING_ID)" \
		--pos-dir data/tts_positive/$(MWW_JIUMING_ID) \
		--neg-dir data/tts_negative/$(MWW_JIUMING_ID) \
		--n-pos-edge $(JIUMING_N_POS_EDGE) \
		--n-neg-edge $(JIUMING_N_NEG_EDGE) \
		--n-pos-piper $(JIUMING_N_POS_PIPER) \
		--n-neg-piper $(JIUMING_N_NEG_PIPER) \
		2>&1 | tee $(LOG_DIR)/jiuming_tts_host.log

# Step 2: Docker 内生成 CosyVoice2 样本
jiuming-tts-cosy: $(LOG_DIR)
	docker run $(DOCKER_COMMON) --shm-size=8g \
		$(COSYVOICE_IMAGE) \
		"python3 -u /workspace/scripts/mww/generate_cosyvoice_samples.py \
		  --keyword '$(MWW_JIUMING_PHRASE)' \
		  --keyword-id '$(MWW_JIUMING_ID)' \
		  --pos-dir /workspace/data/tts_positive/$(MWW_JIUMING_ID) \
		  --neg-dir /workspace/data/tts_negative/$(MWW_JIUMING_ID) \
		  --n-pos $(JIUMING_N_POS_COSY) \
		  --n-neg $(JIUMING_N_NEG_COSY) \
		  --ref-dir /workspace/data/real_voices_jiuming" \
		2>&1 | tee $(LOG_DIR)/jiuming_tts_cosy.log

# Step 3: 合并所有样本 + MWW 训练
mww-jiuming: build-mww $(LOG_DIR)
	python3 scripts/mww/merge_tts_samples.py \
		--keyword-id "$(MWW_JIUMING_ID)" \
		--tts-pos-dir data/tts_positive/$(MWW_JIUMING_ID) \
		--real-dir data/real_voices_jiuming \
		--output-dir data/positive_raw/$(MWW_JIUMING_ID) \
		--clean
	docker run $(DOCKER_COMMON) \
		-e KEYWORD_PHRASE="$(MWW_JIUMING_PHRASE)" \
		-e KEYWORD_ID="$(MWW_JIUMING_ID)" \
		-e TRAIN_STEPS="$(MWW_JIUMING_STEPS)" \
		$(MWW_IMAGE) \
		"chmod +x /workspace/scripts/mww/*.sh && /workspace/scripts/mww/run_pipeline_tts.sh" \
		2>&1 | tee $(LOG_DIR)/mww_jiuming.log

# 一键全流程
jiuming-full: jiuming-tts-host jiuming-tts-cosy mww-jiuming

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
# OWW: "你好树实" (real voice) — 旧版，保留兼容
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
# OWW 中文唤醒词 v2（新 pipeline）
#
# 两步流程：
#   1. 宿主机生成对抗性负样本（需要网络，edge-tts）
#   2. Docker 内训练（prepare → augment → train → export）
#
# 用法：
#   make oww-zh                          # 默认训练你好树实
#   make oww-zh KEYWORD=救命 KEYWORD_ID=jiuming
#   make oww-zh-train-only               # 跳过 TTS 生成（已有负样本）
#   make oww-zh-negatives                # 只生成负样本
# ═══════════════════════════════════════════════════════════════════════

KEYWORD ?= 你好树实
KEYWORD_ID ?= nihao_shushi
OWW_ZH_STEPS ?= 80000
OWW_ZH_LAYER ?= 64
OWW_ZH_AUG_ROUNDS ?= 2
OWW_ZH_N_NEG_TRAIN ?= 5000
OWW_ZH_N_NEG_TEST ?= 1000
OWW_ZH_N_POS_TRAIN ?= 15000
OWW_ZH_N_POS_TEST ?= 3000

.PHONY: oww-zh-negatives oww-zh-train-only oww-zh

oww-zh-negatives: $(LOG_DIR)
	python3 scripts/oww/generate_zh_negatives.py \
		--keyword "$(KEYWORD)" \
		--output-dir outputs/oww/$(KEYWORD_ID) \
		--n-train $(OWW_ZH_N_NEG_TRAIN) \
		--n-test $(OWW_ZH_N_NEG_TEST) \
		2>&1 | tee $(LOG_DIR)/oww_$(KEYWORD_ID)_negatives.log

oww-zh-train-only: build-oww $(LOG_DIR)
	docker run $(DOCKER_COMMON) $(DOCKER_OWW_EXTRA) \
		$(OWW_IMAGE) \
		"pip install -e /workspace/work/openWakeWord --no-deps -q && \
		 python3 -u /workspace/scripts/oww/train_zh_oww.py \
		   --keyword-id $(KEYWORD_ID) \
		   --positive-dir /workspace/data/positive_raw/$(KEYWORD_ID) \
		   --n-pos-train $(OWW_ZH_N_POS_TRAIN) \
		   --n-pos-test $(OWW_ZH_N_POS_TEST) \
		   --augment-rounds $(OWW_ZH_AUG_ROUNDS) \
		   --steps $(OWW_ZH_STEPS) \
		   --layer-size $(OWW_ZH_LAYER) \
		   --overwrite-features" \
		2>&1 | tee $(LOG_DIR)/oww_$(KEYWORD_ID)_zh.log

oww-zh: oww-zh-negatives oww-zh-train-only

# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

.PHONY: eval-mww-help eval-mww-helphelp eval-mww-nihao eval-oww-nihao eval-all

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

eval-mww-jiuming:
	docker run $(DOCKER_COMMON) \
		$(MWW_IMAGE) \
		"pip install -e /workspace/work/micro-wake-word --no-deps -q && \
		 python3 /workspace/scripts/eval_model.py \
		  --model /workspace/outputs/$(MWW_JIUMING_ID).tflite \
		  --pos /workspace/data/real_voices_jiuming \
		  --neg /workspace/data/tts_negative/jiuming \
		  --cutoff 0.10 --window 3"

eval-mww-nihao:
	docker run $(DOCKER_COMMON) \
		$(MWW_IMAGE) \
		"pip install -e /workspace/work/micro-wake-word --no-deps -q && \
		 python3 /workspace/scripts/eval_model.py \
		  --model /workspace/outputs/$(MWW_NIHAO_ID).tflite \
		  --pos /workspace/data/positive_raw/$(MWW_NIHAO_ID) \
		  --cutoff 0.10 --window 3"

eval-all: eval-mww-help eval-mww-helphelp eval-mww-jiuming eval-mww-nihao eval-oww-nihao

# OWW 模型评估
OWW_EVAL_THRESHOLD ?= 0.5
OWW_EVAL_MAX_POS ?= 500

eval-oww-nihao:
	docker run $(DOCKER_COMMON) $(DOCKER_OWW_EXTRA) \
		$(OWW_IMAGE) \
		"pip install -e /workspace/work/openWakeWord --no-deps -q && \
		 python3 -u /workspace/scripts/oww/eval_oww_model.py \
		   --model /workspace/outputs/oww/nihao_shushi.onnx \
		   --pos /workspace/data/positive_raw/nihao_shushi \
		   --neg /workspace/outputs/oww/nihao_shushi/negative_test \
		   --threshold $(OWW_EVAL_THRESHOLD) \
		   --max-pos $(OWW_EVAL_MAX_POS) \
		   --max-neg 500"

# ═══════════════════════════════════════════════════════════════════════
# Train ALL
# ═══════════════════════════════════════════════════════════════════════

.PHONY: train-all train-mww train-oww

train-mww: mww-help-me mww-help-help mww-jiuming mww-nihao-shushi
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

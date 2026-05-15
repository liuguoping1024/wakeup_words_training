#!/usr/bin/env bash
# help help v4 MWW 训练（Docker 内）
set -euo pipefail

SCRIPT_DIR="/workspace/scripts/mww"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs"
MWW_DIR="/workspace/work/micro-wake-word"
KEYWORD_ID="help_help_v4"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== help help v4 MWW training ==="

chmod +x ${SCRIPT_DIR}/*.sh
${SCRIPT_DIR}/01_prepare_repos.sh
${SCRIPT_DIR}/00_patch_mww.sh
${SCRIPT_DIR}/02_download_datasets.sh
python3 ${SCRIPT_DIR}/03_prepare_audio.py --data-dir "${DATA_DIR}"

# 对抗性负样本 mmap
ADV_MMAP="${DATA_DIR}/negative_datasets/en_helphelp_adv_v4/en_helphelp_adv_v4/training/en_helphelp_adv_v4_mmap"
if [ -d "$ADV_MMAP" ]; then
    log "Adversarial mmap exists, skip"
else
    log "Generating adversarial mmap..."
    python3 ${SCRIPT_DIR}/generate_adversarial_features.py \
        --input-dir /workspace/outputs/help_help_adversarial_v4 \
        --output-dir ${DATA_DIR}/negative_datasets/en_helphelp_adv_v4/en_helphelp_adv_v4 \
        --data-dir ${DATA_DIR}
fi

# 正样本特征
FEATURES_DIR="${DATA_DIR}/generated_augmented_features"
log "Generating positive features..."
rm -rf "${FEATURES_DIR}"
python3 ${SCRIPT_DIR}/03_generate_features_real.py \
    --positive-dir "${DATA_DIR}/positive_augmented/help_help_v4" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURES_DIR}"

log "Checking mmap..."
find ${FEATURES_DIR} -name '*_mmap' -type d | while read d; do
    echo "  $d: $(ls $d/ | wc -l) files"
done

log "Writing training config..."
cat > "${MWW_DIR}/training_parameters.yaml" <<'YAMLEOF'
window_step_ms: 10
train_dir: trained_models/help_help_v4
features:
- features_dir: generated_augmented_features
  sampling_weight: 2.0
  penalty_weight: 1.0
  truth: true
  truncation_strategy: truncate_start
  type: mmap
- features_dir: negative_datasets/speech/speech
  sampling_weight: 5.0
  penalty_weight: 1.0
  truth: false
  truncation_strategy: random
  type: mmap
- features_dir: negative_datasets/dinner_party/dinner_party
  sampling_weight: 5.0
  penalty_weight: 1.0
  truth: false
  truncation_strategy: random
  type: mmap
- features_dir: negative_datasets/no_speech/no_speech
  sampling_weight: 3.0
  penalty_weight: 1.0
  truth: false
  truncation_strategy: random
  type: mmap
- features_dir: negative_datasets/en_helphelp_adv_v4/en_helphelp_adv_v4
  sampling_weight: 10.0
  penalty_weight: 1.5
  truth: false
  truncation_strategy: random
  type: mmap
- features_dir: negative_datasets/dinner_party_eval/dinner_party_eval
  sampling_weight: 0.0
  penalty_weight: 1.0
  truth: false
  truncation_strategy: split
  type: mmap
training_steps:
- 30000
positive_class_weight:
- 1
negative_class_weight:
- 20
learning_rates:
- 0.001
batch_size: 128
time_mask_max_size:
- 0
time_mask_count:
- 0
freq_mask_max_size:
- 0
freq_mask_count:
- 0
eval_step_interval: 500
clip_duration_ms: 1500
target_minimization: 0.9
minimization_metric: null
maximization_metric: average_viable_recall
YAMLEOF

log "Training 30000 steps..."
cd "${MWW_DIR}"
ln -sfn "${DATA_DIR}/generated_augmented_features" "${MWW_DIR}/generated_augmented_features"
ln -sfn "${DATA_DIR}/negative_datasets" "${MWW_DIR}/negative_datasets"

python3 -u -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tf_nonstreaming 0 \
    --test_tflite_nonstreaming 0 \
    --test_tflite_nonstreaming_quantized 0 \
    --test_tflite_streaming 0 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block "1,1,1,1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3

log "Exporting..."
TFLITE_SRC="${MWW_DIR}/trained_models/${KEYWORD_ID}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
if [ -f "$TFLITE_SRC" ]; then
    cp -f "${TFLITE_SRC}" "${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
    cat > "${OUTPUT_DIR}/${KEYWORD_ID}.json" <<EOF
{
  "type": "micro",
  "wake_word": "help help",
  "author": "local-training",
  "website": "https://github.com/OHF-Voice/micro-wake-word",
  "version": 1,
  "model": "${KEYWORD_ID}.tflite",
  "micro": {
    "probability_cutoff": 0.80,
    "sliding_window_size": 3,
    "minimum_esphome_version": "2024.7.0"
  }
}
EOF
    log "Exported: ${OUTPUT_DIR}/${KEYWORD_ID}.tflite"
else
    log "WARNING: TFLite not found"
fi

log "Verifying with TTS test set..."
TEST_DIR="/workspace/outputs/help_help_edgetts_v4/test"
if [ -d "$TEST_DIR" ] && [ -f "$TEST_DIR/manifest.json" ]; then
    python3 /workspace/scripts/mww/infer_verify_v3.py \
        --test-dir "$TEST_DIR" \
        --model "${OUTPUT_DIR}/${KEYWORD_ID}.tflite" \
        --cutoff 0.80 --window 3 2>&1 || true
fi

log "=== help help v4 complete ==="

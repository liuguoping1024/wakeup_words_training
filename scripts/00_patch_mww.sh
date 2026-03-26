#!/usr/bin/env bash
# Patch micro-wake-word for TF 2.16 / numpy 1.26 compatibility.
# model.evaluate() returns plain np.ndarray in TF 2.16, so .numpy() fails.
# Replace .numpy() calls with np.asarray() which works for both Tensor and ndarray.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

TRAIN_PY="${MWW_DIR}/microwakeword/train.py"

if grep -q "np.asarray(result\[\"fp\"\])" "${TRAIN_PY}" 2>/dev/null; then
  log "train.py already patched, skipping."
  exit 0
fi

log "Patching microwakeword/train.py for TF 2.16 numpy compatibility..."

sed -i \
  's/result\["fp"\]\.numpy()/np.asarray(result["fp"])/g;
   s/ambient_predictions\["tp"\]\.numpy()/np.asarray(ambient_predictions["tp"])/g;
   s/ambient_predictions\["fp"\]\.numpy()/np.asarray(ambient_predictions["fp"])/g;
   s/ambient_predictions\["fn"\]\.numpy()/np.asarray(ambient_predictions["fn"])/g' \
  "${TRAIN_PY}"

log "Patch applied to train.py."

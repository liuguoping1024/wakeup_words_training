#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/outputs/help_me.tflite}"
POS_DIR="${POS_DIR:-${ROOT_DIR}/data/real_voices}"

# 默认值尽量对齐 detect.py 的默认参数：cutoff=0.10，window=5
CUTOFF="${CUTOFF:-0.10}"
WINDOW="${WINDOW:-5}"

# 如果你要顺便测误触发（更慢），可以传 NEG_DIR/ MAX_NEG
NEG_DIR="${NEG_DIR:-}"
MAX_NEG="${MAX_NEG:-300}"

# 推理环境（A113X 上建议先跑 inference/setup_venv.sh）
VENV_PY="${VENV_PY:-${ROOT_DIR}/inference/.venv/bin/python}"
if [[ ! -f "${VENV_PY}" ]]; then
  echo "[warn] 找不到推理 venv: ${VENV_PY}"
  echo "[warn] 将使用当前 python3 执行（可能缺少依赖）"
  PYTHON="python3"
else
  PYTHON="${VENV_PY}"
fi

echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] POS_DIR=${POS_DIR}"
echo "[info] CUTOFF=${CUTOFF} WINDOW=${WINDOW}"
if [[ -n "${NEG_DIR}" ]]; then
  echo "[info] NEG_DIR=${NEG_DIR} MAX_NEG=${MAX_NEG}"
fi
echo ""

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] 模型不存在: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -d "${POS_DIR}" ]]; then
  echo "[error] 正样本目录不存在: ${POS_DIR}"
  exit 1
fi

ARGS=(
  "${ROOT_DIR}/scripts/eval_model.py"
  --model "${MODEL_PATH}"
  --pos "${POS_DIR}"
  --cutoff "${CUTOFF}"
  --window "${WINDOW}"
)

if [[ -n "${NEG_DIR}" ]]; then
  if [[ ! -d "${NEG_DIR}" ]]; then
    echo "[error] 负样本目录不存在: ${NEG_DIR}"
    exit 1
  fi
  ARGS+=(--neg "${NEG_DIR}" --max-neg "${MAX_NEG}")
fi

set +e
"${PYTHON}" "${ARGS[@]}"
rc=$?
set -e

exit $rc


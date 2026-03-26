#!/usr/bin/env bash
# 在任意 Linux/macOS 机器上建立推理虚拟环境
# 支持 x86_64 和 aarch64（树莓派、Jetson 等）
set -euo pipefail

VENV="${1:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "[info] 使用 Python: $(${PYTHON} --version)"
echo "[info] 创建虚拟环境: ${VENV}"
${PYTHON} -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install -q --upgrade pip

ARCH="$(uname -m)"
OS="$(uname -s)"

echo "[info] 平台: ${OS}/${ARCH}"

# ── TFLite 后端（纯 CPU，按优先级安装）──────────────────────────────────────
# ai-edge-litert：Google 官方轻量包，~30MB，支持 x86_64 / aarch64
# tensorflow-cpu：万能兜底，~300MB，任何平台均可用
echo "[info] 安装 TFLite 后端（CPU only）..."
if pip install -q "ai-edge-litert>=1.0.0"; then
  echo "[ok]  TFLite 后端: ai-edge-litert（~30MB）"
else
  echo "[warn] ai-edge-litert 不可用，回退到 tensorflow-cpu（~300MB）..."
  pip install -q "tensorflow-cpu"
  echo "[ok]  TFLite 后端: tensorflow-cpu"
fi

# ── 频谱特征提取 ─────────────────────────────────────────────────────────────
if pip install -q "pymicro-features>=2.0.0" 2>/dev/null; then
  echo "[ok]  前端: pymicro-features"
else
  echo "[warn] pymicro-features 安装失败，尝试从源码安装..."
  pip install -q "git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version"
  echo "[ok]  前端: pymicro-features (from source)"
fi

# ── 其他依赖 ─────────────────────────────────────────────────────────────────
pip install -q "numpy>=1.23,<2.0" soundfile resampy
echo "[ok]  numpy / soundfile / resampy"

# ── 实时麦克风（必需）────────────────────────────────────────────────────────
echo "[info] 安装 portaudio 系统依赖..."
apt-get install -y portaudio19-dev -qq 2>/dev/null || \
  echo "[warn] 无法自动安装 portaudio19-dev，如失败请手动执行：apt-get install -y portaudio19-dev"

if pip install -q pyaudio; then
  echo "[ok]  pyaudio（实时麦克风可用）"
else
  echo "[error] pyaudio 安装失败，请先执行：apt-get install -y portaudio19-dev"
fi

echo ""
echo "=========================================="
echo "  虚拟环境准备完成：${VENV}"
echo "=========================================="
echo ""
echo "激活方式:"
echo "  source ${VENV}/bin/activate"
echo ""
echo "测试 WAV 文件（把 help_me.tflite 放到同目录）:"
echo "  python detect.py --model help_me.tflite --wav test.wav --verbose"
echo ""
echo "批量测试目录下所有 WAV:"
echo "  python detect.py --model help_me.tflite --wav-dir ./samples/"
echo ""
echo "实时麦克风:"
echo "  python detect.py --model help_me.tflite"

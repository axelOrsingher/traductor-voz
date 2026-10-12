#!/usr/bin/env bash
# Create the Python 3.11 uv venv used by the Linux recon scripts (GPU benchmark,
# global-hotkey test, overlay test). Reproduces exactly what docs/porting/linux-recon.md
# measured on the XPS 15 (GTX 1650 Ti, CUDA 12.9 driver).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/../../.venv-recon}"

command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv venv --python 3.11 "$VENV"
. "$VENV/bin/activate"

uv pip install numpy soundfile librosa
uv pip install ctranslate2 transformers sentencepiece
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough to convert opus-mt
uv pip install "onnx-asr[hub]" silero-vad evdev pynput pyside6

# IMPORTANT: onnxruntime-gpu 1.31.x is a CUDA 13 build and fails on a CUDA 12 box
# with "libcublasLt.so.13: cannot open shared object file". Pin the CUDA 12 build:
uv pip install "onnxruntime-gpu==1.22.0"

# CUDA 12 runtime libs that ctranslate2 + onnxruntime-gpu dlopen at runtime:
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 \
               nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12

echo "venv ready at $VENV"
echo "run: scripts/linux/gpu_bench.sh"

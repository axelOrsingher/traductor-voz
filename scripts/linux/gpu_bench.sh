#!/usr/bin/env bash
# Wrapper that points the dynamic loader at the pip-installed CUDA 12 libraries
# (cuBLAS / cuDNN / cuRAND / cuFFT / cudart) before running gpu_bench.py.
#
# LD_LIBRARY_PATH MUST be set before the Python process starts: both CTranslate2
# and onnxruntime-gpu dlopen these at session-creation time using the loader's
# startup search path, so exporting it from inside Python is too late.
#
# Requires a uv venv (Python 3.11) with:
#   ctranslate2 transformers sentencepiece torch(cpu) soundfile silero-vad "onnx-asr[hub]"
#   onnxruntime-gpu==1.22.0  (the CUDA 12 build -- newer 1.31 is CUDA 13 only)
#   nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
#   nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/../../.venv-recon}"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/curand/lib:$SP/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec python "$HERE/gpu_bench.py" "$@"

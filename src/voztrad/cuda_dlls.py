"""Registra los directorios de DLLs de CUDA 12 / cuDNN 9 en Windows.

Desde Python 3.8 las dependencias de extensiones no se buscan en PATH, así que
se agregan explícitamente con os.add_dll_directory. También se anteponen al PATH
porque CTranslate2 carga cuDNN/cuBLAS en forma diferida con LoadLibrary.

cudart se PRECARGA por ruta absoluta: onnxruntime-gpu 1.29 (build CUDA 12) usa
símbolos de cudart >= 12.8 que el toolkit 12.6 no tiene, y el orden entre
directorios de add_dll_directory no está garantizado. Una DLL ya cargada gana
sobre cualquier búsqueda por nombre posterior.

Importar este módulo ANTES que ctranslate2 / faster_whisper / onnxruntime.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

TOOLS = Path.home() / "tools"

# Se pueden sobreescribir con variables de entorno.
CUDART_BIN = Path(os.environ.get("VOZTRAD_CUDART_BIN", TOOLS / "cuda_cudart-12.9.79" / "bin"))
CUDNN_BIN = Path(os.environ.get("VOZTRAD_CUDNN_BIN", TOOLS / "cudnn-9.27.0.42_cuda12" / "bin" / "x64"))
CUDA_BIN = Path(os.environ.get(
    "VOZTRAD_CUDA_BIN",
    Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")) / "bin",
))

_handles = []


def setup() -> list[Path]:
    if sys.platform != "win32":
        return []
    added = []
    for d in (CUDART_BIN, CUDNN_BIN, CUDA_BIN):
        if d.is_dir():
            _handles.append(os.add_dll_directory(str(d)))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(d)
    cudart = CUDART_BIN / "cudart64_12.dll"
    if cudart.is_file():
        _handles.append(ctypes.WinDLL(str(cudart)))
    return added


ADDED = setup()

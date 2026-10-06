"""Convierte el encoder de Parakeet TDT 0.6B v3 (ONNX) a fp16, sin PyTorch.

Necesita `onnx` + `onnxruntime` (CPU) en un venv APARTE: en el del proyecto el override
de pyproject.toml bloquea onnxruntime de CPU (pisaría al de GPU). Desde una carpeta fuera del
proyecto:

    uv venv conv && uv pip install --python conv/Scripts/python.exe onnx onnxruntime huggingface_hub
    conv/Scripts/python.exe <proyecto>/scripts/convert_parakeet_fp16.py

Usa el conversor de ONNX Runtime (el de onnxconverter-common deja un Add fp32/fp16 mezclado
en /pre_encode y el modelo no carga). Resultado medido: 1,7 GB de VRAM (vs 3,1-3,5),
55/57 transcripciones idénticas a fp32.
"""

import shutil
from pathlib import Path

import onnx
from huggingface_hub import snapshot_download
from onnxruntime.transformers.float16 import convert_float_to_float16

OUT = Path(__file__).resolve().parents[1] / "models" / "parakeet-tdt-0.6b-v3-fp16"
src = Path(snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx",
                             allow_patterns=["encoder-model.onnx*", "decoder_joint-model.onnx",
                                             "config.json", "vocab.txt"]))
OUT.mkdir(parents=True, exist_ok=True)
for f in ("decoder_joint-model.onnx", "config.json", "vocab.txt"):
    shutil.copy(src / f, OUT / f)
m16 = convert_float_to_float16(onnx.load(src / "encoder-model.onnx"), keep_io_types=True,
                               disable_shape_infer=True)
onnx.save_model(m16, OUT / "encoder-model.onnx", save_as_external_data=True,
                all_tensors_to_one_file=True, location="encoder-model.onnx.data")
print("ok ->", OUT)

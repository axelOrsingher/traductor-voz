"""Smoke test de Parakeet TDT 0.6B v3 vía ONNX (onnx-asr) sobre CUDA.

Uso: python scripts/smoke_parakeet.py [wav] [--int8]
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voztrad import cuda_dlls  # noqa: E402  (antes que onnxruntime)

import onnx_asr  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
wav = args[0] if args else str(ROOT / "samples" / "en_smoke.wav")
quant = "int8" if "--int8" in sys.argv else None

print("DLL dirs:", [str(p) for p in cuda_dlls.ADDED])
ort.set_default_logger_severity(3)

t = time.perf_counter()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization=quant,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"load ({quant or 'fp32'}): {(time.perf_counter() - t) * 1000:.0f} ms")

audio, sr = sf.read(wav, dtype="float32")
for i in range(3):
    t = time.perf_counter()
    text = model.recognize(audio, sample_rate=sr)
    print(f"run {i}: {(time.perf_counter() - t) * 1000:.0f} ms | audio {len(audio) / sr:.2f}s | {text}")

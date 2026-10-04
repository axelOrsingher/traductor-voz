"""Paso 2: smoke test de CTranslate2 en GPU con faster-whisper.

Uso: python scripts/smoke_gpu.py [wav] [modelo]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voztrad import cuda_dlls  # noqa: E402  (antes que ctranslate2)

import ctranslate2  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

wav = sys.argv[1] if len(sys.argv) > 1 else "samples/en_smoke.wav"
model_name = sys.argv[2] if len(sys.argv) > 2 else "base"

print("DLL dirs:", [str(p) for p in cuda_dlls.ADDED])
print("ctranslate2", ctranslate2.__version__, "| CUDA devices:", ctranslate2.get_cuda_device_count())
print("compute types cuda:", sorted(ctranslate2.get_supported_compute_types("cuda")))

t = time.perf_counter()
model = WhisperModel(model_name, device="cuda", compute_type="float16")
print(f"load {model_name}: {(time.perf_counter() - t) * 1000:.0f} ms")

for i in range(3):  # la 1ª pasada incluye el warm-up de CUDA/cuDNN
    t = time.perf_counter()
    segments, info = model.transcribe(wav, language="en", beam_size=1,
                                      condition_on_previous_text=False, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments)
    print(f"run {i}: {(time.perf_counter() - t) * 1000:.0f} ms | audio {info.duration:.2f}s | {text}")

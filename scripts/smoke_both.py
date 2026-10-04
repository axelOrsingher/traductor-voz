"""Ambos backends de ASR (CTranslate2 + onnxruntime) en GPU dentro del mismo proceso."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voztrad import cuda_dlls  # noqa: E402,F401

import onnx_asr  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

ort.set_default_logger_severity(3)


def vram() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip() + " MiB"


audio, sr = sf.read(ROOT / "samples" / "en_smoke.wav", dtype="float32")
print("VRAM inicial:", vram())

t = time.perf_counter()
wh = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
print(f"whisper turbo cargado en {time.perf_counter() - t:.1f}s | VRAM {vram()}")

t = time.perf_counter()
pk = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",
                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"parakeet cargado en {time.perf_counter() - t:.1f}s | VRAM {vram()}")

for name, fn in [
    ("whisper", lambda a: " ".join(s.text.strip() for s in wh.transcribe(
        a, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=False)[0])),
    ("parakeet", lambda a: pk.recognize(a, sample_rate=16000)),
]:
    for i in range(4):
        t = time.perf_counter()
        text = fn(audio)
        print(f"{name:8} run {i}: {(time.perf_counter() - t) * 1000:6.0f} ms | {text}")
print("VRAM final:", vram())

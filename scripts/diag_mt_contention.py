"""Diagnóstico: ¿el MT (CT2 CPU) se frena después de una inferencia de ASR?"""

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voztrad import cuda_dlls  # noqa: E402,F401

import onnx_asr  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402

from voztrad.mt import Translator  # noqa: E402

ort.set_default_logger_severity(3)
S = "Mira, yo lo que haría es verificar primero si el cliente ya lo tiene."
audio, _ = sf.read(ROOT / "samples" / "synth" / "B_rio4.wav", dtype="float32")
tr = Translator("es", "en")
tr.translate(S)


def mt_after(fn, label, n=6):
    ms = []
    for _ in range(n):
        if fn:
            fn()
        t = time.perf_counter()
        tr.translate(S)
        ms.append((time.perf_counter() - t) * 1000)
    print(f"{label:45} MT p50 {statistics.median(ms):5.0f} ms  max {max(ms):5.0f}")


mt_after(None, "MT solo")
mt_after(lambda: time.sleep(0.3), "MT tras 300 ms idle")

for spin in ("1", "0"):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.intra_op.allow_spinning", spin)
    pk = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", sess_options=so,
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pk.recognize(audio, sample_rate=16000)
    mt_after(lambda: pk.recognize(audio, sample_rate=16000), f"MT tras parakeet (spinning={spin})")
    del pk

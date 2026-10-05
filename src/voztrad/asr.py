"""Backends de ASR enchufables. Ambos reciben float32 16 kHz mono.

- whisper: faster-whisper (CTranslate2, float16). Padea SIEMPRE a 30 s.
- parakeet: Parakeet TDT 0.6B v3 vía onnx-asr (ONNX, GPU). Procesa solo el
  audio que hay. Detecta el idioma solo (no admite forzarlo).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from . import cuda_dlls  # noqa: F401  (antes que ctranslate2 / onnxruntime)
from .config import SR


class ASR(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, lang: str) -> str: ...


class WhisperASR:
    def __init__(self, model: str = "large-v3-turbo"):
        from faster_whisper import WhisperModel
        self.name = f"whisper-{model}"
        self.model = WhisperModel(model, device="cuda", compute_type="float16")

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language=lang, beam_size=1, condition_on_previous_text=False,
            vad_filter=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segments).strip()


class ParakeetASR:
    def __init__(self, quantization: str | None = None):
        import onnx_asr
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        # Sin spin-wait: el pool de ORT queda girando tras cada inferencia y le roba
        # CPU al MT (medido: MT p50 131 ms / max 898 ms con spinning, 86 / 94 sin).
        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.name = f"parakeet-tdt-0.6b-v3-{quantization or 'fp32'}"
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", quantization=quantization, sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        text = str(self.model.recognize(audio, sample_rate=SR)).strip()
        # Parakeet pega oraciones: "amor.¿estás" -> "amor. ¿estás"
        text = _GLUED.sub(r"\1 \2", text)
        return _OPEN_SPACE.sub(r"\1", text)  # "¿ cómo" -> "¿cómo"


_GLUED = __import__("re").compile(r"([.,;:?!])([¿¡A-Za-zÁÉÍÓÚÑáéíóúñ])")
_OPEN_SPACE = __import__("re").compile(r"([¿¡])\s+")


def load(kind: str, whisper_model: str = "large-v3-turbo", parakeet_quant: str | None = None) -> ASR:
    asr = WhisperASR(whisper_model) if kind == "whisper" else ParakeetASR(parakeet_quant)
    asr.transcribe(np.zeros(SR, dtype=np.float32), "en")  # warm-up: la 1ª pasada CUDA tarda 2-3 s de más
    return asr

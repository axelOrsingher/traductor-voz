"""Backends de ASR enchufables. Ambos reciben float32 16 kHz mono.

- whisper: faster-whisper (CTranslate2, float16). Padea SIEMPRE a 30 s.
- parakeet: Parakeet TDT 0.6B v3 vía onnx-asr (ONNX, GPU). Procesa solo el
  audio que hay. Detecta el idioma solo (no admite forzarlo).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from . import cuda_dlls  # noqa: F401  (antes que ctranslate2 / onnxruntime)
from .config import MODELS, SR


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
        # fp16: encoder convertido sin torch (scripts/convert_parakeet_fp16.py). Medido: VRAM
        # 1,7 GB vs 3,1-3,5 GB en fp32, 55/57 transcripciones idénticas, +30 ms. Hace lugar
        # para TranslateGemma (refine.py) en los 6 GB.
        path = None
        if quantization == "fp16":
            path = FP16_DIR if (FP16_DIR / "encoder-model.onnx").exists() else None
            quantization = None
            self.name = "parakeet-tdt-0.6b-v3-" + ("fp16" if path else "fp32")
        else:
            self.name = f"parakeet-tdt-0.6b-v3-{quantization or 'fp32'}"
        cuda = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", path=path, quantization=quantization, sess_options=so,
            providers=[cuda, "CPUExecutionProvider"])

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        text = str(self.model.recognize(audio, sample_rate=SR)).strip()
        # Parakeet pega oraciones: "amor.¿estás" -> "amor. ¿estás"
        text = _GLUED.sub(r"\1 \2", text)
        return _OPEN_SPACE.sub(r"\1", text)  # "¿ cómo" -> "¿cómo"


FP16_DIR = MODELS / "parakeet-tdt-0.6b-v3-fp16"
_GLUED = __import__("re").compile(r"([.,;:?!])([¿¡A-Za-zÁÉÍÓÚÑáéíóúñ])")
_OPEN_SPACE = __import__("re").compile(r"([¿¡])\s+")


def load(kind: str, whisper_model: str = "large-v3-turbo", parakeet_quant: str | None = None) -> ASR:
    asr = WhisperASR(whisper_model) if kind == "whisper" else ParakeetASR(parakeet_quant)
    asr.transcribe(np.zeros(SR, dtype=np.float32), "en")  # warm-up: la 1ª pasada CUDA tarda 2-3 s de más
    return asr

"""Parámetros del motor. Todo lo que afecta latencia es configurable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

VOICE_REFS = [MODELS / "voice" / "me.wav", MODELS / "voice" / "preview_from_call.wav"]

SR = 16000          # todo el pipeline interno trabaja a 16 kHz mono
FRAME = 512         # Silero VAD v5+: EXACTAMENTE 512 muestras a 16 kHz
FRAME_MS = FRAME * 1000 / SR  # 32 ms


@dataclass
class SegmenterConfig:
    enter_prob: float = 0.50       # IDLE -> SPEAKING
    enter_frames: int = 2          # frames seguidos sobre enter_prob (64 ms)
    exit_prob: float = 0.35        # SPEAKING -> EMIT si la prob queda por debajo...
    hangover_ms: int = 500         # ...durante este tiempo
    min_speech_ms: int = 250       # descartar tos / click / "ajá"
    max_segment_ms: int = 8000     # corte forzado
    overlap_ms: int = 200          # solape arrastrado tras un corte forzado
    preroll_frames: int = 10       # 320 ms volcados al entrar en SPEAKING
    keep_tail_ms: int = 96         # silencio final que se deja en el segmento


@dataclass
class Config:
    asr: str = "whisper"                     # whisper | parakeet
    whisper_model: str = "large-v3-turbo"
    parakeet_quant: str | None = None        # None (fp32) | "int8"
    b_mode: str = "ptt"                      # ptt | vad
    ptt_key: str = "F8"
    ptt_max_ms: int = 28000
    normalize_voseo: bool = True
    mic: str | None = None                   # substring del nombre; None = default
    loopback: str | None = None              # substring; None = loopback de la salida default
    cable: str = "CABLE Input"               # salida hacia la que se escribe el TTS
    tts: str = "pocket"                      # pocket (tu voz clonada) | piper
    tts_voice: Path = MODELS / "en_US-lessac-medium.onnx"   # voz Piper (fallback)
    voice_ref: Path | None = None            # grabación de tu voz; None = primera de VOICE_REFS que exista
    a_queue_max: int = 2                     # backpressure de la dirección A
    host: str = "127.0.0.1"
    port: int = 8765                         # transporte JSON-lines hacia la UI
    enable_a: bool = True
    enable_b: bool = True
    gpu_warm: bool = True                    # cargar la GPU mientras hay voz (ver gpuwarm.py)
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)

"""TTS con Piper (ONNX, CPU). Entrega audio de a una oración para arrancar antes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from piper import PiperVoice


def load_tts(cfg, status=print):
    """Pocket (voz clonada) si hay grabación de referencia; si no, Piper."""
    if cfg.tts == "pocket":
        from .config import VOICE_REFS
        ref = cfg.voice_ref or next((p for p in VOICE_REFS if p.exists()), None)
        if ref is None or not Path(ref).exists():
            status("no hay grabación de tu voz (models/voice/me.wav): uso Piper")
        else:
            from .tts_pocket import PocketTTS
            tts = PocketTTS(Path(ref))
            next(iter(tts.stream("Ready.")), None)  # warm-up
            status(f"voz clonada desde {Path(ref).name} ({tts.voice_seconds:.0f} s)")
            tts.name = f"pocket:{Path(ref).stem}"
            return tts
    tts = Piper(cfg.tts_voice)
    tts.name = f"piper:{Path(cfg.tts_voice).stem}"
    return tts


class Piper:
    def __init__(self, model: Path):
        self.voice = PiperVoice.load(str(model))
        self.sr = self.voice.config.sample_rate
        next(iter(self.voice.synthesize("Ready.")))  # warm-up

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self.voice.synthesize(text):
            yield chunk.audio_float_array

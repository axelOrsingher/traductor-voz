"""TTS con Piper (ONNX, CPU). Entrega audio de a una oración para arrancar antes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from piper import PiperVoice


class Piper:
    def __init__(self, model: Path):
        self.voice = PiperVoice.load(str(model))
        self.sr = self.voice.config.sample_rate
        next(iter(self.voice.synthesize("Ready.")))  # warm-up

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self.voice.synthesize(text):
            yield chunk.audio_float_array

"""Fin de turno "inteligente": Pipecat Smart Turn v3.2 (ONNX, CPU, 8,7 MB, BSD-2).

Mira el audio de la frase (últimos 8 s) y decide si la persona TERMINÓ o solo hizo una
pausa. Así se puede cerrar la frase con ~200 ms de silencio en vez de esperar 500 ms, sin
partir frases a la mitad. Benchmark oficial v3.2: castellano 89,6%, global 92,6%.

Las features son las de WhisperFeatureExtractor de transformers (chunk_length=8,
do_normalize=True, padding a 8 s a la derecha), portadas a numpy para no depender de
transformers. Verificado contra el original: ver tests/test_smartturn.py.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from faster_whisper.feature_extractor import FeatureExtractor

from .config import MODELS, SR

N_FFT, HOP, N_MELS, SECONDS = 400, 160, 80, 8
N_SAMPLES = SECONDS * SR


class SmartTurn:
    def __init__(self, path=MODELS / "smart-turn-v3.2-cpu.onnx"):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = 1   # no competir con los hilos de audio
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.mel = FeatureExtractor.get_mel_filters(SR, N_FFT, n_mels=N_MELS).astype(np.float32)
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # hann periódica
        self(np.zeros(SR, np.float32))  # warm-up

    def features(self, audio: np.ndarray) -> np.ndarray:
        x = audio[-N_SAMPLES:].astype(np.float32)
        n = len(x)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)            # do_normalize (solo la parte real)
        x = np.concatenate([x, np.zeros(N_SAMPLES - n, np.float32)])  # padding="max_length" a la derecha
        x = np.pad(x, N_FFT // 2, mode="reflect")                # STFT center=True
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
        spec = np.abs(np.fft.rfft(frames * self.win, axis=1)) ** 2   # (frames, 201)
        mel = self.mel @ spec.T                                   # (80, frames)
        log = np.log10(np.maximum(mel, 1e-10))[:, :-1]
        log = np.maximum(log, log.max() - 8.0)
        return ((log + 4.0) / 4.0).astype(np.float32)             # (80, 800)

    def __call__(self, audio: np.ndarray) -> float:
        """Probabilidad de que la frase esté COMPLETA (>0,5 = terminó)."""
        out = self.sess.run(None, {"input_features": self.features(audio)[None]})[0]
        return float(np.asarray(out).reshape(-1)[0])

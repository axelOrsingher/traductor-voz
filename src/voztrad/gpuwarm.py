"""Calienta la GPU mientras alguien habla, para que el ASR la encuentre a clock alto.

Medido en esta laptop (RTX 4050, driver 596): tras >2 s ociosa la GPU cae a
P8 / 210 MHz y el ASR se vuelve lento (parakeet 44 -> ~280 ms, whisper 342 ->
~595 ms). Un keep-alive liviano y periódico NO alcanza (el driver lo ignora),
pero 0,6 s de carga continua sí. Así que se carga la GPU solo mientras hay voz
(VAD en SPEAKING o PTT apretado) y se pausa mientras corre el ASR.

Alternativa a nivel sistema (requiere admin): `nvidia-smi -lgc 1500,3105`.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np
import onnxruntime as ort

from .config import MODELS


class GpuWarmer:
    BATCH = 2048  # ~10 ms de GPU por llamada: poca espera si el ASR arranca justo

    def __init__(self, hold_s: float = 0.3):
        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # Silero VAD en CUDA con batch grande: modelo que ya está en disco, carga trivial.
        self.sess = ort.InferenceSession(str(MODELS / "silero_vad.onnx"), so,
                                         providers=["CUDAExecutionProvider"])
        b = self.BATCH
        self.inp = {"input": np.zeros((b, 576), np.float32),
                    "state": np.zeros((2, b, 128), np.float32),
                    "sr": np.array(16000, np.int64)}
        self.sess.run(None, self.inp)
        self.hold_s = hold_s
        self._until = 0.0
        # trabajos reales de GPU en curso (ASR, MT, parciales): contador, pueden superponerse
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._wake = threading.Event()
        self.calls = 0
        threading.Thread(target=self._run, name="gpuwarm", daemon=True).start()

    def poke(self):
        """Llamar en cada frame con voz: mantiene la carga hold_s más."""
        self._until = time.perf_counter() + self.hold_s
        self._wake.set()

    def pause(self):
        with self._busy_lock:
            self._busy += 1

    def resume(self):
        with self._busy_lock:
            self._busy = max(0, self._busy - 1)

    @contextmanager
    def paused(self):
        """Mientras corre un trabajo real de GPU, el calentador no compite."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            while time.perf_counter() < self._until:
                if self._busy:
                    time.sleep(0.005)
                    continue
                self.sess.run(None, self.inp)
                self.calls += 1

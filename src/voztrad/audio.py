"""Audio I/O. Platform-neutral framing (Frame/Framer) lives here; the capture /
playback backend is selected at import time:

- Windows: WASAPI via PyAudioWPatch (audio_win.py)
- Linux:   PipeWire via parec/pacat (audio_linux.py)

Both backends expose the same surface used by the engine:
    PyAudio, Capture, CablePlayer, find_input, find_loopback, find_output
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import soxr

from .config import FRAME, SR


@dataclass
class Frame:
    samples: np.ndarray  # float32, FRAME samples @ 16 kHz mono
    t_cb: float          # perf_counter of the callback that completed the frame
    t_ready: float       # perf_counter when the frame became available


class Framer:
    """Downmix + streaming resample (soxr) + fixed 512-sample frames."""

    def __init__(self, rate: int, channels: int):
        self.channels = channels
        self._rs = soxr.ResampleStream(rate, SR, 1, dtype="float32") if rate != SR else None
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, t_cb: float, data: bytes) -> list[Frame]:
        x = np.frombuffer(data, dtype=np.float32)
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        if self._rs is not None:
            x = self._rs.resample_chunk(x)
        self._buf = np.concatenate((self._buf, x))
        n = len(self._buf) // FRAME
        if not n:
            return []
        now = time.perf_counter()
        frames = [Frame(self._buf[i * FRAME:(i + 1) * FRAME].copy(), t_cb, now) for i in range(n)]
        self._buf = self._buf[n * FRAME:]
        return frames


if sys.platform == "win32":
    from .audio_win import (  # noqa: F401,E402
        CablePlayer, Capture, PyAudio, find_default_output, find_input, find_loopback, find_output)
else:
    from .audio_linux import (  # noqa: F401,E402
        CablePlayer, Capture, PyAudio, find_default_output, find_input, find_loopback, find_output)

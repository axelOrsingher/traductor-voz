"""Captura WASAPI (mic / loopback), framing a 512 @ 16 kHz y salida a VB-Cable.

El callback de PortAudio SOLO hace put() en una cola: si se demora, mete
glitches en el audio real de la llamada.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from .config import FRAME, SR


def _wasapi_devices(pa: pyaudio.PyAudio):
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["hostApi"] == api:
            yield d


def find_input(pa: pyaudio.PyAudio, name: str | None) -> dict:
    if name is None:
        api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return pa.get_device_info_by_index(api["defaultInputDevice"])
    for d in _wasapi_devices(pa):
        if name in d["name"] and d["maxInputChannels"] and not d.get("isLoopbackDevice"):
            return d
    raise RuntimeError(f"no hay entrada WASAPI que contenga {name!r}")


def find_loopback(pa: pyaudio.PyAudio, name: str | None) -> dict:
    if name is None:
        return pa.get_default_wasapi_loopback()
    for d in _wasapi_devices(pa):
        if name in d["name"] and d.get("isLoopbackDevice"):
            return d
    raise RuntimeError(f"no hay loopback WASAPI que contenga {name!r}")


def find_output(pa: pyaudio.PyAudio, name: str) -> dict:
    for d in _wasapi_devices(pa):
        if name in d["name"] and d["maxOutputChannels"] and not d.get("isLoopbackDevice"):
            return d
    raise RuntimeError(f"no hay salida WASAPI que contenga {name!r}")


class Capture:
    """Stream de entrada. Los bloques crudos van a self.q como (t_callback, bytes)."""

    def __init__(self, pa: pyaudio.PyAudio, device: dict):
        self.device = device
        self.rate = int(device["defaultSampleRate"])
        self.channels = max(1, int(device["maxInputChannels"]))
        self.q: queue.Queue[tuple[float, bytes]] = queue.Queue()
        self._stream = pa.open(
            format=pyaudio.paFloat32, channels=self.channels, rate=self.rate, input=True,
            input_device_index=device["index"], frames_per_buffer=self.rate // 100,  # 10 ms
            stream_callback=self._cb, start=False)

    def _cb(self, in_data, frame_count, time_info, status):
        self.q.put((time.perf_counter(), in_data))
        return (None, pyaudio.paContinue)

    def start(self):
        self._stream.start_stream()

    def close(self):
        self._stream.stop_stream()
        self._stream.close()


@dataclass
class Frame:
    samples: np.ndarray  # float32, FRAME muestras @ 16 kHz mono
    t_cb: float          # perf_counter del callback que completó el frame
    t_ready: float       # perf_counter cuando el frame quedó disponible


class Framer:
    """Downmix + resample streaming (soxr) + frames fijos de 512 muestras."""

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


class CablePlayer:
    """Escribe audio hacia VB-Cable ("CABLE Input"). Escritura bloqueante, solo desde el hilo TTS."""

    def __init__(self, pa: pyaudio.PyAudio, device: dict):
        self.device = device
        self.rate = int(device["defaultSampleRate"])
        self.channels = min(2, int(device["maxOutputChannels"]))
        self._stream = pa.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate,
                               output=True, output_device_index=device["index"],
                               frames_per_buffer=self.rate // 100)

    def play(self, audio: np.ndarray, sr: int):
        if sr != self.rate:
            audio = soxr.resample(audio, sr, self.rate)
        audio = np.repeat(audio.astype(np.float32)[:, None], self.channels, axis=1)
        self._stream.write(audio.tobytes())

    def close(self):
        self._stream.close()

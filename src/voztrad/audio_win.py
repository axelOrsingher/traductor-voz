"""Windows audio backend: WASAPI capture (mic / loopback) + output to VB-Cable.

The PortAudio callback ONLY does put() on a queue: if it blocks, it glitches the
real call audio. Shared helpers (Frame, Framer) live in audio.py."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

PyAudio = pyaudio.PyAudio


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


def find_default_output(pa: pyaudio.PyAudio) -> dict:
    api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    return pa.get_device_info_by_index(api["defaultOutputDevice"])


class Capture:
    """Input stream. Raw blocks go to self.q as (t_callback, bytes)."""

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


class CablePlayer:
    """Writes audio to VB-Cable ("CABLE Input"). Blocking write, from the TTS thread only."""

    def __init__(self, pa: pyaudio.PyAudio, device: dict):
        self.device = device
        self.rate = int(device["defaultSampleRate"])
        self.channels = min(2, int(device["maxOutputChannels"]))
        self._stream = pa.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate,
                               output=True, output_device_index=device["index"],
                               frames_per_buffer=self.rate // 100)

        self._q: queue.Queue = queue.Queue()
        self._queued = 0
        self._lock = threading.Lock()
        self._rs: soxr.ResampleStream | None = None
        self._rs_sr = 0
        threading.Thread(target=self._writer, name="cable-out", daemon=True).start()

    def play(self, audio: np.ndarray, sr: int, last: bool = True):
        audio = audio.astype(np.float32)
        if sr != self.rate:
            if self._rs is None or self._rs_sr != sr:
                self._rs, self._rs_sr = soxr.ResampleStream(sr, self.rate, 1, dtype="float32"), sr
            audio = self._rs.resample_chunk(audio, last=last)
            if last:
                self._rs = None
        if len(audio):
            with self._lock:
                self._queued += len(audio)
            self._q.put((len(audio), np.repeat(audio[:, None], self.channels, axis=1).tobytes()))

    @property
    def queued_s(self) -> float:
        with self._lock:
            return self._queued / self.rate

    def _writer(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            n, data = item
            self._stream.write(data)
            with self._lock:
                self._queued -= n

    def close(self):
        self._q.put(None)
        self._stream.close()

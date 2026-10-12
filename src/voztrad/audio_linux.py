"""Linux audio backend over PipeWire using the PulseAudio-compat raw tools
(`parec` / `pacat`). Capture streams run at 16 kHz mono float32 directly, so the
Framer does no resampling. Exposes the same surface as audio_win.py.

Device descriptors are plain dicts: {"name": <human>, "target": <pulse node>}.
The virtual devices come from scripts/linux/voztrad-audio.sh (voztrad_calls /
voztrad_mic / voztrad_mic_input)."""

from __future__ import annotations

import queue
import subprocess
import threading
import time

import numpy as np
import soxr

from .config import SR

CALLS_MONITOR = "voztrad_calls.monitor"   # system/call audio routed here
MIC_SINK = "voztrad_mic"                   # engine writes translated TTS here
_CHUNK = SR * 4 // 100                      # 10 ms of float32 mono


class PyAudio:
    """Dummy to keep the engine's `self.pa = PyAudio()` call portable."""

    def terminate(self):
        pass


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def find_input(pa, name: str | None) -> dict:
    """The microphone (direction B). None -> default source."""
    target = name or _pactl("get-default-source") or "@DEFAULT_SOURCE@"
    return {"name": target, "target": target}


def find_loopback(pa, name: str | None) -> dict:
    """What the other side says (direction A) = the call sink's monitor."""
    target = name or CALLS_MONITOR
    return {"name": target, "target": target}


def find_output(pa, name: str) -> dict:
    """Where the TTS is written. The Windows default is "CABLE Input"; on Linux
    that maps to the voztrad_mic null-sink the call app reads as a microphone."""
    target = MIC_SINK if (name in (None, "CABLE Input")) else name
    return {"name": target, "target": target}


class Capture:
    """`parec` subprocess -> self.q as (t, bytes) at 16 kHz mono float32."""

    def __init__(self, pa, device: dict):
        self.device = device
        self.rate = SR
        self.channels = 1
        self.q: queue.Queue[tuple[float, bytes]] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._stop = False

    def _reader(self):
        assert self._proc and self._proc.stdout
        while not self._stop:
            data = self._proc.stdout.read(_CHUNK)
            if not data:
                break
            self.q.put((time.perf_counter(), data))

    def start(self):
        cmd = ["parec", "--raw", f"--rate={SR}", "--channels=1", "--format=float32le",
               "-d", self.device["target"], "--latency-msec=20"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=0)
        threading.Thread(target=self._reader, name="parec-in", daemon=True).start()

    def close(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except Exception:
                self._proc.kill()


class CablePlayer:
    """`pacat` subprocess fed from a queue. queued_s is estimated from a running
    playback-end timestamp (good enough for tempo/gating; no per-sample tracking
    is possible through a subprocess)."""

    def __init__(self, pa, device: dict):
        self.device = device
        self.rate = SR
        self.channels = 1
        cmd = ["pacat", "--raw", f"--rate={SR}", "--channels=1", "--format=float32le",
               "-d", device["target"], "--latency-msec=40"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=0)
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._play_end = 0.0
        self._rs: soxr.ResampleStream | None = None
        self._rs_sr = 0
        threading.Thread(target=self._writer, name="pacat-out", daemon=True).start()

    def play(self, audio: np.ndarray, sr: int, last: bool = True):
        audio = audio.astype(np.float32)
        if sr != self.rate:
            if self._rs is None or self._rs_sr != sr:
                self._rs, self._rs_sr = soxr.ResampleStream(sr, self.rate, 1, dtype="float32"), sr
            audio = self._rs.resample_chunk(audio, last=last)
            if last:
                self._rs = None
        if len(audio):
            dur = len(audio) / self.rate
            with self._lock:
                self._play_end = max(self._play_end, time.perf_counter()) + dur
            self._q.put(audio.tobytes())

    @property
    def queued_s(self) -> float:
        with self._lock:
            return max(0.0, self._play_end - time.perf_counter())

    def _writer(self):
        assert self._proc and self._proc.stdin
        while True:
            data = self._q.get()
            if data is None:
                return
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return

    def close(self):
        self._q.put(None)
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except Exception:
                self._proc.kill()

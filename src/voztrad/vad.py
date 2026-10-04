"""Silero VAD v6 (ONNX, CPU) + segmentador con histéresis y pre-roll."""

from __future__ import annotations

import itertools
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from .audio import Frame
from .config import FRAME, FRAME_MS, MODELS, SR, SegmenterConfig

_seq = itertools.count(1)


class SileroVAD:
    CONTEXT = 64  # muestras del frame anterior que el modelo espera como contexto

    def __init__(self, path=MODELS / "silero_vad.onnx"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        assert frame.shape == (FRAME,), frame.shape
        x = np.concatenate((self._ctx, frame[None, :]), axis=1)
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._ctx = x[:, -self.CONTEXT:]
        return float(out[0, 0])


@dataclass
class Segment:
    direction: str            # "A" (ellos->yo, en) | "B" (yo->ellos, es)
    audio: np.ndarray         # float32 16 kHz mono
    t_speech_end: float       # callback del último frame con voz (o soltar la tecla en PTT)
    t_emit: float             # cuando el segmentador lo entregó
    capture_ms: float         # demora cola+resample del último frame
    forced: bool = False      # cortado por max_segment_ms
    seq: int = field(default_factory=lambda: next(_seq))
    ts: dict = field(default_factory=dict)  # tiempos de las etapas siguientes

    @property
    def dur_s(self) -> float:
        return len(self.audio) / SR


class Segmenter:
    """IDLE --[p>enter en N frames]--> SPEAKING --[p<exit durante hangover]--> EMIT."""

    def __init__(self, direction: str, cfg: SegmenterConfig):
        self.d, self.cfg = direction, cfg
        self.vad = SileroVAD()
        self.preroll: deque[Frame] = deque(maxlen=cfg.preroll_frames)
        self.hang_frames = math.ceil(cfg.hangover_ms / FRAME_MS)
        self.max_frames = math.ceil(cfg.max_segment_ms / FRAME_MS)
        self.overlap_frames = math.ceil(cfg.overlap_ms / FRAME_MS)
        self.tail_frames = math.ceil(cfg.keep_tail_ms / FRAME_MS)
        self.min_frames = math.ceil(cfg.min_speech_ms / FRAME_MS)
        self._reset()

    def _reset(self):
        self.speaking = False
        self.above = 0
        self.silence = 0
        self.frames: list[Frame] = []
        self.voiced = 0
        self.last_voiced: Frame | None = None

    def push(self, f: Frame) -> Segment | None:
        p = self.vad(f.samples)
        if not self.speaking:
            self.preroll.append(f)
            self.above = self.above + 1 if p > self.cfg.enter_prob else 0
            if self.above >= self.cfg.enter_frames:
                self.speaking = True
                self.frames = list(self.preroll)  # incluye los frames que dispararon
                self.preroll.clear()
                self.voiced = self.above
                self.last_voiced = f
                self.silence = 0
            return None

        self.frames.append(f)
        if p < self.cfg.exit_prob:
            self.silence += 1
        else:
            self.silence = 0
            self.voiced += 1
            self.last_voiced = f

        if self.silence >= self.hang_frames:
            keep = len(self.frames) - self.silence + self.tail_frames
            long_enough = self.voiced >= self.min_frames
            seg = self._make(self.frames[:keep], forced=False) if long_enough else None
            self._reset()
            return seg

        if len(self.frames) >= self.max_frames:
            seg = self._make(self.frames, forced=True)
            carry = self.frames[-self.overlap_frames:]
            self.frames, self.voiced, self.silence = list(carry), 0, 0
            return seg
        return None

    def _make(self, frames: list[Frame], forced: bool) -> Segment:
        last = frames[-1] if forced else (self.last_voiced or frames[-1])
        return Segment(
            direction=self.d,
            audio=np.concatenate([fr.samples for fr in frames]),
            t_speech_end=last.t_cb,
            t_emit=time.perf_counter(),
            capture_ms=(last.t_ready - last.t_cb) * 1000,
            forced=forced,
        )


class PushToTalk:
    """Compuerta por tecla: marca inicio y fin exactos, sin VAD ni hangover."""

    def __init__(self, direction: str, max_ms: int, preroll_frames: int = 10):
        self.d = direction
        self.preroll: deque[Frame] = deque(maxlen=preroll_frames)
        self.max_frames = math.ceil(max_ms / FRAME_MS)
        self.frames: list[Frame] | None = None

    def push(self, f: Frame, pressed: bool, t_release: float | None) -> Segment | None:
        if self.frames is None:
            self.preroll.append(f)
            if pressed:
                self.frames = list(self.preroll)
                self.preroll.clear()
            return None
        self.frames.append(f)
        if pressed and len(self.frames) < self.max_frames:
            return None
        frames, self.frames = self.frames, None
        if len(frames) * FRAME_MS < 250:
            return None
        end = t_release if (not pressed and t_release) else frames[-1].t_cb
        return Segment(self.d, np.concatenate([fr.samples for fr in frames]), end,
                       time.perf_counter(), (frames[-1].t_ready - frames[-1].t_cb) * 1000,
                       forced=pressed)

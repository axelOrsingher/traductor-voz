"""Motor headless: captura -> VAD/PTT -> ASR -> MT -> (TTS -> VB-Cable).

Hilos:
  cap-A / cap-B   callbacks de PortAudio, solo put() en cola
  proc-A / proc-B downmix + resample + framing + VAD/PTT -> segmentos
  asr             UNO SOLO, dueño del modelo GPU; prioridad B > A
  mt              CPU
  tts             solo dirección B, escribe en VB-Cable
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

from . import asr as asr_mod
from .audio import CablePlayer, Capture, Framer, find_input, find_loopback, find_output
from .config import ROOT, SR, Config
from .langid import detect
from .mt import Translator
from .transport import JsonLinesServer
from .vad import PushToTalk, Segment, Segmenter
from .voseo import normalize

LANG = {"A": ("en", "es"), "B": ("es", "en")}
STAGES = ["capture", "vad", "asr_wait", "asr", "mt", "tts", "total"]


class DualQueue:
    """B siempre primero. A acotada: si se pasa de max_a se descarta el más viejo."""

    def __init__(self, max_a: int | None):
        self.a: deque = deque()
        self.b: deque = deque()
        self.max_a = max_a
        self.cv = threading.Condition()

    def put(self, item, direction: str):
        dropped = None
        with self.cv:
            if direction == "B":
                self.b.append(item)
            else:
                self.a.append(item)
                if self.max_a is not None and len(self.a) > self.max_a:
                    dropped = self.a.popleft()
            self.cv.notify()
        return dropped

    def get(self):
        with self.cv:
            while not (self.a or self.b):
                self.cv.wait()
            return self.b.popleft() if self.b else self.a.popleft()


class Engine:
    def __init__(self, cfg: Config, save_segments: bool = False):
        self.cfg = cfg
        self.stop = threading.Event()
        self.results: list[dict] = []
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (ROOT / "logs").mkdir(exist_ok=True)
        self.log = open(ROOT / "logs" / f"run-{stamp}.jsonl", "a", encoding="utf-8")
        self.seg_dir = ROOT / "out" / "segments" / stamp if save_segments else None
        if self.seg_dir:
            self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.server = JsonLinesServer(cfg.host, cfg.port, self._on_command)
        self._emit({"type": "status", "msg": "cargando modelos"})

        t = time.perf_counter()
        self.asr = asr_mod.load(cfg.asr, cfg.whisper_model, cfg.parakeet_quant)
        self.warmer = None
        if cfg.gpu_warm:
            from .gpuwarm import GpuWarmer
            self.warmer = GpuWarmer()
        self.mt = {("en", "es"): Translator("en", "es"), ("es", "en"): Translator("es", "en")}
        for tr in self.mt.values():
            tr.translate("hola")
        self.tts = None
        if cfg.enable_b:
            from .tts import load_tts
            self.tts = load_tts(cfg, lambda m: self._emit({"type": "status", "msg": m}))
        self._emit({"type": "status", "msg": f"modelos listos en {time.perf_counter() - t:.1f}s",
                    "asr": self.asr.name, "config": _cfg_dict(cfg)})

        self.asr_q = DualQueue(cfg.a_queue_max)
        self.mt_q = DualQueue(None)
        self.tts_q = DualQueue(None)

        self.pa = pyaudio.PyAudio()
        self.caps: dict[str, Capture] = {}
        if cfg.enable_a:
            self.caps["A"] = Capture(self.pa, find_loopback(self.pa, cfg.loopback))
        if cfg.enable_b:
            self.caps["B"] = Capture(self.pa, find_input(self.pa, cfg.mic))
            self.player = CablePlayer(self.pa, find_output(self.pa, cfg.cable))
            from .ptt import PushToTalkKey
            self.ptt = PushToTalkKey(cfg.ptt_key, on_change=lambda down: self._emit(
                {"type": "ptt", "down": down}, quiet=True))
        for d, c in self.caps.items():
            self._emit({"type": "status", "msg": f"{d}: {c.device['name']} {c.rate} Hz x{c.channels}"})
        if cfg.enable_b:
            self._emit({"type": "status", "msg": f"B -> {self.player.device['name']} | modo {cfg.b_mode}"
                        + (f" (tecla {cfg.ptt_key})" if cfg.b_mode == "ptt" else "")})

    # ---------------------------------------------------------------- hilos
    def run(self):
        threads = [threading.Thread(target=self._proc, args=(d,), name=f"proc-{d}", daemon=True)
                   for d in self.caps]
        threads += [threading.Thread(target=f, name=n, daemon=True)
                    for f, n in ((self._asr_loop, "asr"), (self._mt_loop, "mt"), (self._tts_loop, "tts"))]
        for t in threads:
            t.start()
        for c in self.caps.values():
            c.start()
        self._emit({"type": "status", "msg": "escuchando (Ctrl+C para terminar)"})
        try:
            while not self.stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _proc(self, d: str):
        cap = self.caps[d]
        framer = Framer(cap.rate, cap.channels)
        seg_vad = Segmenter(d, self.cfg.seg)
        ptt = PushToTalk(d, self.cfg.ptt_max_ms, self.cfg.seg.preroll_frames) if d == "B" else None
        active = False
        while not self.stop.is_set():
            t_cb, data = cap.q.get()
            for f in framer.push(t_cb, data):
                if ptt is not None and self.cfg.b_mode == "ptt":
                    seg = ptt.push(f, self.ptt.pressed, self.ptt.t_release)
                    talking = self.ptt.pressed
                else:
                    seg = seg_vad.push(f)
                    talking = seg_vad.speaking
                if talking and self.warmer:
                    self.warmer.poke()
                if talking != active:
                    active = talking
                    self._emit({"type": "voice", "dir": d, "active": active}, quiet=True)
                if seg is not None:
                    self._on_segment(seg)

    def _on_segment(self, seg: Segment):
        if self.seg_dir:
            sf.write(self.seg_dir / f"{seg.direction}_{seg.seq:04d}.wav", seg.audio, SR)
        dropped = self.asr_q.put(seg, seg.direction)
        if dropped is not None:
            self._emit({"type": "drop", "dir": "A", "seq": dropped.seq,
                        "msg": "backpressure: segmento A viejo descartado"})

    def _asr_loop(self):
        while True:
            seg: Segment = self.asr_q.get()
            src, _ = LANG[seg.direction]
            if self.warmer:
                self.warmer.pause()
            seg.ts["asr_start"] = time.perf_counter()
            try:
                text = self.asr.transcribe(seg.audio, src)
            finally:
                seg.ts["asr_end"] = time.perf_counter()
                if self.warmer:
                    self.warmer.resume()
            if not text:
                continue
            seg.ts["text"] = text
            self.mt_q.put(seg, seg.direction)

    def _mt_loop(self):
        while True:
            seg: Segment = self.mt_q.get()
            src, tgt = LANG[seg.direction]
            text = seg.ts["text"]
            lang = detect(text)
            seg.ts["lang"] = lang or src
            if lang not in (None, "es", "en"):
                # alucinación en otro idioma (p.ej. francés sobre audio en castellano): ni se traduce ni se dice
                self._emit({"type": "skip", "dir": seg.direction, "seq": seg.seq, "lang": lang,
                            "src": text, "msg": f"descartado: idioma {lang}"})
                continue
            seg.ts["mt_start"] = time.perf_counter()
            if lang == tgt:
                # ya está en el idioma destino (del otro lado hablan castellano / yo hablo en inglés)
                seg.ts["translation"] = text
                seg.ts["passthrough"] = True
            else:
                mt_in = normalize(text) if (src == "es" and self.cfg.normalize_voseo) else text
                seg.ts["translation"] = self.mt[(src, tgt)].translate(mt_in)
                if mt_in != text:
                    seg.ts["normalized"] = mt_in
            seg.ts["mt_end"] = time.perf_counter()
            if seg.direction == "B":
                self.tts_q.put(seg, "B")
            else:
                self._finish(seg, seg.ts["mt_end"])

    def _tts_loop(self):
        while True:
            seg: Segment = self.tts_q.get()
            t0 = time.perf_counter()
            first = None
            for chunk in self.tts.stream(seg.ts["translation"]):
                if first is None:
                    first = time.perf_counter()
                    seg.ts["tts_first"] = first - t0
                    self._finish(seg, first)  # la voz empieza a salir hacia Teams
                self.player.play(chunk, self.tts.sr, last=False)
            self.player.play(np.zeros(0, np.float32), self.tts.sr, last=True)  # vacía el resampler

    # ---------------------------------------------------------- métricas
    def _finish(self, seg: Segment, t_out: float):
        ts = seg.ts
        lat = {
            "capture": seg.capture_ms,
            "vad": (seg.t_emit - seg.t_speech_end) * 1000,
            "asr_wait": (ts["asr_start"] - seg.t_emit) * 1000,
            "asr": (ts["asr_end"] - ts["asr_start"]) * 1000,
            "mt": (ts["mt_end"] - ts["mt_start"]) * 1000,
            "tts": ts["tts_first"] * 1000 if "tts_first" in ts else None,
            "total": (t_out - seg.t_speech_end) * 1000,
        }
        ev = {"type": "phrase", "dir": seg.direction, "seq": seg.seq, "asr": self.asr.name,
              "tts": getattr(self.tts, "name", None) if seg.direction == "B" else None,
              "gpu_warm": self.warmer is not None,
              "mode": (self.cfg.b_mode if seg.direction == "B" else "vad"),
              "audio_s": round(seg.dur_s, 2), "forced": seg.forced,
              "src": ts["text"], "tgt": ts["translation"], "norm": ts.get("normalized"),
              "lang": ts.get("lang"), "passthrough": ts.get("passthrough", False),
              "lat": {k: (round(v, 1) if v is not None else None) for k, v in lat.items()}}
        self.results.append(ev)
        self._emit(ev)

    def _emit(self, ev: dict, quiet: bool = False):
        """quiet: solo a la UI (estado efímero: PTT, actividad de voz), sin log ni consola."""
        ev = {"t": round(time.time(), 3), **ev}
        self.server.publish(ev)
        if quiet:
            return
        self.log.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.log.flush()
        print(format_event(ev), flush=True)

    def state(self) -> dict:
        # la UI puede preguntar mientras todavía se cargan los modelos
        ptt = getattr(self, "ptt", None)
        asr = getattr(self, "asr", None)
        return {"type": "state", "ready": hasattr(self, "pa"), "asr": asr.name if asr else None,
                "b_mode": self.cfg.b_mode, "ptt_key": self.cfg.ptt_key, "ptt": bool(ptt and ptt.pressed),
                "enable_a": self.cfg.enable_a, "enable_b": self.cfg.enable_b,
                "hangover_ms": self.cfg.seg.hangover_ms}

    def _on_command(self, cmd: dict):
        c = cmd["cmd"]
        if c == "ptt" and self.cfg.enable_b:
            self.ptt.set_pressed(bool(cmd["down"]))
        elif c == "b_mode" and cmd["value"] in ("ptt", "vad"):
            self.cfg.b_mode = cmd["value"]
            self._emit({"type": "status", "msg": f"B en modo {self.cfg.b_mode}"})
            self._emit(self.state(), quiet=True)
        elif c == "state":
            self._emit(self.state(), quiet=True)
        elif c == "stop":
            self.stop.set()

    def close(self):
        self.stop.set()
        for c in self.caps.values():
            try:
                c.close()
            except OSError:
                pass
        print(summary(self.results))
        self.log.close()
        self.server.close()


# ------------------------------------------------------------------ salida
def format_event(ev: dict) -> str:
    if ev["type"] != "phrase":
        return f"[{ev['type']}] {ev.get('msg', '')}"
    l = ev["lat"]
    src, tgt = LANG[ev["dir"]]
    parts = " | ".join(f"{k} {l[k]:.0f}" for k in STAGES if l.get(k) is not None)
    s = (f"\n[{ev['dir']} #{ev['seq']} {ev['mode']} {ev['audio_s']}s{' CORTE' if ev['forced'] else ''}]\n"
         f"  {src.upper()}: {ev['src']}\n")
    if ev.get("norm"):
        s += f"  (norm): {ev['norm']}\n"
    return s + f"  {tgt.upper()}: {ev['tgt']}\n  ms: {parts}"


def summary(results: list[dict]) -> str:
    out = ["", "=" * 72, "Resumen de latencias (ms)"]
    for d in ("A", "B"):
        rows = [r for r in results if r["dir"] == d]
        if not rows:
            continue
        out.append(f"\nDirección {d} — {len(rows)} frases — {rows[0]['asr']}")
        out.append(f"  {'etapa':10} {'p50':>8} {'p95':>8} {'max':>8}")
        for k in STAGES:
            v = [r["lat"][k] for r in rows if r["lat"].get(k) is not None]
            if v:
                out.append(f"  {k:10} {np.percentile(v, 50):8.0f} {np.percentile(v, 95):8.0f} {max(v):8.0f}")
    return "\n".join(out)


def _cfg_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    d = asdict(cfg)
    d["tts_voice"] = str(cfg.tts_voice)
    d["voice_ref"] = str(cfg.voice_ref) if cfg.voice_ref else None
    return d


def setup_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

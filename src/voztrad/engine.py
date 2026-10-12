"""Motor headless: captura -> VAD/PTT -> ASR -> MT -> (TTS -> VB-Cable).

Hilos:
  cap-A / cap-B   callbacks de PortAudio, solo put() en cola
  proc-A / proc-B downmix + resample + framing + VAD/PTT -> segmentos
  asr             UNO SOLO, dueño del modelo GPU; prioridad B > A
  mt              CPU
  tts             solo dirección B, escribe en VB-Cable
"""

from __future__ import annotations

import contextlib
import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    pyaudio = None  # listen-in-Spanish path (WASAPI-only) is disabled on Linux

from . import asr as asr_mod
from .audio import (CablePlayer, Capture, Framer, PyAudio, find_default_output,
                    find_input, find_loopback, find_output)
from .config import FRAME_MS, ROOT, SR, Config
from .langid import detect
from .tempo import Wsola, rate_for_backlog
from .mt import Translator
from .transport import JsonLinesServer
from .vad import PushToTalk, Segment, Segmenter
from .voseo import normalize

LANG = {"A": ("en", "es"), "B": ("es", "en")}
STAGES = ["capture", "vad", "asr_wait", "asr", "mt", "tts", "total"]
PARTIAL_EVERY_S = 0.7    # subtítulo en vivo cada 0,7 s de habla
PARTIAL_MIN_MS = 1000    # recién después de 1 s (antes no hay nada que leer)


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

    def get(self, timeout: float | None = None):
        with self.cv:
            if not (self.a or self.b):
                self.cv.wait_for(lambda: self.a or self.b, timeout)
            if not (self.a or self.b):
                return None
            return self.b.popleft() if self.b else self.a.popleft()

    def __len__(self):
        with self.cv:
            return len(self.a) + len(self.b)


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
        # con el refinador, Parakeet en fp16 para que TranslateGemma entre en los 6 GB
        quant = cfg.parakeet_quant or ("fp16" if cfg.refine else None)
        self.asr = asr_mod.load(cfg.asr, cfg.whisper_model, quant)
        self.refiner = None
        self.refine_q: queue.Queue = queue.Queue()
        self._asr_busy = False
        if cfg.refine:
            try:
                from .refine import Refiner
                self.refiner = Refiner()
            except Exception as e:  # noqa: BLE001 — sin refinador se sigue con la traducción rápida
                self._emit({"type": "status", "msg": f"traducción refinada no disponible: {e}"})
        self.warmer = None
        if cfg.gpu_warm:
            from .gpuwarm import GpuWarmer
            self.warmer = GpuWarmer()
        self.mt = {("en", "es"): Translator("en", "es", compute_type=cfg.mt_compute_type),
                   ("es", "en"): Translator("es", "en", compute_type=cfg.mt_compute_type)}
        for tr in self.mt.values():
            tr.translate("hola")
        self.tts = None
        if cfg.enable_b:
            from .tts import load_tts
            self.tts = load_tts(cfg, lambda m: self._emit({"type": "status", "msg": m}))
        self._emit({"type": "status", "msg": f"modelos listos en {time.perf_counter() - t:.1f}s",
                    "asr": self.asr.name, "config": _cfg_dict(cfg)})

        self.smart_turn = None
        if cfg.smart_turn:
            from .smartturn import SmartTurn
            self.smart_turn = SmartTurn()
        self._partials: dict[str, np.ndarray] = {}
        self._partial_lock = threading.Lock()
        self.asr_q = DualQueue(cfg.a_queue_max)
        self.mt_q = DualQueue(None)
        self.tts_q = DualQueue(None)

        self.pa = PyAudio()
        self.caps: dict[str, Capture] = {}
        self.listen_ok = False
        if cfg.enable_a:
            self.caps["A"] = self._open_a_capture()
            if cfg.listen and self.listen_ok:
                from .ducker import Ducker
                from .tts import Piper
                self.listen_tts = Piper(cfg.listen_voice)
                self.listen_player = CablePlayer(self.pa, find_default_output(self.pa))
                self.ducker = Ducker(cfg.duck_apps, cfg.listen_duck)
                self.listen_q: queue.Queue = queue.Queue()
                self.listen_results: list[float] = []
                self._listen_until = 0.0
                self._emit({"type": "status", "msg": f"escucha en castellano -> {self.listen_player.device['name']}"
                            f" (Teams al {round(cfg.listen_duck * 100)}% mientras suena)"})
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

    def _open_a_capture(self):
        """Loopback de todo el sistema EXCEPTO este proceso: así la traducción al castellano que
        suena en los auriculares no vuelve a entrar por la dirección A."""
        if sys.platform != "win32":
            # PipeWire: A captures the voztrad_calls monitor; our Spanish TTS goes to the
            # real output (not voztrad_calls), so it is never re-captured -> listen is safe.
            self.listen_ok = True
            return Capture(self.pa, find_loopback(self.pa, self.cfg.loopback))
        if self.cfg.loopback is None:
            try:
                from .procloop import ProcessLoopbackCapture
                cap = ProcessLoopbackCapture()
                self.listen_ok = True
                return cap
            except Exception as e:  # noqa: BLE001 — Windows viejo o API no disponible
                self._emit({"type": "status", "msg": f"process loopback no disponible ({e}); "
                            "uso loopback común y desactivo la escucha en castellano"})
        return Capture(self.pa, find_loopback(self.pa, self.cfg.loopback))

    # ---------------------------------------------------------------- hilos
    def run(self):
        threads = [threading.Thread(target=self._proc, args=(d,), name=f"proc-{d}", daemon=True)
                   for d in self.caps]
        threads += [threading.Thread(target=f, name=n, daemon=True)
                    for f, n in ((self._asr_loop, "asr"), (self._mt_loop, "mt"), (self._tts_loop, "tts"))]
        if self.refiner:
            threads.append(threading.Thread(target=self._refine_loop, name="refine", daemon=True))
        if hasattr(self, "listen_q"):
            threads.append(threading.Thread(target=self._listen_loop, name="listen", daemon=True))
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
        seg_vad.turn = self.smart_turn
        ptt = PushToTalk(d, self.cfg.ptt_max_ms, self.cfg.seg.preroll_frames) if d == "B" else None
        active = False
        last_partial = 0.0
        while not self.stop.is_set():
            t_cb, data = cap.q.get()
            for f in framer.push(t_cb, data):
                if (d == "B" and self.cfg.listen_gate and self.cfg.b_mode != "ptt"
                        and time.perf_counter() < getattr(self, "_listen_until", 0.0)):
                    # suena la traducción al castellano: si no usás auriculares, el mic la
                    # tomaría como voz tuya y la mandaría traducida a la llamada
                    f.samples = np.zeros_like(f.samples)
                if ptt is not None and self.cfg.b_mode == "ptt":
                    seg = ptt.push(f, self.ptt.pressed, self.ptt.t_release)
                    talking = self.ptt.pressed
                else:
                    seg_vad.interp = d == "B" and self.cfg.b_mode == "interp"
                    seg = seg_vad.push(f)
                    talking = seg_vad.speaking
                if talking and self.warmer:
                    self.warmer.poke()
                if talking != active:
                    active = talking
                    self._emit({"type": "voice", "dir": d, "active": active}, quiet=True)
                if seg is not None:
                    self._on_segment(seg)
                # subtítulos en vivo: re-transcribir lo que va diciendo cada PARTIAL_EVERY_S
                if self.cfg.partials and talking:
                    ptt_mode = ptt is not None and self.cfg.b_mode == "ptt"
                    live = ptt.frames if ptt_mode else seg_vad.frames
                    quiet = 0 if ptt_mode else seg_vad.silence  # ya hay silencio: viene el final
                    now = time.perf_counter()
                    if (live and quiet == 0 and len(live) * FRAME_MS >= PARTIAL_MIN_MS
                            and now - last_partial >= PARTIAL_EVERY_S):
                        last_partial = now
                        with self._partial_lock:
                            self._partials[d] = np.concatenate([fr.samples for fr in live])
                elif not talking:
                    last_partial = 0.0

    def _on_segment(self, seg: Segment):
        if self.seg_dir:
            sf.write(self.seg_dir / f"{seg.direction}_{seg.seq:04d}.wav", seg.audio, SR)
        dropped = self.asr_q.put(seg, seg.direction)
        if dropped is not None:
            self._emit({"type": "drop", "dir": "A", "seq": dropped.seq,
                        "msg": "backpressure: segmento A viejo descartado"})

    def _gpu(self):
        """Envolver todo trabajo real de GPU: el calentador se pausa mientras tanto."""
        return self.warmer.paused() if self.warmer else contextlib.nullcontext()

    def _partial(self):
        """Subtítulo provisorio de lo que se está diciendo. Solo con la GPU libre: los finales
        siempre van primero, y si hay uno esperando el parcial se descarta."""
        with self._partial_lock:
            d = "B" if "B" in self._partials else ("A" if "A" in self._partials else None)
            audio = self._partials.pop(d) if d else None
        if audio is None or len(self.asr_q):
            return
        src, tgt = LANG[d]
        with self._gpu():
            text = self.asr.transcribe(audio, src)
        if not text or len(self.asr_q):
            return
        lang = detect(text)
        if lang not in (None, "es", "en"):
            return
        with self._gpu():
            tr = text if lang == tgt else self.mt[(src, tgt)].translate(
                normalize(text) if (src == "es" and self.cfg.normalize_voseo) else text)
        self._emit({"type": "partial", "dir": d, "src": text, "tgt": tr}, quiet=True)

    def _asr_loop(self):
        while True:
            seg: Segment = self.asr_q.get(timeout=0.05)
            if seg is None:
                if self.cfg.partials:
                    self._partial()
                continue
            with self._partial_lock:  # ya llegó el final: el parcial de esa dirección sobra
                self._partials.pop(seg.direction, None)
            src, _ = LANG[seg.direction]
            seg.ts["asr_start"] = time.perf_counter()
            self._asr_busy = True
            try:
                with self._gpu():
                    text = self.asr.transcribe(seg.audio, src)
            finally:
                self._asr_busy = False
            seg.ts["asr_end"] = time.perf_counter()
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
                with self._gpu():
                    seg.ts["translation"] = self.mt[(src, tgt)].translate(mt_in)
                if mt_in != text:
                    seg.ts["normalized"] = mt_in
            seg.ts["mt_end"] = time.perf_counter()
            if seg.direction == "B":
                self.tts_q.put(seg, "B")
            else:
                self._finish(seg, seg.ts["mt_end"])
                if self.cfg.listen and hasattr(self, "listen_q") and not seg.ts.get("passthrough"):
                    self.listen_q.put(seg)
                if self.refiner and not seg.ts.get("passthrough") and len(text.split()) >= 2:
                    self.refine_q.put(seg)

    def _listen_loop(self):
        """Traducción al castellano en los auriculares, con Teams bajito mientras suena."""
        while True:
            seg: Segment = self.listen_q.get()
            if not self.cfg.listen:
                continue
            self.ducker.duck()
            t0 = time.perf_counter()
            first = True
            for chunk in self.listen_tts.stream(seg.ts["translation"]):
                if first:
                    now = time.perf_counter()
                    total = (now - seg.t_speech_end) * 1000
                    self.listen_results.append(total)
                    self._emit({"type": "listen", "seq": seg.seq, "tgt": seg.ts["translation"],
                                "tts_ms": round((now - t0) * 1000, 1), "total_ms": round(total, 1)})
                    first = False
                self.listen_player.play(chunk, self.listen_tts.sr, last=False)
                self._listen_until = time.perf_counter() + self.listen_player.queued_s + 0.3
            self.listen_player.play(np.zeros(0, np.float32), self.listen_tts.sr, last=True)
            # restaurar Teams cuando terminó de sonar y no hay otra frase esperando
            while self.listen_q.empty() and self.listen_player.queued_s > 0.02:
                self._listen_until = time.perf_counter() + self.listen_player.queued_s + 0.3
                time.sleep(0.03)
            if self.listen_q.empty():
                time.sleep(0.25)
                if self.listen_q.empty():
                    self.ducker.restore()

    def _tts_loop(self):
        while True:
            seg: Segment = self.tts_q.get()
            t0 = time.perf_counter()
            first = None
            # Tempo adaptativo: si todavía está sonando inglés anterior, este fragmento va más
            # rápido (sin cambiar el tono) para no acumular atraso en monólogos.
            backlog = self.player.queued_s
            # Traducción buena (TranslateGemma) SOLO si cabe dentro del audio que ya está
            # sonando: así no suma latencia. Si no llega a tiempo, va la rápida.
            budget_ms = backlog * 1000 - 120
            if self.refiner and not seg.ts.get("passthrough") and budget_ms > self.refiner.ema_ms:
                good = self.refiner.translate(seg.ts["text"], "es", "en", timeout=budget_ms / 1000)
                if good:
                    seg.ts["translation_fast"], seg.ts["translation"] = seg.ts["translation"], good
                    seg.ts["refined_voice"] = True
                backlog = self.player.queued_s
            rate = rate_for_backlog(backlog)
            seg.ts["tempo"], seg.ts["backlog_s"] = rate, round(backlog, 2)
            stretch = Wsola(rate, self.tts.sr) if rate != 1.0 else None
            for chunk in self.tts.stream(seg.ts["translation"]):
                if stretch:
                    chunk = stretch.push(chunk)
                if first is None and len(chunk):
                    first = time.perf_counter()
                    seg.ts["tts_first"] = first - t0
                    self._finish(seg, first)  # la voz empieza a salir hacia Teams
                self.player.play(chunk, self.tts.sr, last=False)
            tail = stretch.push(np.zeros(0, np.float32), last=True) if stretch else np.zeros(0, np.float32)
            self.player.play(tail, self.tts.sr, last=True)  # vacía WSOLA y el resampler
            if self.refiner and not seg.ts.get("refined_voice") and not seg.ts.get("passthrough"):
                self.refine_q.put(seg)  # la voz salió con la rápida: al menos refinar el subtítulo

    def _refine_loop(self):
        """Subtítulos: reemplaza la traducción rápida por la de TranslateGemma cuando llega."""
        while True:
            seg: Segment = self.refine_q.get()
            src, tgt = LANG[seg.direction]
            t0 = time.perf_counter()
            # Ceder la GPU al ASR: medido, con TranslateGemma generando el ASR pasa de 119 a
            # 187 ms. Un subtítulo refinado 100 ms después no se nota; un ASR más lento, sí.
            while (len(self.asr_q) or self._asr_busy) and time.perf_counter() - t0 < 1.0:
                time.sleep(0.01)
            good = self.refiner.translate(seg.ts["text"], src, tgt, timeout=3.0)
            if good and good != seg.ts["translation"]:
                self._emit({"type": "refine", "dir": seg.direction, "seq": seg.seq, "tgt": good,
                            "fast": seg.ts["translation"], "ms": round((time.perf_counter() - t0) * 1000)})

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
              "audio_s": round(seg.dur_s, 2), "forced": seg.forced, "partial": seg.partial,
              "src": ts["text"], "tgt": ts["translation"], "norm": ts.get("normalized"),
              "lang": ts.get("lang"), "passthrough": ts.get("passthrough", False),
              "tempo": ts.get("tempo"), "backlog_s": ts.get("backlog_s"),
              "refined": bool(ts.get("refined_voice")), "fast": ts.get("translation_fast"),
              "end": seg.end_reason, "turn_p": None if seg.turn_prob is None else round(seg.turn_prob, 3),
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
                "hangover_ms": self.cfg.seg.hangover_ms,
                "listen": bool(self.cfg.listen and getattr(self, "listen_ok", False)),
                "listen_available": bool(getattr(self, "listen_ok", False))}

    def _on_command(self, cmd: dict):
        c = cmd["cmd"]
        if c == "ptt" and self.cfg.enable_b:
            self.ptt.set_pressed(bool(cmd["down"]))
        elif c == "b_mode" and cmd["value"] in ("ptt", "vad", "interp"):
            self.cfg.b_mode = cmd["value"]
            self._emit({"type": "status", "msg": f"B en modo {self.cfg.b_mode}"})
            self._emit(self.state(), quiet=True)
        elif c == "listen":
            self.cfg.listen = bool(cmd["value"])
            if not self.cfg.listen and hasattr(self, "ducker"):
                self.ducker.restore()
            self._emit({"type": "status", "msg": f"escucha en castellano {'activada' if self.cfg.listen else 'desactivada'}"})
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
        if hasattr(self, "ducker"):
            self.ducker.restore_now()  # nunca dejar Teams bajito
        if getattr(self, "refiner", None):
            self.refiner.close()
        print(summary(self.results))
        lr = getattr(self, "listen_results", [])
        if lr:
            print(f"\nEscucha en castellano — {len(lr)} frases: fin de frase -> audio "
                  f"p50 {np.percentile(lr, 50):.0f} / p95 {np.percentile(lr, 95):.0f} ms")
        self.log.close()
        self.server.close()


# ------------------------------------------------------------------ salida
def format_event(ev: dict) -> str:
    if ev["type"] == "listen":
        return f"  🔊 ES (#{ev['seq']}): tts {ev['tts_ms']:.0f} | total {ev['total_ms']:.0f} ms"
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
    from pathlib import PurePath
    return {k: (str(v) if isinstance(v, PurePath) else v) for k, v in asdict(cfg).items()}


def setup_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

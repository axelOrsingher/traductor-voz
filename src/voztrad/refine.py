"""Traducción "buena" con TranslateGemma-4B (llama.cpp, GPU) para refinar la rápida (opus-mt).

Investigación medida en esta máquina (2026-10-05): ningún traductor rápido mejora a opus-mt;
TranslateGemma es el único que arregla modismos ("está bárbaro" -> "it's great", "infra" ->
"infrastructure team", "lo vemos" -> "let's discuss it") pero tarda ~250-400 ms por frase.
Por eso va en un SEGUNDO nivel: el subtítulo sale al instante con opus-mt y se reemplaza; la
voz usa la versión buena solo si hay tiempo (ver Engine._tts_loop).

llama-server corre como proceso hijo dentro de un Job de Windows con KILL_ON_JOB_CLOSE: si el
motor muere, el servidor (y sus 2,5 GB de VRAM) muere con él.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import MODELS, ROOT

LLAMA_DIR = Path.home() / "tools" / "llama.cpp-b11433-cuda12.4"
MODEL = MODELS / "translategemma-4b-it.Q4_K_M.gguf"
NAMES = {"es": "Spanish", "en": "English"}
CODES = {"es": "es-AR", "en": "en"}


def prompt(text: str, src: str, tgt: str) -> str:
    """Formato oficial de TranslateGemma (leído del template del GGUF)."""
    s, t = NAMES[src], NAMES[tgt]
    u = (f"You are a professional {s} ({CODES[src]}) to {t} ({CODES[tgt]}) translator. Your goal is to "
         f"accurately convey the meaning and nuances of the original {s} text while adhering to {t} "
         f"grammar, vocabulary, and cultural sensitivities.\n"
         f"Produce only the {t} translation, without any additional explanations or commentary. "
         f"Please translate the following {s} text into {t}:\n\n\n{text}")
    return f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n"


def _kill_with_parent(proc: subprocess.Popen):
    """Mete al proceso en un Job que se cierra (y lo mata) cuando muere este proceso."""
    if sys.platform != "win32":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    job = k32.CreateJobObjectW(None, None)

    class _Basic(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", ctypes.c_uint32),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_uint32),
                    ("f", ctypes.c_size_t), ("g", ctypes.c_uint32), ("h", ctypes.c_uint32)]

    class _Ext(ctypes.Structure):
        _fields_ = [("basic", _Basic), ("io", ctypes.c_uint64 * 6), ("p", ctypes.c_size_t),
                    ("j", ctypes.c_size_t), ("pp", ctypes.c_size_t), ("pj", ctypes.c_size_t)]

    info = _Ext()
    info.basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # Extended limit info
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.AssignProcessToJobObject(job, int(proc._handle))
    return job  # el handle queda abierto mientras viva el motor


class Refiner:
    def __init__(self, port: int = 8781, model: Path = MODEL):
        exe = LLAMA_DIR / "llama-server.exe"
        if not exe.exists() or not model.exists():
            raise FileNotFoundError(f"falta {exe if not exe.exists() else model}")
        self.url = f"http://127.0.0.1:{port}"
        log = open(ROOT / "logs" / "llama-server.txt", "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(exe), "-m", str(model), "-ngl", "99", "-c", "1024", "-np", "1", "-fa", "on",
             "--threads", "2", "--no-webui", "--no-jinja", "--port", str(port), "--host", "127.0.0.1"],
            cwd=LLAMA_DIR, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self._job = _kill_with_parent(self.proc)
        self._lock = threading.Lock()
        self.ema_ms = 350.0  # latencia típica estimada; se actualiza con cada pedido
        self._wait_ready(90)
        for src, tgt in (("es", "en"), ("en", "es")):  # warm-up: cachea el prefijo del prompt
            self.translate("Hola, ¿cómo estás?" if src == "es" else "Hi, how are you?", src, tgt)

    def _wait_ready(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server terminó al arrancar (ver logs/llama-server.txt)")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.3)
        raise RuntimeError("llama-server no respondió")

    def translate(self, text: str, src: str, tgt: str, timeout: float = 5.0) -> str | None:
        """Traducción buena, o None si no llegó a tiempo / falló (se usa la rápida)."""
        body = json.dumps({"prompt": prompt(text, src, tgt), "temperature": 0, "cache_prompt": True,
                           "n_predict": min(200, 3 * len(text.split()) + 24),
                           "stop": ["<end_of_turn>"]}).encode()
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        if not self._lock.acquire(timeout=timeout):
            return None
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())["content"].strip()
            self.ema_ms = 0.8 * self.ema_ms + 0.2 * (time.perf_counter() - t0) * 1000
            return out or None
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return None
        finally:
            self._lock.release()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()

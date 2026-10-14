"""High-quality translation with TranslateGemma-4B (llama.cpp) that refines the fast one (opus-mt).

Measured on the Windows dev box (2026-10-05): no fast translator beats opus-mt; TranslateGemma
is the only one that fixes idioms ("está bárbaro" -> "it's great", "infra" -> "infrastructure
team", "lo vemos" -> "let's discuss it"), but it takes ~250-400 ms per phrase. So it is a
SECOND tier: the subtitle shows opus-mt instantly and is then replaced; your voice uses the good
version only if there is time (see Engine._tts_loop).

llama-server runs as a child process that must die with the engine, even on a crash (it holds
~2.5 GB of VRAM): a Windows Job with KILL_ON_JOB_CLOSE, PR_SET_PDEATHSIG on Linux, and a small
watchdog process elsewhere (macOS).
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import MODELS, ROOT

# Pinned Windows build installed by scripts/fetch_models.py.
LLAMA_DIR = Path.home() / "tools" / "llama.cpp-b11433-cuda12.4"
MODEL = MODELS / "translategemma-4b-it.Q4_K_M.gguf"
NAMES = {"es": "Spanish", "en": "English"}
CODES = {"es": "es-AR", "en": "en"}


def find_server() -> Path | None:
    """llama-server binary: $VOZTRAD_LLAMA_SERVER, then the pinned Windows build, then PATH
    (e.g. `brew install llama.cpp` on macOS, a distro or self-built package on Linux)."""
    env = os.environ.get("VOZTRAD_LLAMA_SERVER")
    if env:
        return Path(env) if Path(env).exists() else None
    pinned = LLAMA_DIR / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if pinned.exists():
        return pinned
    found = shutil.which("llama-server")
    return Path(found) if found else None


def prompt(text: str, src: str, tgt: str) -> str:
    """Official TranslateGemma format (read from the GGUF chat template)."""
    s, t = NAMES[src], NAMES[tgt]
    u = (f"You are a professional {s} ({CODES[src]}) to {t} ({CODES[tgt]}) translator. Your goal is to "
         f"accurately convey the meaning and nuances of the original {s} text while adhering to {t} "
         f"grammar, vocabulary, and cultural sensitivities.\n"
         f"Produce only the {t} translation, without any additional explanations or commentary. "
         f"Please translate the following {s} text into {t}:\n\n\n{text}")
    return f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n"


def _die_with_parent_linux():
    """preexec_fn: the kernel sends SIGTERM to the child when the parent dies."""
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


# Watchdog for platforms without a kernel mechanism (macOS): polls the parent and kills the child.
_WATCHDOG = """
import os, signal, sys, time
parent, child = int(sys.argv[1]), int(sys.argv[2])
def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
while alive(child):
    if os.getppid() != parent or not alive(parent):
        try:
            os.kill(child, signal.SIGTERM); time.sleep(2); os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        break
    time.sleep(0.5)
"""


def _windows_job(proc: subprocess.Popen):
    """Put the process in a Job that is closed (killing it) when this process dies."""
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
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limit info
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.AssignProcessToJobObject(job, int(proc._handle))
    return job  # the handle stays open while the engine lives


def _guard_child(proc: subprocess.Popen):
    """Make sure the child dies with this process. Returns a handle to keep alive."""
    if sys.platform == "win32":
        return _windows_job(proc)
    if sys.platform.startswith("linux"):
        return None  # PR_SET_PDEATHSIG was set in preexec_fn
    return subprocess.Popen([sys.executable, "-c", _WATCHDOG, str(os.getpid()), str(proc.pid)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Refiner:
    def __init__(self, port: int = 8781, model: Path = MODEL):
        exe = find_server()
        if exe is None:
            raise FileNotFoundError("llama-server not found (scripts/fetch_models.py, PATH, "
                                    "or VOZTRAD_LLAMA_SERVER)")
        if not model.exists():
            raise FileNotFoundError(f"missing {model}")
        self.url = f"http://127.0.0.1:{port}"
        log = open(ROOT / "logs" / "llama-server.txt", "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(exe), "-m", str(model), "-ngl", "99", "-c", "1024", "-np", "1", "-fa", "on",
             "--threads", "2", "--no-webui", "--no-jinja", "--port", str(port), "--host", "127.0.0.1"],
            cwd=exe.parent, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            preexec_fn=_die_with_parent_linux if sys.platform.startswith("linux") else None)
        self._guard = _guard_child(self.proc)
        self._lock = threading.Lock()
        self.ema_ms = 350.0  # typical latency estimate; updated on every request
        self._wait_ready(90)
        for src, tgt in (("es", "en"), ("en", "es")):  # warm-up: caches the prompt prefix
            self.translate("Hola, ¿cómo estás?" if src == "es" else "Hi, how are you?", src, tgt)

    def _wait_ready(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup (see logs/llama-server.txt)")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.3)
        raise RuntimeError("llama-server did not respond")

    def translate(self, text: str, src: str, tgt: str, timeout: float = 5.0) -> str | None:
        """Good translation, or None if it was late / failed (the fast one is used)."""
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

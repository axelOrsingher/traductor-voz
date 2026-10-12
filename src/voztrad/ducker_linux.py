"""Duck the call app on Linux via pactl: lower every sink-input whose
application.name matches one of the configured apps, restore it afterwards.

duck()/restore() are queued and run on a worker thread so the translation starts
playing without waiting for pactl. Volumes are always restored (atexit), including
on a hard exit."""

from __future__ import annotations

import atexit
import queue
import re
import subprocess
import threading

_VOL = re.compile(r"/\s*(\d+)%")


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True).stdout
    except Exception:
        return ""


class Ducker:
    def __init__(self, apps=("ms-teams.exe", "teams.exe"), level: float = 0.35):
        # strip the Windows .exe so "teams.exe" also matches the Linux app.name "teams"
        self.names = {a.lower().removesuffix(".exe") for a in apps}
        self.level = level
        self._saved: dict[str, int] = {}   # sink-input index -> original volume %
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, name="ducker", daemon=True).start()
        atexit.register(self.restore_now)

    def duck(self):
        self._q.put("duck")

    def restore(self):
        self._q.put("restore")

    def _worker(self):
        while True:
            op = self._q.get()
            try:
                self.duck_now() if op == "duck" else self.restore_now()
            except Exception:
                pass

    def _matching_inputs(self) -> list[tuple[str, int]]:
        """[(sink-input index, current volume %)] for streams of the configured apps."""
        out, idx, app, vol = [], None, None, None
        for ln in _pactl("list", "sink-inputs").splitlines():
            s = ln.strip()
            if s.startswith("Sink Input #"):
                idx, app, vol = s.split("#", 1)[1], None, None
            elif s.startswith("application.name = "):
                app = s.split("=", 1)[1].strip().strip('"').lower()
            elif s.startswith("Volume:") and vol is None:
                m = _VOL.search(s)
                vol = int(m.group(1)) if m else None
            if idx and app is not None and vol is not None:
                if any(n in app for n in self.names):
                    out.append((idx, vol))
                idx = app = vol = None  # one record done
        return out

    def duck_now(self):
        with self._lock:
            if self._saved:
                return
            for idx, vol in self._matching_inputs():
                self._saved[idx] = vol
                _pactl("set-sink-input-volume", idx, f"{max(1, int(vol * self.level))}%")

    def restore_now(self):
        with self._lock:
            for idx, vol in self._saved.items():
                _pactl("set-sink-input-volume", idx, f"{vol}%")
            self._saved.clear()

    @property
    def ducked(self) -> bool:
        return bool(self._saved)

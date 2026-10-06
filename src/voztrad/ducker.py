"""Baja el volumen de Teams mientras suena la traducción al castellano (y lo restaura).

En el Teams nuevo el audio de la llamada sale de msedgewebview2.exe, hijo de
ms-teams.exe (verificado en esta máquina): se baja toda sesión cuyo proceso o algún
ancestro sea una de las apps configuradas. El volumen original se restaura siempre,
también al cerrar el motor (atexit), para no dejar Teams bajito.

Buscar las sesiones tarda ~250 ms (medido): duck()/restore() se ENCOLAN y los ejecuta un
hilo propio, así la traducción empieza a sonar sin esperar.
"""

from __future__ import annotations

import atexit
import queue
import threading

import comtypes
import psutil


class Ducker:
    def __init__(self, apps=("ms-teams.exe", "teams.exe"), level: float = 0.35):
        self.apps = {a.lower() for a in apps}
        self.level = level
        # una entrada por SESIÓN (un mismo proceso puede tener varias: ms-teams.exe tiene 2)
        self._saved: list[tuple[object, float]] = []
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, name="ducker", daemon=True).start()
        atexit.register(self.restore_now)

    def duck(self):
        self._q.put("duck")

    def restore(self):
        self._q.put("restore")

    def _worker(self):
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        while True:
            op = self._q.get()
            try:
                self.duck_now() if op == "duck" else self.restore_now()
            except Exception:  # noqa: BLE001 — una sesión que muere no puede tirar el hilo
                pass

    def _matches(self, pid: int) -> bool:
        try:
            p = psutil.Process(pid)
            for _ in range(4):
                if p is None:
                    return False
                if p.name().lower() in self.apps:
                    return True
                p = p.parent()
        except psutil.Error:
            pass
        return False

    def duck_now(self):
        from pycaw.pycaw import AudioUtilities
        with self._lock:
            if self._saved:
                return
            for s in AudioUtilities.GetAllSessions():
                if s.ProcessId and self._matches(s.ProcessId):
                    vol = s.SimpleAudioVolume
                    orig = vol.GetMasterVolume()
                    self._saved.append((vol, orig))
                    vol.SetMasterVolume(orig * self.level, None)

    def restore_now(self):
        with self._lock:
            for vol, orig in self._saved:
                try:
                    vol.SetMasterVolume(orig, None)
                except Exception:  # noqa: BLE001 — la sesión pudo haber muerto
                    pass
            self._saved.clear()

    @property
    def ducked(self) -> bool:
        return bool(self._saved)

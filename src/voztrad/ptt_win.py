"""Push-to-talk global por polling de GetAsyncKeyState (sin hooks ni dependencias)."""

from __future__ import annotations

import ctypes
import threading
import time

_VK = {f"F{i}": 0x6F + i for i in range(1, 25)}
_VK.update({"RCTRL": 0xA3, "LCTRL": 0xA2, "RALT": 0xA5, "CAPSLOCK": 0x14, "PAUSE": 0x13,
            "SCROLLLOCK": 0x91, "INSERT": 0x2D})


class PushToTalkKey:
    """Estado de la tecla. pressed / t_release los lee el hilo vad-B.

    También se puede manejar desde el transporte (set_pressed), p. ej. un botón en la UI.
    """

    def __init__(self, key: str, poll_ms: float = 5.0, on_change=None):
        self.vk = _VK[key.upper()]
        self.on_change = on_change
        self.pressed = False
        self.t_release: float | None = None
        self._user32 = ctypes.WinDLL("user32")
        self._poll = poll_ms / 1000
        threading.Thread(target=self._run, name="ptt", daemon=True).start()

    def set_pressed(self, down: bool):
        if self.pressed and not down:
            self.t_release = time.perf_counter()
        changed = down != self.pressed
        self.pressed = down
        if changed and self.on_change:
            self.on_change(down)

    def _run(self):
        last = False
        while True:
            down = bool(self._user32.GetAsyncKeyState(self.vk) & 0x8000)
            if down != last:
                self.set_pressed(down)
                last = down
            time.sleep(self._poll)

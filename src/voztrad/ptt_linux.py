"""Global push-to-talk on Linux/Wayland via python-evdev.

Reads raw key events from every keyboard under /dev/input/event* (user must be in
the 'input' group), so it works regardless of which window has focus — unlike
pynput under Wayland. Same surface as the Windows backend: `.pressed`,
`.t_release`, `.set_pressed()`, `on_change`."""

from __future__ import annotations

import threading
import time

from evdev import InputDevice, categorize, ecodes, list_devices

_KEYS = {f"F{i}": getattr(ecodes, f"KEY_F{i}") for i in range(1, 13)}
_KEYS.update({
    "RCTRL": ecodes.KEY_RIGHTCTRL, "LCTRL": ecodes.KEY_LEFTCTRL,
    "RALT": ecodes.KEY_RIGHTALT, "LALT": ecodes.KEY_LEFTALT,
    "CAPSLOCK": ecodes.KEY_CAPSLOCK, "PAUSE": ecodes.KEY_PAUSE,
    "SCROLLLOCK": ecodes.KEY_SCROLLLOCK, "INSERT": ecodes.KEY_INSERT,
})


class PushToTalkKey:
    def __init__(self, key: str, poll_ms: float = 5.0, on_change=None):
        self.code = _KEYS[key.upper()]
        self.on_change = on_change
        self.pressed = False
        self.t_release: float | None = None
        self._devices = self._open_keyboards()
        for dev in self._devices:
            threading.Thread(target=self._run, args=(dev,), name="ptt", daemon=True).start()

    def _open_keyboards(self) -> list[InputDevice]:
        out = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
                if self.code in dev.capabilities().get(ecodes.EV_KEY, []):
                    out.append(dev)
            except Exception:
                pass
        return out

    def set_pressed(self, down: bool):
        if self.pressed and not down:
            self.t_release = time.perf_counter()
        changed = down != self.pressed
        self.pressed = down
        if changed and self.on_change:
            self.on_change(down)

    def _run(self, dev: InputDevice):
        try:
            for event in dev.read_loop():
                if event.type == ecodes.EV_KEY and event.code == self.code:
                    if event.value == 1:       # key down (ignore 2 = autorepeat)
                        self.set_pressed(True)
                    elif event.value == 0:     # key up
                        self.set_pressed(False)
        except OSError:
            pass  # device unplugged

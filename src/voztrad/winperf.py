"""Evita que Windows 11 trate al motor como proceso de fondo.

Sin ventana en primer plano, Windows aplica power throttling (EcoQoS) y manda
los hilos a los E-cores / baja frecuencia: medido, el MT pasaba de ~45-85 ms a
~290 ms tras unos cientos de ms de inactividad. Se desactiva por proceso.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import sys

_ProcessPowerThrottling = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
_HIGH_PRIORITY_CLASS = 0x80
_ABOVE_NORMAL_PRIORITY_CLASS = 0x8000


class _PPTS(ctypes.Structure):
    _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]


def disable_throttling(priority: str = "above_normal") -> bool:
    if sys.platform != "win32":
        return False
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    h = k.GetCurrentProcess()
    st = _PPTS(_PROCESS_POWER_THROTTLING_CURRENT_VERSION,
               _PROCESS_POWER_THROTTLING_EXECUTION_SPEED | _PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
               0)  # ControlMask = lo que controlamos, StateMask 0 = "apagado"
    k.SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    ok = bool(k.SetProcessInformation(h, _ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st)))
    if priority:
        k.SetPriorityClass.argtypes = [w.HANDLE, w.DWORD]
        cls = _HIGH_PRIORITY_CLASS if priority == "high" else _ABOVE_NORMAL_PRIORITY_CLASS
        k.SetPriorityClass(h, cls)
    # Resolución de timer de 1 ms: time.sleep / waits más precisos en los hilos de audio.
    ctypes.WinDLL("winmm").timeBeginPeriod(1)
    return ok

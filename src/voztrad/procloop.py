"""Captura de TODO el audio del sistema EXCEPTO el de nuestro proceso (Windows 10 20H1+).

WASAPI "process loopback" (ActivateAudioInterfaceAsync + VAD\\Process_Loopback) en modo
EXCLUDE_TARGET_PROCESS_TREE con nuestro PID. Así podemos reproducir la traducción al
castellano en los auriculares sin que la dirección A se escuche a sí misma.
PyAudioWPatch no lo expone; esto va directo por COM (comtypes).

Misma interfaz que audio.Capture: .q con (t_callback, bytes float32 intercalado),
.rate, .channels, .device, start(), close().
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from ctypes import POINTER, Structure, WinDLL, byref, c_int, c_longlong, c_ubyte, c_uint32, c_uint64, c_ulong, c_ushort, c_void_p
from ctypes.wintypes import DWORD, HANDLE, WORD

import comtypes
from comtypes import COMMETHOD, GUID, HRESULT, COMObject, IUnknown

RATE, CHANNELS = 48000, 2
_LOOPBACK, _EVENTCALLBACK = 0x00020000, 0x00040000
_AUTOCONVERTPCM, _SRC_DEFAULT_QUALITY = 0x80000000, 0x08000000
_SILENT = 0x2
_INCLUDE, _EXCLUDE = 0, 1


class WAVEFORMATEX(Structure):
    _fields_ = [("wFormatTag", WORD), ("nChannels", WORD), ("nSamplesPerSec", DWORD),
                ("nAvgBytesPerSec", DWORD), ("nBlockAlign", WORD), ("wBitsPerSample", WORD), ("cbSize", WORD)]


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize", (["in"], DWORD, "mode"), (["in"], DWORD, "flags"),
                  (["in"], c_longlong, "buf"), (["in"], c_longlong, "period"),
                  (["in"], POINTER(WAVEFORMATEX), "fmt"), (["in"], POINTER(GUID), "session")),
        COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32), "n")),
        COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(c_longlong), "lat")),
        COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32), "pad")),
        COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], DWORD, "mode"), (["in"], POINTER(WAVEFORMATEX), "fmt"),
                  (["out"], POINTER(POINTER(WAVEFORMATEX)), "closest")),
        COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(POINTER(WAVEFORMATEX)), "fmt")),
        COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(c_longlong), "dflt"), (["out"], POINTER(c_longlong), "mn")),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle", (["in"], HANDLE, "h")),
        COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "iid"), (["out"], POINTER(POINTER(IUnknown)), "svc")),
    ]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(POINTER(c_ubyte)), "data"),
                  (["out"], POINTER(c_uint32), "frames"), (["out"], POINTER(DWORD), "flags"),
                  (["out"], POINTER(c_uint64), "devpos"), (["out"], POINTER(c_uint64), "qpcpos")),
        COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32, "n")),
        COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32), "n")),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(c_int), "hr"),
                           (["out"], POINTER(POINTER(IUnknown)), "iface"))]


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [COMMETHOD([], HRESULT, "ActivateCompleted",
                           (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "op"))]


class IAgileObject(IUnknown):  # el handler tiene que ser ágil (lo llama un hilo del pool MTA)
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class _Handler(COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.done = threading.Event()
        self.client = None
        self.hr = 0

    def ActivateCompleted(self, op):
        try:
            hr, unk = op.GetActivateResult()
            self.hr = hr
            if hr >= 0:
                self.client = unk.QueryInterface(IAudioClient)
        except Exception as e:  # noqa: BLE001 — se reporta desde el hilo que espera
            self.hr = getattr(e, "hresult", -1)
        finally:
            self.done.set()
        return 0


class _PROCESS_LOOPBACK_PARAMS(Structure):
    _fields_ = [("TargetProcessId", DWORD), ("ProcessLoopbackMode", c_int)]


class _ACTIVATION_PARAMS(Structure):
    _fields_ = [("ActivationType", c_int), ("params", _PROCESS_LOOPBACK_PARAMS)]


class _BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class _PROPVARIANT(Structure):
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("blob", _BLOB)]


def _activate(pid: int, mode: int):
    params = _ACTIVATION_PARAMS(1, _PROCESS_LOOPBACK_PARAMS(pid, mode))  # 1 = PROCESS_LOOPBACK
    pv = _PROPVARIANT(65, 0, 0, 0, _BLOB(ctypes.sizeof(params), ctypes.cast(byref(params), c_void_p)))  # VT_BLOB
    handler = _Handler()
    op = POINTER(IActivateAudioInterfaceAsyncOperation)()
    fn = WinDLL("Mmdevapi.dll").ActivateAudioInterfaceAsync
    fn.argtypes = [ctypes.c_wchar_p, POINTER(GUID), c_void_p,
                   POINTER(IActivateAudioInterfaceCompletionHandler),
                   POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]
    fn.restype = HRESULT
    fn("VAD\\Process_Loopback", byref(IAudioClient._iid_), byref(pv),
       handler.QueryInterface(IActivateAudioInterfaceCompletionHandler), byref(op))
    if not handler.done.wait(5):
        raise RuntimeError("process loopback: la activación no respondió")
    if handler.hr < 0 or handler.client is None:
        raise RuntimeError(f"process loopback: activación falló (hr=0x{handler.hr & 0xFFFFFFFF:08X})")
    return handler.client


class ProcessLoopbackCapture:
    """Audio del sistema excluyendo el árbol de procesos `pid` (por defecto, el nuestro)."""

    def __init__(self, pid: int | None = None, include: bool = False):
        self.rate, self.channels = RATE, CHANNELS
        self.pid = pid or os.getpid()
        self.mode = _INCLUDE if include else _EXCLUDE
        what = "solo" if include else "todo menos"
        self.device = {"name": f"Process loopback ({what} PID {self.pid})"}
        self.q: queue.Queue[tuple[float, bytes]] = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="cap-procloop", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error:
            raise self._error

    def start(self):  # compatible con Capture: arranca solo
        pass

    def close(self):
        self._stop.set()
        self._thread.join(2)

    def _run(self):
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        try:
            client = _activate(self.pid, self.mode)
            fmt = WAVEFORMATEX(3, CHANNELS, RATE, RATE * 8, 8, 32, 0)  # float32 estéreo 48 kHz
            client.Initialize(0, _LOOPBACK | _EVENTCALLBACK | _AUTOCONVERTPCM | _SRC_DEFAULT_QUALITY,
                              2_000_000, 0, byref(fmt), None)
            k32 = ctypes.WinDLL("kernel32")
            k32.CreateEventW.restype = HANDLE
            ev = k32.CreateEventW(None, False, False, None)
            client.SetEventHandle(ev)
            cap = client.GetService(byref(IAudioCaptureClient._iid_)).QueryInterface(IAudioCaptureClient)
            client.Start()
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        bpf = 4 * CHANNELS
        last = time.perf_counter()
        block = RATE // 100  # relleno de silencio en bloques de 10 ms
        try:
            while not self._stop.is_set():
                k32.WaitForSingleObject(ev, 20)
                got = False
                while cap.GetNextPacketSize() > 0:
                    data, frames, flags, _, _ = cap.GetBuffer()
                    if flags & _SILENT:
                        buf = bytes(frames * bpf)
                    else:
                        buf = ctypes.string_at(data, frames * bpf)
                    cap.ReleaseBuffer(frames)
                    now = time.perf_counter()
                    self.q.put((now, buf))
                    last, got = now, True
                # Sin nada sonando no llegan paquetes: se rellena con silencio para que el VAD
                # vea el final de la frase (si no, el segmentador nunca cierra).
                now = time.perf_counter()
                if not got and now - last > 0.1:
                    n = int((now - last) * RATE) // block * block
                    for _ in range(n // block):
                        self.q.put((now, bytes(block * bpf)))
                    last += n / RATE
        finally:
            client.Stop()
            k32.CloseHandle(ev)

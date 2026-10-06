"""Process loopback EXCLUDE: lo que suena desde OTRO proceso se captura, lo nuestro no.

Uso: python scripts/smoke_procloop.py
"""

import subprocess
import sys
import time
import winsound
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from voztrad.procloop import ProcessLoopbackCapture  # noqa: E402

WAV = str(ROOT / "samples" / "synth" / "A_en03.wav")


def grab(cap, seconds):
    end = time.perf_counter() + seconds
    blocks = []
    while time.perf_counter() < end:
        try:
            blocks.append(cap.q.get(timeout=0.05)[1])
        except Exception:  # noqa: BLE001
            pass
    x = np.frombuffer(b"".join(blocks), np.float32)
    return len(x) / 2 / 48000, float(np.abs(x).max()) if len(x) else 0.0


cap = ProcessLoopbackCapture()
print(cap.device["name"])
grab(cap, 0.5)
print("silencio:          %.2f s capturados, pico %.3f" % grab(cap, 1.0))
p = subprocess.Popen([sys.executable, "-c", f"import winsound; winsound.PlaySound(r'{WAV}', winsound.SND_FILENAME)"])
print("otro proceso suena: %.2f s capturados, pico %.3f" % grab(cap, 4.0))
p.wait()
winsound.PlaySound(WAV, winsound.SND_FILENAME | winsound.SND_ASYNC)
print("NOSOTROS sonamos:   %.2f s capturados, pico %.3f" % grab(cap, 4.0))
cap.close()

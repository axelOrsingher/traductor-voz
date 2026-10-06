"""Renderiza el overlay con eventos de ejemplo (sin motor) y guarda capturas PNG."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from voztrad.ui.overlay import Overlay  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out"
out.mkdir(parents=True, exist_ok=True)

EVENTS = [
    {"type": "state", "asr": "parakeet-tdt", "b_mode": "interp", "ptt_key": "F8", "ptt": False, "listen": True, "listen_available": True},
    {"type": "phrase", "dir": "A", "src": "Can you share your screen so we can look at the dashboard?",
     "tgt": "¿Podés compartir tu pantalla así vemos el tablero?", "lat": {"total": 784}},
    {"type": "phrase", "dir": "B", "src": "Hola Mateo, ¿cómo estás? ¿Todo tranquilo?",
     "tgt": "Hi Mateo, how are you? All good?", "lat": {"total": 447}},
    {"type": "phrase", "dir": "A", "src": "sí te escucho, tenemos de mujer igual", "tgt": "sí te escucho, tenemos de mujer igual",
     "passthrough": True, "lat": {"total": 650}},
    {"type": "phrase", "dir": "B", "src": "Me parece perfecto. Yo más tarde también voy a trabajar en EWT.",
     "tgt": "Sounds perfect. I'm also going to work on EWT later.", "lat": {"total": 512}},
    {"type": "phrase", "dir": "A", "src": "The latency numbers look much better after the last deploy.",
     "tgt": "Los números de latencia se ven mucho mejor después del último despliegue.", "lat": {"total": 812}},
    {"type": "voice", "dir": "A", "active": True},
    {"type": "partial", "dir": "A", "src": "So the plan for next week is", "tgt": "Así que el plan para la semana que viene es"},
    {"type": "ptt", "down": True},
]

app = QApplication(sys.argv)
w = Overlay(port=1)  # puerto inválido: no se conecta, solo renderiza
w._set_connected(True)
for ev in EVENTS:
    w.on_event(ev)


def shoot():
    for name, (wd, ht) in {"overlay_wide.png": (980, 320), "overlay_narrow.png": (440, 560)}.items():
        w.resize(wd, ht)
        app.processEvents()
        w.grab().save(str(out / name))
    w.close()
    app.quit()


w.show()
QTimer.singleShot(600, shoot)
app.exec()
print("ok ->", out)

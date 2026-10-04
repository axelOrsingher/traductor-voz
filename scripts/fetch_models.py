"""Baja los modelos que no vienen del HF cache (models/ no se versiona).

ASR (faster-whisper / parakeet) y MT (opus-mt CT2) se bajan solos al primer uso.
Uso: python scripts/fetch_models.py
"""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
FILES = {
    "models/silero_vad.onnx":
        "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx",
    "models/en_US-lessac-medium.onnx": f"{PIPER}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "models/en_US-lessac-medium.onnx.json": f"{PIPER}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    # voz argentina: solo para generar audio de prueba (samples/synth)
    "models/test-voices/es_AR-daniela-high.onnx": f"{PIPER}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
    "models/test-voices/es_AR-daniela-high.onnx.json": f"{PIPER}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
}

for rel, url in FILES.items():
    dst = ROOT / rel
    if dst.exists():
        print(f"ok    {rel}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"baja  {rel}")
    urllib.request.urlretrieve(url, dst)

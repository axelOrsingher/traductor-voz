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
    # escuchar a los otros en castellano: ald (default, 90 ms) y daniela (rioplatense, 316 ms)
    "models/es_MX-ald-medium.onnx": f"{PIPER}/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
    "models/es_MX-ald-medium.onnx.json": f"{PIPER}/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
    "models/es_AR-daniela-high.onnx": f"{PIPER}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
    "models/es_AR-daniela-high.onnx.json": f"{PIPER}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
    # fin de turno (Pipecat Smart Turn v3.2, BSD-2)
    "models/smart-turn-v3.2-cpu.onnx":
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
    # traducción refinada (TranslateGemma-4B, licencia Gemma) — corre en llama.cpp (ver abajo)
    "models/translategemma-4b-it.Q4_K_M.gguf":
        "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf",
    # voz argentina: también para generar audio de prueba (samples/synth)
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

# llama.cpp (binarios oficiales CUDA 12.4) para TranslateGemma
LLAMA = Path.home() / "tools" / "llama.cpp-b11433-cuda12.4"
REL = "https://github.com/ggml-org/llama.cpp/releases/download/b11433/"
if not (LLAMA / "llama-server.exe").exists():
    import zipfile
    LLAMA.mkdir(parents=True, exist_ok=True)
    for z in ("llama-b11433-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"):
        print(f"baja  {z}")
        tmp = LLAMA / z
        urllib.request.urlretrieve(REL + z, tmp)
        zipfile.ZipFile(tmp).extractall(LLAMA)
        tmp.unlink()
else:
    print(f"ok    {LLAMA}")

if not (ROOT / "models" / "parakeet-tdt-0.6b-v3-fp16" / "encoder-model.onnx").exists():
    print("\nFALTA Parakeet fp16: ver scripts/convert_parakeet_fp16.py (venv aparte). Sin él se usa fp32.")

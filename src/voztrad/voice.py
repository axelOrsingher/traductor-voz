"""Grabación guiada de tu voz para clonarla (Pocket TTS).

  python -m voztrad voice            # graba castellano + inglés, arma previews y los reproduce
  python -m voztrad voice use en     # elige la referencia: es | en | mix

Guarda en models/voice/ (no se versiona): me_es.wav, me_en.wav, me_mix.wav y
me.wav (la elegida, la que usa el motor).
"""

from __future__ import annotations

import shutil
import sys
import time
import winsound
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

from .audio import find_input
from .config import MODELS, ROOT

VOICE = MODELS / "voice"
SECONDS = 20

READ_ES = (
    "Hola, ¿cómo andan? Les cuento un poco cómo venimos con el proyecto. Esta semana "
    "terminamos la parte del backend y arrancamos con las pruebas. Mañana a la mañana lo "
    "repasamos todos juntos, y si les parece, el viernes hacemos la demo con el cliente. "
    "Cualquier cosa me escriben, ¿dale?"
)
READ_EN = (
    "Hi everyone, thanks for joining. Let me give you a quick update on the project. This week "
    "we finished the backend and started testing. Tomorrow morning we'll review everything "
    "together, and if that works for you, we'll do the demo with the client on Friday. "
    "If you have any questions, just let me know."
)
PREVIEW_TEXT = "Hi, this is my voice speaking English through the translator. Does it sound like me?"


def _record(pa: pyaudio.PyAudio, dev: dict, seconds: int) -> tuple[np.ndarray, int]:
    rate, ch = int(dev["defaultSampleRate"]), max(1, int(dev["maxInputChannels"]))
    s = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                input_device_index=dev["index"], frames_per_buffer=rate // 10)
    frames = []
    for i in range(seconds * 10):
        frames.append(s.read(rate // 10, exception_on_overflow=False))
        if i % 10 == 0:
            left = seconds - i // 10
            lvl = np.abs(np.frombuffer(frames[-1], np.float32)).max()
            bar = "#" * int(min(1.0, lvl * 3) * 30)
            print(f"\r  grabando… {left:2d}s  nivel {bar:<30}", end="", flush=True)
    s.close()
    print("\r" + " " * 60 + "\r", end="")
    x = np.frombuffer(b"".join(frames), np.float32).reshape(-1, ch).mean(axis=1)
    return x, rate


def _trim_and_check(x: np.ndarray, sr: int) -> tuple[np.ndarray, list[str]]:
    warn = []
    peak = float(np.abs(x).max())
    if peak > 0.98:
        warn.append("satura (clipping): alejate un poco del micrófono o bajá la ganancia")
    if peak < 0.05:
        warn.append("muy bajo: acercate al micrófono o subí la ganancia")
    # recortar silencio de los bordes (bloques de 50 ms bajo 2% del pico)
    blk = sr // 20
    e = np.array([np.abs(x[i:i + blk]).max() for i in range(0, len(x), blk)])
    voiced = np.where(e > 0.02 * max(peak, 1e-6) + 0.005)[0]
    if len(voiced):
        x = x[max(0, voiced[0] - 2) * blk:(voiced[-1] + 3) * blk]
    if len(x) / sr < 8:
        warn.append(f"quedaron solo {len(x) / sr:.0f} s de voz: conviene repetir")
    return x, warn


def record(mic: str | None = None):
    VOICE.mkdir(parents=True, exist_ok=True)
    pa = pyaudio.PyAudio()
    dev = find_input(pa, mic)
    print(f"\nMicrófono: {dev['name']}\n"
          "Buscá un lugar silencioso. Hablá natural, como en una llamada.\n")
    takes = {}
    for key, title, text in (("es", "CASTELLANO", READ_ES), ("en", "INGLÉS (como te salga, no importa el acento)", READ_EN)):
        while True:
            print(f"── {title} ── leé esto en voz alta ({SECONDS} s):\n\n  {text}\n")
            input("  Enter para empezar…")
            x, sr = _record(pa, dev, SECONDS)
            x, warn = _trim_and_check(x, sr)
            for w in warn:
                print(f"  ⚠ {w}")
            ans = input("  ¿Guardar esta toma? [Enter = sí, r = repetir] ").strip().lower()
            if ans != "r":
                break
        sf.write(VOICE / f"me_{key}.wav", x, sr)
        takes[key] = (x, sr)
        print(f"  ✓ me_{key}.wav ({len(x) / sr:.0f} s)\n")
    pa.terminate()
    (xes, sres), (xen, sren) = takes["es"], takes["en"]
    if sres == sren:
        sf.write(VOICE / "me_mix.wav", np.concatenate([xes, np.zeros(sres // 3, np.float32), xen]), sres)
    previews()


def previews():
    """Genera y reproduce la misma frase con cada referencia, para elegir."""
    from .tts_pocket import PocketTTS
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    refs = [k for k in ("en", "es", "mix") if (VOICE / f"me_{k}.wav").exists()]
    print("Generando muestras de tu voz clonada (tarda unos segundos cada una)…")
    for k in refs:
        tts = PocketTTS(VOICE / f"me_{k}.wav")
        t = time.perf_counter()
        audio = np.concatenate(list(tts.stream(PREVIEW_TEXT)))
        path = out / f"voice_preview_{k}.wav"
        sf.write(path, audio, tts.sr)
        print(f"  ▶ referencia '{k}' ({time.perf_counter() - t:.1f} s): {path.name}")
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        time.sleep(0.4)
    best = "en" if "en" in refs else refs[0]
    use(best, quiet=True)
    print(f"\nQuedó elegida '{best}'. Para cambiarla:  python -m voztrad voice use es|en|mix\n"
          f"Para volver a escucharlas: abrí out\\voice_preview_*.wav")


def use(key: str, quiet: bool = False):
    src = VOICE / f"me_{key}.wav"
    if not src.exists():
        sys.exit(f"no existe {src}")
    shutil.copyfile(src, VOICE / "me.wav")
    if not quiet:
        print(f"Voz del motor: {src.name} -> me.wav (reiniciá la app para usarla)")

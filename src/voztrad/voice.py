"""Guided recording of your voice for cloning (Pocket TTS).

  python -m voztrad voice            # record Spanish + English, build previews and play them
  python -m voztrad voice use en     # pick the reference: es | en | mix

Saves to models/voice/ (not versioned): me_es.wav, me_en.wav, me_mix.wav and
me.wav (the chosen one, used by the engine). Recording and playback go through the
platform audio backend (audio.py), so this works wherever the engine does.
"""

from __future__ import annotations

import shutil
import sys
import time

import numpy as np
import soundfile as sf

from .audio import CablePlayer, Capture, PyAudio, find_default_output, find_input
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


def _record(pa, dev: dict, seconds: int) -> tuple[np.ndarray, int]:
    cap = Capture(pa, dev)
    rate, ch = cap.rate, cap.channels
    need = seconds * rate * ch  # float32 samples, interleaved
    chunks, have, next_tick = [], 0, 0
    cap.start()
    try:
        while have < need:
            _, data = cap.q.get(timeout=5)
            x = np.frombuffer(data, np.float32)
            chunks.append(x)
            have += len(x)
            if have >= next_tick:  # once per second
                left = seconds - have // (rate * ch)
                bar = "#" * int(min(1.0, float(np.abs(x).max()) * 3) * 30)
                print(f"\r  grabando… {left:2d}s  nivel {bar:<30}", end="", flush=True)
                next_tick += rate * ch
    finally:
        cap.close()
    print("\r" + " " * 60 + "\r", end="")
    x = np.concatenate(chunks)[:need]
    x = x[:len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), rate


def _play(pa, audio: np.ndarray, sr: int):
    """Play on the default output and block until it has finished."""
    player = CablePlayer(pa, find_default_output(pa))
    player.play(audio, sr)
    while player.queued_s > 0:
        time.sleep(0.05)
    time.sleep(0.2)  # device buffer
    player.close()


def _trim_and_check(x: np.ndarray, sr: int) -> tuple[np.ndarray, list[str]]:
    warn = []
    peak = float(np.abs(x).max())
    if peak > 0.98:
        warn.append("satura (clipping): alejate un poco del micrófono o bajá la ganancia")
    if peak < 0.05:
        warn.append("muy bajo: acercate al micrófono o subí la ganancia")
    # trim silence at the edges (50 ms blocks under 2% of the peak)
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
    pa = PyAudio()
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
    """Synthesize and play the same sentence with each reference, to pick one."""
    from .tts_pocket import PocketTTS
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    refs = [k for k in ("en", "es", "mix") if (VOICE / f"me_{k}.wav").exists()]
    print("Generando muestras de tu voz clonada (tarda unos segundos cada una)…")
    pa = PyAudio()
    for k in refs:
        tts = PocketTTS(VOICE / f"me_{k}.wav")
        t = time.perf_counter()
        audio = np.concatenate(list(tts.stream(PREVIEW_TEXT)))
        path = out / f"voice_preview_{k}.wav"
        sf.write(path, audio, tts.sr)
        print(f"  ▶ referencia '{k}' ({time.perf_counter() - t:.1f} s): {path.name}")
        _play(pa, audio, tts.sr)
        time.sleep(0.4)
    pa.terminate()
    best = "en" if "en" in refs else refs[0]
    use(best, quiet=True)
    print(f"\nQuedó elegida '{best}'. Para cambiarla:  python -m voztrad voice use es|en|mix\n"
          f"Para volver a escucharlas: abrí out/voice_preview_*.wav")


def use(key: str, quiet: bool = False):
    src = VOICE / f"me_{key}.wav"
    if not src.exists():
        sys.exit(f"no existe {src}")
    shutil.copyfile(src, VOICE / "me.wav")
    if not quiet:
        print(f"Voz del motor: {src.name} -> me.wav (reiniciá la app para usarla)")

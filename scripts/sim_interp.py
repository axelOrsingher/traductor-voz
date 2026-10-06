"""Simula manos libres (vad) vs intérprete (interp) sobre el MISMO audio, con modelos reales.

Reloj virtual de audio (cada frame = 32 ms) + tiempos de cómputo MEDIDOS de ASR, MT y TTS.
La salida es secuencial como en el motor: un fragmento no empieza a sonar hasta que
terminó el anterior. Reporta cuándo empieza a sonar el inglés respecto de cuándo empezaste
a hablar, y el retraso de cada fragmento.

Uso: python scripts/sim_interp.py [wav]   (default: samples/synth/B_long.wav)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voztrad.winperf import disable_throttling  # noqa: E402

disable_throttling()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

from voztrad import asr as asr_mod  # noqa: E402
from voztrad.audio import Framer  # noqa: E402
from voztrad.config import FRAME_MS, SR, Config, SegmenterConfig  # noqa: E402
from voztrad.mt import Translator  # noqa: E402
from voztrad.tempo import rate_for_backlog  # noqa: E402
from voztrad.tts import load_tts  # noqa: E402
from voztrad.vad import Segmenter  # noqa: E402
from voztrad.voseo import normalize  # noqa: E402

wav = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "synth" / "B_long.wav"
x, sr = sf.read(wav, dtype="float32")
x = soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x
x = np.concatenate([np.zeros(SR // 2, np.float32), x, np.zeros(SR * 2, np.float32)])

asr = asr_mod.load("parakeet")
mt = Translator("es", "en")
mt.translate("hola")
tts = load_tts(Config())


from voztrad.smartturn import SmartTurn  # noqa: E402

smart_turn = SmartTurn()


def segments(interp: bool, smart: bool = False):
    seg = Segmenter("B", SegmenterConfig())
    seg.interp = interp
    seg.turn = smart_turn if smart else None
    x48 = np.repeat(soxr.resample(x, SR, 48000).astype(np.float32)[:, None], 2, axis=1)
    fr = Framer(48000, 2)
    out, n, start = [], 0, None
    for i in range(0, len(x48), 480):
        for f in fr.push(0.0, x48[i:i + 480].tobytes()):
            n += 1
            if seg.speaking and start is None:
                start = (n - len(seg.frames)) * FRAME_MS / 1000
            if (s := seg.push(f)) is not None:
                out.append((n * FRAME_MS / 1000, s))  # (instante virtual de emisión, segmento)
    return start, out


for mode, tempo, smart in (("vad", False, False), ("vad", False, True), ("interp", True, False),
                           ("interp", True, True)):
    speech_start, segs = segments(mode == "interp", smart)
    mode = mode + (" + SMART TURN" if smart else "")
    free_at = 0.0       # pipeline (ASR->MT->TTS) libre
    play_end = 0.0      # fin de lo que ya está sonando
    rows = []
    for t_emit, s in segs:
        t0 = max(t_emit, free_at)
        c = time.perf_counter()
        text = asr.transcribe(s.audio, "es")
        en = mt.translate(normalize(text))
        chunks, first = [], None
        for ch in tts.stream(en):
            if first is None:
                first = time.perf_counter() - c
            chunks.append(ch)
        compute = time.perf_counter() - c
        ready = t0 + first
        rate = rate_for_backlog(max(0.0, play_end - ready)) if tempo else 1.0
        dur = sum(len(ch) for ch in chunks) / tts.sr / rate
        start = max(ready, play_end)
        play_end = start + dur
        free_at = t0 + compute
        rows.append((t_emit, start, dur, text, en))
    print(f"\n=== {mode.upper()}{' + TEMPO' if tempo else ''}  ({len(rows)} fragmento{'s' if len(rows) != 1 else ''})")
    for t_emit, start, dur, text, en in rows:
        print(f"  cierre {t_emit - speech_start:5.2f}s → suena {start - speech_start:5.2f}s "
              f"(+{(start - t_emit) * 1000:4.0f} ms) dura {dur:4.1f}s | {text}\n{'':48}→ {en}")
    speech_end = speech_start + (len(x) / SR - 2.5 - 0.5)
    print(f"  ▶ primer inglés a los {rows[0][1] - speech_start:.2f}s de empezar a hablar; "
          f"último termina {play_end - speech_end:+.2f}s respecto de cuando dejaste de hablar")

"""Tests del segmentador sobre audio real (samples/synth) pasado por el Framer 48 kHz estéreo."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from voztrad.audio import Framer
from voztrad.config import SR, SegmenterConfig
from voztrad.vad import PushToTalk, Segmenter

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "synth"


def _frames(x16: np.ndarray, lead_s=1.0, tail_s=1.5):
    x = np.concatenate([np.zeros(int(lead_s * SR), np.float32), x16, np.zeros(int(tail_s * SR), np.float32)])
    x48 = np.repeat(soxr.resample(x, SR, 48000).astype(np.float32)[:, None], 2, axis=1)
    fr = Framer(48000, 2)
    for i in range(0, len(x48), 480):  # bloques de 10 ms como el callback
        yield from fr.push(i / 48000, x48[i:i + 480].tobytes())


def _load(name):
    x, sr = sf.read(SAMPLES / name, dtype="float32")
    return x if sr == SR else soxr.resample(x, sr, SR).astype(np.float32)


def test_una_frase_un_segmento():
    x = _load("A_en02.wav")
    seg = Segmenter("A", SegmenterConfig())
    segs = [s for f in _frames(x) if (s := seg.push(f))]
    assert len(segs) == 1
    s = segs[0]
    # el wav trae ~0.4 s de silencio propio; voz ~2.1 s + pre-roll 320 ms + cola 96 ms
    assert 2.2 < s.dur_s < 2.8
    assert not s.forced


def test_pausa_entre_oraciones_parte_la_frase():
    seg = Segmenter("A", SegmenterConfig())
    segs = [s for f in _frames(_load("A_smoke.wav")) if (s := seg.push(f))]
    assert len(segs) == 2 and not any(s.forced for s in segs)


def test_descarta_ruido_corto():
    click = np.zeros(int(0.15 * SR), np.float32)
    click[::40] = 0.8
    seg = Segmenter("A", SegmenterConfig())
    assert not [s for f in _frames(click) if (s := seg.push(f))]


def test_corte_forzado_a_8s_con_solape():
    x = np.concatenate([_load("A_smoke.wav")] * 2)  # ~16 s
    cfg = SegmenterConfig(hangover_ms=3000)  # que las pausas no cierren la frase
    seg = Segmenter("A", cfg)
    segs = [s for f in _frames(x) if (s := seg.push(f))]
    assert segs[0].forced and abs(segs[0].dur_s - cfg.max_segment_ms / 1000) < 0.1
    assert len(segs) >= 2


def test_ptt_marca_inicio_y_fin():
    x = _load("B_rio1.wav")
    ptt = PushToTalk("B", max_ms=28000)
    frames = list(_frames(x, lead_s=0.5, tail_s=0.5))
    n_on = len(frames) - 10  # soltar la tecla poco antes del final
    segs = []
    for i, f in enumerate(frames):
        pressed = 10 <= i < n_on
        if (s := ptt.push(f, pressed, f.t_cb if not pressed else None)):
            segs.append(s)
    assert len(segs) == 1 and not segs[0].forced

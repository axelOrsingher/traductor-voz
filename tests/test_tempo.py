"""WSOLA: acelera sin cambiar el tono, en streaming (trozos de 80 ms como Pocket TTS)."""

import numpy as np
import pytest

from voztrad.tempo import Wsola, rate_for_backlog

SR = 24000


def _stream(x, w, chunk=1920):
    out = [w.push(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    out.append(w.push(np.zeros(0, np.float32), last=True))
    return np.concatenate(out)


def _f0(x):
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[int(np.argmax(sp))]


@pytest.mark.parametrize("rate", [1.0, 1.15, 1.3])
def test_duracion_y_tono(rate):
    t = np.arange(2 * SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = _stream(x, Wsola(rate, SR))
    assert abs(len(y) / len(x) - 1 / rate) < 0.04       # dura 1/rate
    assert abs(_f0(y[SR // 10: -SR // 10]) - 220) < 3    # mismo tono (no "ardilla")
    assert np.isfinite(y).all() and np.abs(y).max() < 1.0


def test_trozos_no_cambian_el_resultado():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32) * 0.1
    a = _stream(x, Wsola(1.2, SR), chunk=1920)
    b = _stream(x, Wsola(1.2, SR), chunk=777)
    n = min(len(a), len(b))
    assert np.allclose(a[:n], b[:n], atol=1e-5)


def test_tempo_segun_cola():
    assert rate_for_backlog(0.2) == 1.0
    assert rate_for_backlog(1.5) == 1.15
    assert rate_for_backlog(4.0) == 1.3

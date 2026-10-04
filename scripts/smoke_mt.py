"""Smoke test de MT: opus-mt en<->es vía CTranslate2 int8 en CPU.

Evalúa las frases rioplatenses (es->en) con y sin normalización de voseo,
y unas frases de reunión en inglés (en->es). Mide ms por frase.
"""

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voztrad.mt import Translator  # noqa: E402
from voztrad.voseo import normalize  # noqa: E402

RIO = [
    "Che, ¿vos tenés el informe que te mandé ayer?",
    "Dale, lo vemos en la próxima reunión y después te aviso.",
    "No sé si se entiende, pero la idea sería arrancar por el backend.",
    "Mirá, yo lo que haría es fijarme primero si el cliente ya lo tiene.",
    "Está bárbaro, pero habría que chequearlo con el equipo de infra.",
]
EN = [
    "Hi everyone, thanks for joining.",
    "Let's start with the backend status and then review the deployment plan for next week.",
    "Can you share your screen?",
    "I think we should push the release to Monday.",
    "Sorry, you're on mute.",
]


def bench(tr: Translator, sents, label, transform=lambda s: s):
    print(f"\n== {label}")
    times = []
    for s in sents:
        src = transform(s)
        t = time.perf_counter()
        out = tr.translate(src)
        times.append((time.perf_counter() - t) * 1000)
        print(f"  {times[-1]:5.0f} ms | {s}")
        if src != s:
            print(f"           norm: {src}")
        print(f"           -> {out}")
    print(f"  p50 {statistics.median(times):.0f} ms | max {max(times):.0f} ms")


es_en = Translator("es", "en")
en_es = Translator("en", "es")
es_en.translate("hola")  # warm-up
en_es.translate("hello")
bench(es_en, RIO, "es->en (crudo)")
bench(es_en, RIO, "es->en (voseo normalizado)", normalize)
bench(en_es, EN, "en->es")

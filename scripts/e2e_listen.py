"""E2E de "escuchar en castellano": este proceso hace de Teams (reproduce inglés), el motor
traduce, lo dice en castellano por los auriculares, baja Teams y NO se re-captura.

Mide desde afuera (loopback común de la salida) cuándo empieza a sonar el castellano
después de que termina cada frase en inglés.

Uso: python scripts/e2e_listen.py [--rounds 1]
"""

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import winsound
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pyaudiowpatch as pyaudio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--port", type=int, default=8766)
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

log = open(ROOT / "out" / "e2e-listen-engine.txt", "w", encoding="utf-8")
eng = subprocess.Popen([sys.executable, "-m", "voztrad", "run", "--no-b", "--port", str(args.port)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"})
events: queue.Queue = queue.Queue()
for _ in range(600):
    try:
        sock = socket.create_connection(("127.0.0.1", args.port))
        break
    except OSError:
        if eng.poll() is not None:
            sys.exit(f"el motor murió, ver {log.name}")
        time.sleep(0.5)


def reader():
    buf = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            events.put(json.loads(line))


threading.Thread(target=reader, daemon=True).start()
all_events = []


def drain(timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            all_events.append(events.get(timeout=0.05))
        except queue.Empty:
            pass


while not any(e.get("type") == "status" and "escuchando" in e.get("msg", "") for e in all_events):
    drain(0.5)
    if eng.poll() is not None:
        sys.exit(f"el motor murió, ver {log.name}")
print("motor listo")

# loopback COMÚN de la salida: oye todo (inglés de "Teams" + castellano del motor)
pa = pyaudio.PyAudio()
lb = pa.get_default_wasapi_loopback()
rate, ch = int(lb["defaultSampleRate"]), lb["maxInputChannels"]
rq: queue.Queue = queue.Queue()
rec = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True, input_device_index=lb["index"],
              frames_per_buffer=rate // 200,
              stream_callback=lambda d, n, ti, st: (rq.put((time.perf_counter(), d)), (None, pyaudio.paContinue))[1])

from pycaw.pycaw import AudioUtilities  # noqa: E402

from voztrad.ducker import Ducker  # noqa: E402

probe = Ducker()


def teams_volume():
    v = [s.SimpleAudioVolume.GetMasterVolume() for s in AudioUtilities.GetAllSessions()
         if s.ProcessId and probe._matches(s.ProcessId)]
    return round(min(v), 2) if v else None


wavs = sorted((ROOT / "samples" / "synth").glob("A_en0*.wav"))
results = []
for _ in range(args.rounds):
    for w in wavs:
        while not rq.empty():
            rq.get()
        winsound.PlaySound(str(w), winsound.SND_FILENAME)   # "Teams" habla (bloquea hasta terminar)
        t_end = time.perf_counter()
        vols = []
        for _ in range(30):                                   # 3 s mirando volumen de Teams
            vols.append(teams_volume())
            time.sleep(0.1)
        blocks = []
        while not rq.empty():
            blocks.append(rq.get())
        onset = next((t - len(np.frombuffer(b, np.float32)) / ch / rate for t, b in blocks
                      if t > t_end + 0.05 and np.abs(np.frombuffer(b, np.float32)).max() > 0.02), None)
        results.append((w.name, (onset - t_end) * 1000 if onset else None, min(v for v in vols if v is not None) if any(vols) else None, vols[-1]))
        time.sleep(0.8)

drain(2)
sock.sendall(b'{"cmd": "stop"}\n')
eng.wait(30)
rec.close()
pa.terminate()

ph = [e for e in all_events if e.get("type") == "phrase" and e["dir"] == "A"]
ls = [e for e in all_events if e.get("type") == "listen"]
pa_ = [e for e in all_events if e.get("type") == "partial"]
rf_ = [e for e in all_events if e.get("type") == "refine"]
print(f"refinados: {len(rf_)}" + (f" | p50 {np.percentile([e['ms'] for e in rf_], 50):.0f} ms" if rf_ else ""))
for e in rf_[:5]:
    print(f"  rápida: {e['fast']}")
    print(f"  buena : {e['tgt']}")
print(f"parciales en vivo: {len(pa_)}", [p["tgt"][:40] for p in pa_[:4]])
print(f"\nfrases A: {len(ph)} (se reprodujeron {len(wavs) * args.rounds}); "
      f"en castellano re-capturadas (eco): {sum(e.get('passthrough') for e in ph)}")
for name, ms, vmin, vend in results:
    print(f"  {name}: castellano suena {ms and round(ms)} ms después de terminar el inglés | "
          f"Teams min {vmin} -> al final {vend}")
ext = [r[1] for r in results if r[1] is not None]
if ext:
    print(f"externo (fin del inglés -> castellano audible): p50 {np.percentile(ext, 50):.0f} / p95 {np.percentile(ext, 95):.0f} ms")
if ls:
    print(f"motor (fin de frase -> 1er audio castellano): p50 {np.percentile([e['total_ms'] for e in ls], 50):.0f} ms")
for e in ph[:4]:
    print(f"  EN: {e['src']}\n  ES: {e['tgt']}")
print(f"Teams ahora: {teams_volume()}")

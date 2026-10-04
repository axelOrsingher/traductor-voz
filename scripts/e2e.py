"""Prueba end-to-end del motor en vivo, sin intervención humana.

A: reproduce samples/<dir>/A_*.wav por los parlantes -> el motor los toma por loopback.
B: el motor usa "CABLE Output" como micrófono. El harness aprieta PTT por TCP,
   inyecta B_*.wav en "CABLE Input", suelta PTT, y graba "CABLE Output" (lo que
   oiría Teams) para medir desde afuera cuándo empieza a sonar el inglés y
   transcribirlo.

Uso: python scripts/e2e.py [--asr whisper|parakeet] [--rounds 2] [--dir samples/synth]
"""

import argparse
import json
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
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

from voztrad.audio import find_input, find_output  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--asr", default="whisper")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--dir", default=str(ROOT / "samples" / "synth"))
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--engine-args", default="", help="flags extra para voztrad run")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

tag = args.asr + args.engine_args.replace(" ", "").replace("--", "_")
log = open(ROOT / "out" / f"e2e-engine-{tag}.txt", "w", encoding="utf-8")
eng = subprocess.Popen([sys.executable, "-m", "voztrad", "run", "--asr", args.asr,
                        "--mic", "CABLE Output", "--port", str(args.port), *args.engine_args.split()],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"),
                            "PYTHONIOENCODING": "utf-8"})

events: queue.Queue = queue.Queue()
for _ in range(600):
    try:
        sock = socket.create_connection(("127.0.0.1", args.port))
        break
    except OSError:
        if eng.poll() is not None:
            sys.exit(f"el motor murió, ver {log.name}")
        time.sleep(0.5)
else:
    sys.exit("no pude conectar al motor")


def _recv():
    try:
        return sock.recv(65536)
    except OSError:  # el motor cerró al terminar
        return b""


def reader():
    buf = b""
    while chunk := _recv():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            events.put(json.loads(line))


threading.Thread(target=reader, daemon=True).start()


def send(cmd):
    sock.sendall((json.dumps(cmd) + "\n").encode())


def wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            ev = events.get(timeout=0.1)
        except queue.Empty:
            continue
        if pred(ev):
            return ev
    return None


if not wait_for(lambda e: e.get("type") == "status" and "escuchando" in e.get("msg", ""), 300):
    sys.exit("el motor no llegó a 'escuchando'")
print(f"motor listo ({args.asr})")
time.sleep(1.0)

d = Path(args.dir)
a_wavs = sorted(d.glob("A_*.wav"))
b_wavs = sorted(d.glob("B_*.wav"))
phrases = []

# ------------------------------------------------------------------ A
for r in range(args.rounds):
    for w in a_wavs:
        winsound.PlaySound(str(w), winsound.SND_FILENAME)
        ev = wait_for(lambda e: e.get("type") == "phrase" and e["dir"] == "A", 6)
        if ev is None:
            print(f"A {w.name}: SIN FRASE")
        else:
            phrases.append(ev)
            # frases largas pueden partirse: juntar lo que llegue enseguida
            while (more := wait_for(lambda e: e.get("type") == "phrase" and e["dir"] == "A", 1.2)):
                phrases.append(more)
        time.sleep(0.4)

# ------------------------------------------------------------------ B
rec_q: queue.Queue = queue.Queue()
pa = pyaudio.PyAudio()
cin = find_input(pa, "CABLE Output")
rate, ch = int(cin["defaultSampleRate"]), cin["maxInputChannels"]
rec = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
              input_device_index=cin["index"], frames_per_buffer=rate // 200,
              stream_callback=lambda data, n, ti, st: (rec_q.put((time.perf_counter(), data)), (None, pyaudio.paContinue))[1])
out_dev = find_output(pa, "CABLE Input")
orate, och = int(out_dev["defaultSampleRate"]), min(2, out_dev["maxOutputChannels"])
play = pa.open(format=pyaudio.paFloat32, channels=och, rate=orate, output=True,
               output_device_index=out_dev["index"])

from faster_whisper import WhisperModel  # noqa: E402
checker = WhisperModel("base", device="cpu", compute_type="int8")  # verificador independiente, en CPU

external = []
for r in range(args.rounds):
    for w in b_wavs:
        x, sr = sf.read(w, dtype="float32")
        x = np.repeat(soxr.resample(x, sr, orate).astype(np.float32)[:, None], och, axis=1)
        send({"cmd": "ptt", "down": True})
        time.sleep(0.05)
        play.write(x.tobytes())
        time.sleep(0.15)  # que termine de drenar el buffer del cable
        while not rec_q.empty():
            rec_q.get()
        t_up = time.perf_counter()
        send({"cmd": "ptt", "down": False})
        ev = wait_for(lambda e: e.get("type") == "phrase" and e["dir"] == "B", 10)
        time.sleep(4.5)  # dejar que el TTS termine de sonar
        blocks = []
        while not rec_q.empty():
            blocks.append(rec_q.get())
        audio = np.concatenate([np.frombuffer(b, np.float32).reshape(-1, ch).mean(axis=1) for _, b in blocks])
        onset = next((t - len(np.frombuffer(b, np.float32)) / ch / rate for t, b in blocks
                      if np.abs(np.frombuffer(b, np.float32)).max() > 0.02 and t > t_up), None)
        heard = " ".join(s.text.strip() for s in checker.transcribe(
            soxr.resample(audio, rate, 16000).astype(np.float32), language="en", beam_size=1)[0])
        ext_ms = (onset - t_up) * 1000 if onset else None
        external.append(ext_ms)
        if ev:
            phrases.append(ev)
        print(f"B {w.name}: motor={ev['lat']['total'] if ev else None} ms | "
              f"medido en CABLE Output={ext_ms and round(ext_ms)} ms | Teams oiría: {heard!r}")

rec.close()
play.close()
pa.terminate()
send({"cmd": "stop"})
eng.wait(timeout=30)
log.close()

print()
for ev in phrases:
    print(f"[{ev['dir']}] {ev['src']}\n     -> {ev['tgt']}\n     {ev['lat']}")
print(Path(log.name).read_text(encoding="utf-8").split("=" * 72)[-1])
ext = [e for e in external if e is not None]
if ext:
    print(f"B externo (soltar PTT -> primer audio en CABLE Output): "
          f"p50 {np.percentile(ext, 50):.0f} / p95 {np.percentile(ext, 95):.0f} ms  (n={len(ext)})")

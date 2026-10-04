"""Paso 4: capturar 5 s de la salida del sistema por WASAPI loopback.

Reproduce samples/en_smoke.wav por el dispositivo por defecto mientras graba el
loopback de ese mismo dispositivo. El callback solo hace put() en una cola.
Guarda el crudo (48 kHz estéreo) y la versión 16 kHz mono vía soxr, y la
transcribe con whisper para verificar que se capturó bien.

Uso: python scripts/smoke_loopback.py [--no-play] [--seconds 5]
"""

import argparse
import queue
import sys
import time
import winsound
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pyaudiowpatch as pyaudio  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=5.0)
ap.add_argument("--no-play", action="store_true", help="no reproducir el wav; capturar lo que suene")
args = ap.parse_args()

out_dir = ROOT / "out"
out_dir.mkdir(exist_ok=True)

q: queue.Queue[bytes] = queue.Queue()


def callback(in_data, frame_count, time_info, status):
    q.put(in_data)
    return (None, pyaudio.paContinue)


with pyaudio.PyAudio() as pa:
    dev = pa.get_default_wasapi_loopback()
    rate, ch = int(dev["defaultSampleRate"]), dev["maxInputChannels"]
    print(f"loopback: [{dev['index']}] {dev['name']} {rate} Hz x{ch}")
    stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                     input_device_index=dev["index"], frames_per_buffer=rate // 100,
                     stream_callback=callback)
    stream.start_stream()
    if not args.no_play:
        winsound.PlaySound(str(ROOT / "samples" / "en_smoke.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
    time.sleep(args.seconds)
    stream.stop_stream()
    stream.close()
    winsound.PlaySound(None, 0)

raw = np.frombuffer(b"".join(q.queue), dtype=np.float32)
raw = raw.reshape(-1, ch)
mono16 = soxr.resample(raw.mean(axis=1), rate, 16000).astype(np.float32)

sf.write(out_dir / "loopback_raw.wav", raw, rate)
sf.write(out_dir / "loopback_16k.wav", mono16, 16000)
dur = len(raw) / rate
rms = float(np.sqrt(np.mean(mono16 ** 2))) if len(mono16) else 0.0
print(f"capturado {dur:.2f} s | peak {np.abs(raw).max():.3f} | rms {rms:.4f}")
print(f"-> {out_dir / 'loopback_raw.wav'}")

from voztrad import cuda_dlls  # noqa: E402,F401
from faster_whisper import WhisperModel  # noqa: E402

model = WhisperModel("base", device="cuda", compute_type="float16")
segs, _ = model.transcribe(mono16, language="en", beam_size=1, condition_on_previous_text=False)
print("whisper:", " ".join(s.text.strip() for s in segs) or "<vacío>")

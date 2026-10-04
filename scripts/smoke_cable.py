"""Paso 5: reproducir un wav hacia VB-Cable y verificar que otro proceso lo
oye como micrófono ("CABLE Output"), igual que lo haría Teams.

Uso: python scripts/smoke_cable.py            (orquesta: lanza el grabador y reproduce)
     python scripts/smoke_cable.py --record N  (modo interno: graba N s de CABLE Output)
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pyaudiowpatch as pyaudio  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402


def find(pa, name, want_input):
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (d["hostApi"] == wasapi and name in d["name"] and not d.get("isLoopbackDevice")
                and (d["maxInputChannels"] if want_input else d["maxOutputChannels"])):
            return d
    raise RuntimeError(f"no encontré {name!r}")


def record(seconds: float) -> None:
    with pyaudio.PyAudio() as pa:
        d = find(pa, "CABLE Output", want_input=True)
        rate, ch = int(d["defaultSampleRate"]), d["maxInputChannels"]
        s = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                    input_device_index=d["index"], frames_per_buffer=rate // 100)
        frames = [s.read(rate // 100) for _ in range(int(seconds * 100))]
        s.close()
    raw = np.frombuffer(b"".join(frames), dtype=np.float32).reshape(-1, ch)
    sf.write(ROOT / "out" / "cable_recorded.wav", raw, rate)


def main() -> None:
    (ROOT / "out").mkdir(exist_ok=True)
    rec = subprocess.Popen([sys.executable, __file__, "--record", "10"])
    time.sleep(1.0)  # que el grabador abra el stream antes de reproducir

    audio, sr = sf.read(ROOT / "samples" / "en_smoke.wav", dtype="float32")
    with pyaudio.PyAudio() as pa:
        d = find(pa, "CABLE Input", want_input=False)
        rate, ch = int(d["defaultSampleRate"]), d["maxOutputChannels"]
        out = soxr.resample(audio, sr, rate).astype(np.float32)
        out = np.repeat(out[:, None], ch, axis=1)
        print(f"reproduciendo {len(audio) / sr:.2f} s -> [{d['index']}] {d['name']} {rate} Hz x{ch}")
        s = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, output=True,
                    output_device_index=d["index"])
        s.write(out.tobytes())
        s.close()
    rec.wait()

    raw, rate = sf.read(ROOT / "out" / "cable_recorded.wav", dtype="float32")
    mono16 = soxr.resample(raw.mean(axis=1), rate, 16000).astype(np.float32)
    print(f"grabado desde CABLE Output: {len(raw) / rate:.2f} s | peak {np.abs(raw).max():.3f}")

    from voztrad import cuda_dlls  # noqa: F401
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cuda", compute_type="float16")
    segs, _ = model.transcribe(mono16, language="en", beam_size=1, condition_on_previous_text=False)
    print("whisper:", " ".join(x.text.strip() for x in segs) or "<vacío>")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--record":
        record(float(sys.argv[2]))
    else:
        main()

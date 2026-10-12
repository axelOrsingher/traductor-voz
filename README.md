# voice-bridge

**Local, real-time, two-way voice translation for calls (Teams, Meet, …).** It listens to the
other side and shows/speaks a Spanish translation, and translates what *you* say into English
and feeds it back into the call as a virtual microphone — all **on-device**, no cloud.

No PyTorch at runtime: faster-whisper and opus-mt run on **CTranslate2**; Silero VAD, Parakeet
and Piper run on **ONNX Runtime**.

```
them  →  call audio  →  capture → [VAD A] → [ASR en] → [MT en→es] → overlay subtitles
                                                                   ↘ (optional) spoken ES → your headphones
you   →  mic → [PTT F8 | VAD] → [ASR es] → [MT es→en] → [TTS] → virtual mic → the call
```

Runs on **Windows 10/11** (WASAPI + VB-CABLE) and **Linux** (PipeWire/Wayland, validated on
Manjaro + KDE Plasma). It **auto-configures the models and precision to your hardware** — see
[Hardware auto-config](#hardware-auto-config).

---

## Highlights

- **Two directions at once.** A = *them → you* (live subtitles + optional spoken Spanish).
  B = *you → them* (your speech, translated and spoken into the call).
- **Low latency** — first subtitle in **~0.4–0.6 s** after the speaker stops (see
  [Latency](#latency)). MT runs on the GPU; subtitles stream live while they talk.
- **Hardware-aware.** Picks ASR backend, Parakeet precision and MT compute type from your GPU's
  VRAM and CUDA architecture; `python -m voztrad detect` shows the chosen profile.
- **Two-tier translation** (Windows / ≥6 GB GPU). opus-mt (~20 ms) shows instantly, then
  TranslateGemma-4B (llama.cpp) replaces the subtitle with a more idiomatic version.
- **Voice cloning (optional).** Kyutai Pocket TTS (ONNX, CPU, zero-shot) speaks your side in
  your own voice; otherwise Piper.
- **Always-on-top subtitle overlay** (PySide6): draggable, resizable, font sizing, right-click
  options. **Push-to-talk (F8) or hands-free / interpreter**, switchable from the overlay.

---

## Quick start

### Linux (PipeWire / Wayland)

```bash
git clone https://github.com/axelOrsingher/voice-bridge && cd voice-bridge
scripts/linux/install-linux.sh      # venv + models + 'input' group + desktop icon
scripts/linux/voice-bridge          # or double-click the "voice-bridge" icon
```
In your call app set **Speaker → `voice-bridge (call audio)`** and
**Microphone → `voice-bridge microphone`**. Hold **F8** to talk. Use headphones.

### Windows 10/11

```powershell
git clone https://github.com/axelOrsingher/voice-bridge; cd voice-bridge
# install VB-CABLE (admin), then restore your speakers as default output:
scripts\set_default_audio.ps1 -Match Speakers
uv sync --extra ui
.venv\Scripts\python.exe scripts\fetch_models.py     # Silero + Piper voices
.venv\Scripts\python.exe -m voztrad app              # engine + overlay
```
In your call app set **Microphone → `CABLE Output`**. Hold **F8** to talk.

**Requirements (both):** Python 3.11, [uv](https://docs.astral.sh/uv/), a recent NVIDIA driver
with CUDA 12.x (runs on CPU too, slower — see auto-config). On Linux the CUDA 12 libraries come
with the venv (`uv sync`); on Windows see [docs/windows-setup.md](docs/windows-setup.md).

---

## Hardware auto-config

At launch, voice-bridge detects your GPU/CPU/audio stack and selects models and precision that
fit. **Any explicit flag overrides the auto choice; `--no-auto` disables it.**

```console
$ python -m voztrad detect
Hardware
  GPU   : NVIDIA GeForce GTX 1650 Ti — 4096 MiB — compute cap 7.5
  CPU   : 16 cores — RAM 33.3 GB
  Audio : PulseAudio (on PipeWire 1.4.10) — voice recording: no
Profile (auto)
  asr: parakeet · parakeet_quant: fp32 · mt_compute_type: int8 · refine: False · tts: piper
```

| Detected | Chosen |
|---|---|
| **No NVIDIA GPU** | everything on CPU (Parakeet int8 + MT int8), no refiner — expect higher latency |
| **Turing (cc 7.5)** | MT `int8`, Parakeet fp32 (int8 MT is faster than float16 here) |
| **Ampere+ (cc ≥ 8.0)** | MT `float16` (tensor cores) |
| **VRAM < 6 GB** | Parakeet fp32 alone, **no** TranslateGemma refiner (won't fit) |
| **VRAM 6–8 GB** | Parakeet fp16 **+** refiner (Windows) |
| **VRAM ≥ 8 GB** | Parakeet fp32 **+** refiner |
| **A voice recording exists** | Pocket TTS (cloned voice); otherwise Piper |

These tiers encode what was measured on the reference hardware — see
[`docs/porting/linux-recon.md`](docs/porting/linux-recon.md) §6.

---

## Usage

Everyday: launch **voice-bridge** (Linux icon / `voztrad app`). It starts the engine + the
subtitle overlay; closing the overlay (✕) shuts everything down. Engine console →
`logs/engine-console.txt`, phrases & latencies → `logs/run-*.jsonl`.

**Overlay:** drag from the top bar, resize from the corner, **A− / A+** font size, **right-click**
for options (original text, latency, opacity, clear). Narrow → it stacks vertically.

```bash
python -m voztrad detect       # show detected hardware + chosen profile
python -m voztrad app          # engine + overlay (auto-config; closing the overlay stops all)
python -m voztrad run          # headless engine only
python -m voztrad ui           # overlay only (connect to a running engine)
python -m voztrad voice        # record your voice for cloning
python -m voztrad bench <dir>  # compare whisper vs parakeet on the same wavs
python -m voztrad report logs/run-*.jsonl   # p50/p95 per stage
```

### Useful `run` / `app` flags (all optional — auto picks sensible defaults)

| Flag | Effect |
|---|---|
| `--no-auto` | don't adapt to hardware; use manual defaults |
| `--asr {parakeet,whisper}` | ASR backend |
| `--parakeet-quant {auto,int8,fp16,fp32}` | Parakeet precision |
| `--mt-compute-type <type>` | CTranslate2 compute type for MT (e.g. `int8`, `float16`) |
| `--b-mode {ptt,vad,interp}` | push-to-talk / hands-free / interpreter |
| `--ptt-key F8` · `--hangover 500` | PTT key · ms of silence to close a phrase |
| `--mic "<substring>"` | pick the input device by name |
| `--tts {pocket,piper}` | cloned voice vs generic Piper |
| `--no-refine` · `--no-listen` · `--no-smart-turn` | turn off the refiner / Spanish listen-back / Smart Turn |
| `--no-norm` · `--no-a` · `--no-b` · `--no-gpu-warm` | skip Rioplatense norm · disable A / B · don't keep the GPU warm |

### Hear the others in Spanish

With **🔊 LISTEN ES** (on by default), what they say is spoken back in Spanish through your
headphones (~0.8 s after each phrase), with the call app ducked to 35% while it plays. Your own
translated output is **never re-captured** (Windows: WASAPI process-loopback EXCLUDE; Linux: the
call audio is routed through a dedicated sink, so our output on the real device is excluded). In
hands-free/interpreter mode the mic is muted while the translation plays (for speaker users).

### Clone your voice

`python -m voztrad voice` guides you (~20 s Spanish + ~20 s English) and saves a reference to
`models/voice/me.wav` (not versioned). With a recording, Pocket TTS speaks your side in your
voice; without one it falls back to Piper. Use only with your own voice or with consent (per the
model license).

### UI transport

The engine speaks JSON-lines on `127.0.0.1:8765`. Events:
`{"type": "phrase"|"status"|"drop", ...}`. Commands: `{"cmd": "ptt", "down": true}`,
`{"cmd": "b_mode", "value": "vad"}`, `{"cmd": "stop"}`.

---

## Latency

End-to-end = from the moment the speaker stops to the output. Measured; your numbers depend on
GPU, hangover and whether Smart Turn / the refiner are on.

| Path | Linux — GTX 1650 Ti (auto: Parakeet fp32 + MT int8) | Windows — dev box (Smart Turn + two-tier) |
|---|---|---|
| **Subtitle** (them → you) | ~0.60 s | **~0.39 s** (p50) |
| **Spanish in your headphones** | 0.72–0.89 s | ~0.53 s |
| **Your voice** (you → them, es→en + TTS) | ~0.78 s | — |

Per-stage (component benchmark, GTX 1650 Ti):

| Stage | Time | VRAM |
|---|---|---|
| opus-mt en↔es (CTranslate2 `int8`) | **~20 ms** | ~170 MiB |
| Parakeet TDT 0.6B v3 (onnx-asr, fp32, CUDA) | **~85 ms** | ~3.48 GB |
| Silero VAD (onnx) | ~25 ms | CPU |

The Linux end-to-end is higher mainly because the default phrase-close hangover is 500 ms;
enabling Smart Turn (closes at ~200 ms) brings it close to the Windows numbers. Full
measurements and the precision trade-offs are in
[`docs/porting/linux-recon.md`](docs/porting/linux-recon.md) §6.

---

## Platform notes

- **Linux:** PipeWire + a Wayland (KDE/KWin) or X11 session. The virtual audio devices are
  created by [`scripts/linux/voztrad-audio.sh`](scripts/linux/voztrad-audio.sh) (reversible),
  the global F8 hotkey uses `python-evdev` (your user in the `input` group), and the overlay is
  launched with `QT_QPA_PLATFORM=xcb` so it stays on top under KWin. The full Windows→Linux
  component mapping is in
  [`docs/porting/windows-to-linux-inventory.md`](docs/porting/windows-to-linux-inventory.md).
- **Windows:** WASAPI + [VB-CABLE](https://vb-audio.com/Cable/). CUDA/cuDNN redist layout and the
  Windows-specific latency fixes (EcoQoS throttling, GPU P-state, spin-wait) are documented in
  [docs/windows-setup.md](docs/windows-setup.md).

### Common issues

- **No audio captured / nothing to translate:** make sure the call app's **output** is the
  voice-bridge call sink (Linux: `voice-bridge (call audio)`; Windows: VB-CABLE) and its
  **microphone** is the voice-bridge mic. On Linux, `scripts/linux/voztrad-audio.sh capture-app "<app>"`
  moves a running app's audio into the call sink.
- **F8 does nothing (Linux):** you must be in the `input` group (`sudo usermod -aG input $USER`,
  then log out/in). pynput is unreliable under Wayland — voice-bridge uses evdev.
- **Overlay not on top (Linux):** launch with `QT_QPA_PLATFORM=xcb` (the launcher already does).
- **Feedback / it translates itself:** use **headphones**. In hands-free mode the mic is gated
  while the Spanish plays, but headphones are recommended.
- **First phrase is slow:** models load (and the GPU ramps its clock) on the first use; it warms
  up after a few seconds.

---

## License & consent

Voice cloning is permitted **only with your own voice or with explicit consent**, per the TTS
model license.

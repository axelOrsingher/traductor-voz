# voice-bridge

**Local, real-time, two-way voice translation for calls (Teams, Meet, …).** It listens to the
other side and shows/speaks a Spanish translation, and translates what *you* say into English
and feeds it back into the call as a virtual microphone — all **on-device**, no cloud.

No PyTorch at runtime: faster-whisper and opus-mt run on **CTranslate2**; Silero VAD, Parakeet
and Piper run on **ONNX Runtime**.

```
them  →  call audio  →  loopback capture → [VAD A] → [ASR en] → [MT en→es] → overlay subtitles
                                                                            ↘ (optional) spoken ES → your headphones
you   →  mic → [PTT F8 | VAD] → [ASR es] → [MT es→en] → [TTS] → virtual mic → the call
```

---

## Highlights

- **Two directions at once.** A = *them → you* (live subtitles, optional spoken Spanish).
  B = *you → them* (your speech, translated and spoken into the call).
- **Low latency** (measured on the dev machine): first subtitle **~390 ms** after the speaker
  finishes, audible Spanish **~530 ms**. Smart Turn v3.2 closes a phrase after ~200 ms of
  silence; MT runs on the GPU; subtitles stream live while they talk.
- **Two-tier translation.** opus-mt (~30 ms) shows instantly, then TranslateGemma-4B
  (llama.cpp, ~340 ms) replaces the subtitle with a much more idiomatic version — without
  adding latency when there is already audio playing. `--no-refine` turns it off.
- **Voice cloning (optional).** Kyutai Pocket TTS (ONNX, CPU, zero-shot) can speak *your* B
  translations in your own voice; otherwise it falls back to Piper. Record once with
  `voztrad voice`.
- **Always-on-top subtitle overlay** (PySide6): draggable, resizable, font sizing, right-click
  options (show original, latency, opacity).
- **Push-to-talk or hands-free.** Hold **F8** to talk, or switch to VAD / interpreter mode.

## Platform support

| Platform | Status |
|---|---|
| **Windows 10/11** | ✅ working (current reference platform) |
| **Linux** (PipeWire / Wayland) | 🚧 in progress — see [`docs/porting/`](docs/porting/) |

The Linux reconnaissance (audio routing, global hotkey, GPU/model VRAM & latency, overlay)
and a full Windows→Linux code inventory live in
[`docs/porting/linux-recon.md`](docs/porting/linux-recon.md) and
[`docs/porting/windows-to-linux-inventory.md`](docs/porting/windows-to-linux-inventory.md).

---

## Requirements

- **Python 3.11** and [uv](https://docs.astral.sh/uv/).
- A recent **NVIDIA driver** + **CUDA 12.x** runtime. cuDNN 9 (CUDA 12) and cudart ≥ 12.8 — the
  exact redist layout is described in [Troubleshooting](#troubleshooting-windows) and resolved
  in [`src/voztrad/cuda_dlls.py`](src/voztrad/cuda_dlls.py).
- A GPU with enough VRAM for the stack. Parakeet fp16 + opus-mt + VAD fit comfortably; the
  TranslateGemma-4B refiner needs several GB more (see the VRAM notes in the porting docs — on
  a 4 GB card the refiner does not fit alongside the ASR).
- **Windows only, for now:** [VB-CABLE](https://vb-audio.com/Cable/) virtual audio device.

## Install (Windows)

1. Install Python 3.11, uv, the NVIDIA driver and CUDA 12.x (plus the cuDNN/cudart redist — see
   [Troubleshooting](#troubleshooting-windows)).
2. Install **VB-CABLE** (requires admin), then restore your real speakers as default output:
   ```powershell
   scripts\set_default_audio.ps1 -Match Speakers
   ```
3. Sync and fetch models:
   ```powershell
   uv sync --extra ui
   .venv\Scripts\python.exe scripts\fetch_models.py   # Silero VAD + Piper voices (ASR/MT self-download)
   .venv\Scripts\python.exe -m pytest tests -q
   ```

---

## Usage

### Everyday

Use the desktop shortcut **"voice-bridge"** (= `pythonw -m voztrad app`): it starts the engine
(no console) plus the subtitle overlay. Closing the overlay (✕) shuts everything down.

In your call app, set the **microphone to `CABLE Output`**. Hold **F8** to speak (or switch to
hands-free from the overlay). Engine console output goes to `logs\engine-console.txt`; phrases
and latencies to `logs\run-*.jsonl`.

**Overlay:** drag from the top bar, resize from the corner, **A− / A+** change font size,
**right-click** for options (original text, latency, opacity, clear). Make it narrow and it stacks.

### Commands

```powershell
python -m voztrad app                      # engine + overlay (closing the overlay stops all)
python -m voztrad ui                        # overlay only (connects to an already-running engine)
python -m voztrad run                       # headless engine; defaults: Parakeet ASR + PTT on F8
python -m voztrad run --asr whisper --b-mode vad   # hands-free
python -m voztrad run --save-segments       # save every phrase under out/segments/
python -m voztrad voice                     # record your voice and pick a reference for cloning
python -m voztrad bench out\segments\<run>  # compare whisper vs parakeet on the SAME audio
python -m voztrad report logs\run-*.jsonl   # p50/p95 per stage
python scripts\e2e.py --asr parakeet        # automated end-to-end test (A and B)
```

### Useful `run` flags

| Flag | Effect |
|---|---|
| `--asr {parakeet,whisper}` | ASR backend (default `parakeet`) |
| `--parakeet-quant int8` | use the int8 Parakeet export |
| `--b-mode {ptt,vad,interp}` | push-to-talk / hands-free / interpreter (default `ptt`) |
| `--ptt-key F8` | push-to-talk key |
| `--hangover 500` | ms of silence to close a phrase |
| `--mic "<substring>"` | pick the input device by name |
| `--tts {pocket,piper}` | cloned voice vs generic Piper (~190 ms faster) |
| `--no-refine` | disable the TranslateGemma-4B refiner |
| `--no-listen` | don't speak the Spanish translation to your headphones |
| `--listen-voice <model.onnx>` | Piper voice for the spoken Spanish |
| `--no-norm` | skip Rioplatense normalization before MT |
| `--no-a` / `--no-b` | disable direction A (them→you) / B (you→them) |
| `--no-gpu-warm` | don't keep the GPU warm during speech |

### Hear the others in Spanish

With **🔊 LISTEN ES** (on by default), what they say in English is spoken back in Spanish
through your headphones (~0.8 s after each phrase), with the call app ducked to 35% while it
plays. Direction A captures all system audio **except voice-bridge's own output** (WASAPI
process-loopback EXCLUDE), so the translation is never re-captured. In hands-free/interpreter
mode the mic is ignored while the translation is playing (in case you use speakers).
`--listen-voice models\es_AR-daniela-high.onnx` selects a Rioplatense voice (slower).

### Clone your voice

Kyutai Pocket TTS (ONNX, CPU, zero-shot) can speak your B translations in your own voice.
Record once with `python -m voztrad voice` (it guides you: ~20 s Spanish + ~20 s English,
checks levels, generates and plays samples). The reference is saved to `models/voice/me.wav`
(not versioned). Without a recording it falls back to Piper. Use only with your own voice or
with consent (per the model license).

### UI transport

The engine speaks JSON-lines on `127.0.0.1:8765`.
Events: `{"type": "phrase"|"status"|"drop", ...}`.
Commands: `{"cmd": "ptt", "down": true}`, `{"cmd": "b_mode", "value": "vad"}`, `{"cmd": "stop"}`.

---

## Troubleshooting (Windows)

What breaks on a fresh machine and how it is resolved:

| Problem | Fix |
|---|---|
| PyPI `onnxruntime-gpu` 1.30 is a **CUDA 13** build → silently falls back to CPU | pin 1.29.0 from Microsoft's CUDA 12 feed (`[tool.uv.index] ort-cuda12`) |
| `piper-tts` drags in CPU `onnxruntime`, clobbering the GPU build | `override-dependencies` in `pyproject.toml` |
| ORT CUDA 12 needs cudart ≥ 12.8 symbols; installed toolkit is 12.6 | cudart 12.9 redist in `~/tools/cuda_cudart-12.9.79`, preloaded in `cuda_dlls.py` |
| cuDNN 9 missing | redist 9.27 in `~/tools/cudnn-9.27.0.42_cuda12/bin/x64` (used by ORT; CT2 ships its own dispatcher) |
| Windows 11 power-throttles windowless processes (EcoQoS): MT 45 → 190 ms after idle | `winperf.disable_throttling()` on startup |
| ORT spin-wait steals CPU from MT (900 ms spikes) | `allow_spinning=0` on ORT sessions |
| GPU drops to P8/210 MHz after ~2 s idle: ASR 2–6× slower | `gpuwarm.py` loads the GPU only while there is speech (~0.6 s is enough). Admin alternative: `nvidia-smi -lgc 1500,3105` |
| Installing VB-CABLE makes it the default output (you hear nothing) | `scripts\set_default_audio.ps1 -Match Speakers` |

DLL paths are overridable with `VOZTRAD_CUDART_BIN`, `VOZTRAD_CUDNN_BIN`, `VOZTRAD_CUDA_BIN`.

---

## License & consent

Voice cloning is permitted **only with your own voice or with explicit consent**, per the TTS
model license.

# CLAUDE.md

voice-bridge (Python package `voztrad`): local, real-time, bidirectional en<->es voice
translator for calls (Teams, Meet, Zoom). Everything runs on the user's machine; no cloud
APIs. Several agents work on it from different machines (Windows, Linux, macOS). This file
is the shared context and the rules.

## How it works

```
A (them -> you):  call audio --capture--> VAD + Smart Turn -> ASR (en) -> MT en->es
                     -> subtitles in the overlay
                     -> (optional) Spanish TTS in your headphones, call app ducked
B (you -> them):  mic -> PTT (F8) | VAD | interpreter -> ASR (es) -> voseo norm -> MT es->en
                     -> TTS (your cloned voice or Piper) -> virtual mic -> call app
```

- Two processes: the **engine** (`python -m voztrad run`) and the **overlay**
  (`voztrad ui`, PySide6). They talk only through JSON-lines over TCP `127.0.0.1:8765`
  (`transport.py`). `voztrad app` starts both; closing the overlay stops the engine.
- Engine threads (see `engine.py` docstring): `cap-A/B` audio callbacks (enqueue only),
  `proc-A/B` framing + VAD/PTT -> segments, **one** `asr` thread that owns the GPU model
  (B has priority over A), `mt`, `tts` (B -> virtual mic), `refine` (TranslateGemma),
  `listen` (Spanish into headphones).
- Internal audio format: 16 kHz mono float32, 512-sample frames (Silero VAD requirement).
- Two-tier translation: opus-mt (CTranslate2, ~20-30 ms) is shown immediately; the
  TranslateGemma-4B refiner (llama.cpp `llama-server` child process, ~340 ms) replaces the
  subtitle, and is used for your voice only if it fits within audio already playing.
- Our own output must never be re-captured by direction A (Windows: WASAPI process
  loopback EXCLUDE; Linux: call app routed through a dedicated sink, our TTS goes to the
  real device).

## Module map (`src/voztrad/`)

| Module | Role |
|---|---|
| `__main__.py` | CLI: `run`, `app`, `ui`, `voice`, `detect`, `bench`, `report` |
| `config.py` | `Config` / `SegmenterConfig` dataclasses; every latency knob lives here |
| `hwdetect.py` | detects GPU/CPU/audio and picks a profile (models, precision, features) |
| `engine.py` | pipeline, threads, queues, metrics -> `logs/run-*.jsonl` |
| `audio.py` + `audio_{win,linux}.py` | neutral framing + per-OS capture/playback backends |
| `procloop.py` | Windows process-loopback capture (COM via comtypes) |
| `ptt.py` + `ptt_{win,linux}.py` | global push-to-talk key |
| `ducker.py` + `ducker_{win,linux}.py` | lower the call app's volume while Spanish plays |
| `vad.py` | Silero VAD v6 + segmenter (hysteresis, pre-roll, soft cuts for interpreter mode) |
| `smartturn.py` | Smart Turn v3.2 end-of-turn model (ONNX CPU) |
| `asr.py` | Parakeet TDT 0.6B v3 (onnx-asr) and faster-whisper backends |
| `mt.py` | opus-mt via CTranslate2 |
| `refine.py` | TranslateGemma-4B via llama-server (currently Windows-only wiring) |
| `tts.py`, `tts_pocket.py` | Piper; Kyutai Pocket TTS (pure ONNX voice cloning) |
| `tempo.py` | streaming WSOLA speed-up when the TTS queue lags |
| `langid.py`, `voseo.py` | text language check; Rioplatense -> neutral Spanish normalization |
| `gpuwarm.py` | keeps the NVIDIA GPU clocked up while someone speaks |
| `winperf.py`, `cuda_dlls.py` | Windows-only: disable EcoQoS throttling; CUDA DLL paths |
| `voice.py` | guided voice recording for cloning (`models/voice/me.wav`) |
| `ui/overlay.py` | subtitle overlay (PySide6) |

## Commands

```bash
uv sync --extra ui                       # venv (Python 3.11 exactly)
python scripts/fetch_models.py           # Silero, Piper voices, Smart Turn, TranslateGemma
python -m voztrad detect                 # hardware + auto profile
python -m voztrad app                    # engine + overlay
python -m voztrad run [flags]            # headless engine (see --help)
python -m voztrad report logs/run-*.jsonl  # p50/p95 per stage
python -m pytest tests -q
python scripts/e2e.py --asr parakeet     # E2E latency (Windows/WASAPI only today)
```
Linux: `scripts/linux/install-linux.sh`, virtual devices via `scripts/linux/voztrad-audio.sh`.
Windows setup: `docs/windows-setup.md`. On Windows use `.venv\Scripts\python.exe`.

## Data and privacy
- `models/`, `logs/`, `out/` are not versioned. `logs/` and `out/` can contain audio and
  transcripts of **real calls**: never commit, upload or paste them anywhere.
- `models/voice/me.wav` is the user's voice. Voice cloning only with the speaker's own voice
  or explicit consent (model license).

## Reference numbers (p50, end of speech -> output)
- Windows, RTX 4050 6 GB, Smart Turn + refiner: subtitle ~0.39 s, Spanish audible ~0.53 s.
  Total VRAM with everything loaded: 4.46 GB.
- Linux, GTX 1650 Ti 4 GB (no Smart Turn, no refiner): subtitle ~0.60 s, Spanish audible
  0.72-0.89 s, your voice ~0.78 s.
Platform pitfalls already solved (and why) are in `docs/windows-setup.md`,
`docs/porting/linux-recon.md`, `docs/porting/windows-to-linux-inventory.md`. Read them
before "fixing" something that looks odd (pinned onnxruntime-gpu build, `allow_spinning=0`,
GPU warm-up, `QT_QPA_PLATFORM=xcb`, etc.).

## Rules

**Language**
- Everything in the repo is **English**: identifiers, comments, docstrings, log/status
  messages, CLI help, docs, commits, PRs. The owner chats in Spanish; answer them in
  Spanish. Exception: Spanish *content* handled on purpose (voseo rules, prompts, test
  phrases). Older modules still have Spanish comments: translate what you touch.
- UI text and the voice-recording guide will be localizable (en/es, chosen independently).
  Don't add user-facing strings in only one language.

**Engineering**
- **No PyTorch.** Inference via CTranslate2, ONNX Runtime, llama.cpp; on Apple Silicon,
  MLX / Core ML / Metal are fine.
- **Measure, don't assume.** Latency/VRAM/quality claims in code, docs or commits come
  from a run on real hardware; name the machine, report p50/p95.
- Latency is the product. Don't make another platform slower; if you change shared code,
  say so in the PR and how you checked the other platforms.
- OS-specific code goes in `*_win.py` / `*_linux.py` / `*_mac.py` behind the neutral
  module, with an explicit branch per OS (no "everything non-Windows is Linux"). The engine
  doesn't import OS-specific modules. Missing capabilities degrade with a clear status
  message, never a crash.
- Hardware-dependent defaults go through `hwdetect.choose_profile()`. Explicit CLI flags
  always win; `--no-auto` falls back to `Config` defaults.
- Audio callbacks only enqueue: a slow callback glitches the real call audio.
- Child processes (llama-server) must die with the engine, even on a crash.

**Workflow**
- Branch + PR to `main`, small focused commits.
- Before a PR: tests pass, and the PR includes an end-to-end latency run on your machine
  (before/after if you touched the hot path).
- Porting notes and measurements go in `docs/porting/<platform>-*.md`.

## Known gaps (2026-10-13)
- `scripts/e2e.py` and `e2e_listen.py` still use PyAudioWPatch (Windows only); there is no
  backend-agnostic end-to-end harness yet.
- The refiner's child-process guard on Linux (`PR_SET_PDEATHSIG`) and macOS (watchdog) is
  written but not yet verified on those systems; `fetch_models.py` only installs llama.cpp
  on Windows (elsewhere: `llama-server` on PATH or `VOZTRAD_LLAMA_SERVER`).
- macOS: not supported yet (port in progress on branch `macos-port`).
- No CI yet.

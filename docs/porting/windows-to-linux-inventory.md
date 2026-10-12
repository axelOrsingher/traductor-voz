# Windows → Linux porting inventory

Code-level companion to [`linux-recon.md`](linux-recon.md). Every Windows-specific
dependency / assumption in the repo, why it breaks on Linux (PipeWire/Wayland/KDE), and the
concrete replacement. Line numbers are from the audited tree; treat them as "here-ish" once
the port starts moving code.

**Cross-cutting risk:** `pyaudiowpatch`, `comtypes` and `winsound` are imported at
**module top level** in `audio.py`, `engine.py`, `voice.py`, `ducker.py`, `procloop.py`.
`tests/test_imports.py` walks and imports every `voztrad.*` module, so **the test suite
fails to even collect on Linux** until these imports are lazy/guarded or the modules are
split per-platform. Fix these first.

## 1. Audio subsystem (the big one)

- **Capture/playback core** — `audio.py:15,21-52`, `engine.py:24,129,138-148`, `voice.py:19`.
  `import pyaudiowpatch as pyaudio` + WASAPI host API (`paWASAPI`), `get_default_wasapi_loopback()`,
  `isLoopbackDevice`. PyAudioWPatch is `sys_platform=='win32'` only → `ModuleNotFoundError` on Linux.
  → Re-implement over PipeWire (`pw-record`/`pw-play`/`sounddevice`). Capture the call sink's
  **monitor** with `pw-record --target voztrad_calls -P '{ stream.capture.sink=true }'`
  (recon §2.a). **Effort: high** — this is the core I/O abstraction.
- **VB-CABLE virtual device** — `config.py:51 cable="CABLE Input"`, `__main__.py:245`,
  `audio.py:112`, `scripts/e2e.py`, `scripts/smoke_cable.py`, `README.md`. Hard-coded Windows
  driver device names. → PipeWire `null-sink` + `remap-source` (`voztrad_mic`/`voztrad_mic_input`,
  recon §2.b). **Effort: med.**
- **Process-loopback (capture everything except our own process)** — `procloop.py` (whole;
  `comtypes`, `ActivateAudioInterfaceAsync`, `"VAD\\Process_Loopback"`, `Mmdevapi.dll`),
  consumer `engine.py:158-170` (already `try/except`→ normal loopback). → Not needed: recon §2.a
  proved audio sent to the real sink is absent from the `voztrad_calls` monitor, so routing
  alone excludes our TTS. Wire the routing in so `listen_ok` stays enabled. **Effort: med.**
- **Teams ducking** — `ducker.py` (whole; `comtypes` + `pycaw` WASAPI session volume, `.exe`
  names), `config.py:63 duck_apps=("ms-teams.exe","teams.exe")`, `engine.py:135-140`.
  → `pactl set-sink-input-volume <idx> 40%` matching `application.name` (recon §2.c, verified).
  Change `duck_apps` to Linux names (`"Teams"`). **Effort: med.**
- **Default-device switch** — `scripts/set_default_audio.ps1` (PowerShell `IPolicyConfig` COM),
  `README.md:14,82`. → `wpctl set-default` / `pactl set-default-sink` (owned by
  `scripts/linux/voztrad-audio.sh`). **Effort: low.**
- **Sample playback** — `voice.py:15,121 winsound.PlaySound`, many `scripts/*`.
  → `paplay`/`pw-play`/`sounddevice`. **Effort: low.**

## 2. Global PTT hotkey (F8)

- `ptt.py:5,9-11,25,40` — `user32.GetAsyncKeyState` polled every 5 ms + Windows VK codes;
  consumer `engine.py:149-151`, default `cfg.ptt_key="F8"`.
  → **python-evdev** on `/dev/input/event*` (user in `input` group), filter `ecodes.KEY_F8`,
  `value==1/0` for down/up. Recon §3 proved evdev is reliable (8/8) and pynput is not. Keep the
  existing `set_pressed()/on_change` interface; add an evdev backend. **Effort: med.**

## 3. Process / window management

- **Kill llama-server when engine dies** — `refine.py:44-67,123-125` Job Object
  `KILL_ON_JOB_CLOSE` (no-ops on Linux → **orphaned llama-server holds VRAM**).
  → `prctl(PR_SET_PDEATHSIG, SIGKILL)` via `preexec_fn`, or new process group + group kill.
  **Effort: low** but important (VRAM leak on a 4 GB card).
- **Overlay always-on-top** — `overlay.py:227-228` `Qt.WindowStaysOnTopHint`. Ignored by KWin
  under native Wayland. → launch with `QT_QPA_PLATFORM=xcb` (recon §5). **Effort: low.**
- **Single-instance mutex** — `__main__.py:69-78 CreateMutexW` (returns True → no guard on
  Linux). → `flock`/AF_UNIX in `$XDG_RUNTIME_DIR`, or reuse the existing TCP `_port_open` check.
  **Effort: low.**
- **Power throttling / priority** — `winperf.py` (whole; `SetProcessInformation`,
  `SetPriorityClass`, `timeBeginPeriod`), called `__main__.py:231`. No-op on Linux. The Linux
  analogue is **GPU down-clocking** (recon §4). → rely on `gpuwarm.py` / `nvidia-smi -lgc`.
  **Effort: low.**
- Already-portable / no-op: `CREATE_NO_WINDOW` (`refine.py:81`, `__main__.py:94`, guarded),
  `pythonw.exe→python.exe` (`__main__.py:95`), TCP stop command (`__main__.py:99-108`),
  overlay font fallback to "Inter" (`overlay.py:32`).

## 4. Paths / filesystem

- **CUDA DLL loader** — `cuda_dlls.py` (whole; `os.add_dll_directory`, `WinDLL(cudart)`, `C:\`
  paths), imported by `asr.py:14`, `mt.py:13` (no-ops on Linux). → put the pip
  `nvidia-*-cu12` libs on `LD_LIBRARY_PATH` **before** Python starts (recon §4 gotcha #2);
  keep `setup()` a no-op. **Effort: med** (launcher/env wiring).
- **llama.cpp binary** — `refine.py:27,72-81` `llama-server.exe` under
  `~/tools/llama.cpp-b11433-cuda12.4`; `fetch_models.py:43-48` downloads the win-cuda zip.
  → Linux `llama-server` (ubuntu-x64 release / build), drop `.exe`. **Effort: med.**
- **fp16 convert script** — `convert_parakeet_fp16.py:7-8` uses `conv/Scripts/python.exe`
  (Windows venv layout). → `conv/bin/python`. **Effort: low.** (Recon §4 strongly recommends
  shipping fp16 Parakeet on the 4 GB card — see below.)
- Backslash strings (`__main__.py:77`, `procloop.py:134`, `voice.py:126`, `README.md`): cosmetic.
- `QSettings("voztrad","overlay")` transparently uses `~/.config/voztrad/overlay.conf` on Linux — no change.

## 5. Packaging / dependencies (`pyproject.toml`, `uv.lock`)

- `PyAudioWPatch` is correctly `win32`-marked — add a Linux audio dep (`sounddevice`/`pulsectl`)
  under `sys_platform=='linux'`.
- **`comtypes` and `pycaw` are NOT platform-marked** → they try to install on Linux and are
  useless/failing. Add `; sys_platform=='win32'`.
- `onnxruntime-gpu` pin + custom CUDA-12 index: recon §4 found the clean CUDA-12 build on this
  box is **`onnxruntime-gpu==1.22.0`** (1.31 is CUDA-13 and also rejects Parakeet external-data;
  `float16` MT is slow on Turing). Verify the index serves a Linux CUDA-12 wheel; plan to pin 1.22.0.
- `override-dependencies=["onnxruntime ; sys_platform=='never'"]` keeps CPU onnxruntime from
  clobbering the GPU build — fine on Linux.
- Re-run `uv lock` on Linux after fixing markers (lockfile has win32/win_amd64 wheels).

## Summary (do-first → nice-to-have)

| Priority | Component | Windows mechanism | Linux replacement | Effort |
|---|---|---|---|---|
| 🔴 1 | top-level Win imports break tests | `pyaudiowpatch`/`comtypes`/`winsound` at import | lazy/guarded imports or platform split | med |
| 🔴 2 | audio capture/playback core | PyAudioWPatch + WASAPI | PipeWire (`pw-record`/`pw-play`/`sounddevice`) | high |
| 🔴 3 | PTT F8 | `GetAsyncKeyState` + VK codes | python-evdev (`input` group) | med |
| 🔴 4 | llama.cpp + CUDA libs | `.exe` + `add_dll_directory` | linux binary + `LD_LIBRARY_PATH` | med |
| 🟠 5 | VB-CABLE | "CABLE Input/Output" | null-sink + remap-source | med |
| 🟠 6 | Teams ducking | pycaw/comtypes | `pactl set-sink-input-volume` | med |
| 🟠 7 | procloop self-exclusion | WASAPI Process_Loopback COM | PipeWire sink-monitor routing | med |
| 🟠 8 | kill child on exit | Job Object | `prctl(PR_SET_PDEATHSIG)` | low |
| 🟢 9 | overlay on-top | `WindowStaysOnTopHint` | `QT_QPA_PLATFORM=xcb` | low |
| 🟢 10 | default device / mutex / winsound / winperf / paths / deps markers | various Win32 | wpctl/pactl, flock, paplay, no-op, pathlib, markers | low |

Already guarded-no-op or cosmetic: `CREATE_NO_WINDOW`, `pythonw`, `winperf`, `cuda_dlls.setup()`,
overlay font, backslash strings. Every 🔴/🟠 replacement is validated on the target machine in
[`linux-recon.md`](linux-recon.md) + [`scripts/linux/`](../../scripts/linux/).

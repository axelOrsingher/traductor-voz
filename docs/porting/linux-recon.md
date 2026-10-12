# Linux reconnaissance — porting from Windows

Target machine: **Dell XPS 15**, NVIDIA **GTX 1650 Ti (4 GB)**, **Wayland** session.
Date: 2026-10-12. All commands below were run on the machine; outputs are verbatim.

Scope of this doc: establish what the Windows-only app needs on Linux — system/audio
stack, a global push-to-talk key under Wayland, GPU model VRAM/latency on a 4 GB card,
and whether the PySide6 overlay can stay on top. Helper scripts live in
[`scripts/linux/`](../../scripts/linux/).

TL;DR for the port:

| Area | Finding | Decision |
|---|---|---|
| Audio capture | PipeWire null sink + loopback cleanly separates "call in" from "our TTS out" | Route call audio through `voztrad_calls`; capture its **monitor** with `stream.capture.sink=true` |
| Virtual mic | null sink + `module-remap-source` shows up as a real `Audio/Source` | Use `voztrad_mic` + `voztrad_mic_input` |
| Global PTT key | `pynput` is focus-dependent under Wayland (misses keys); `evdev` is reliable | Use **python-evdev** (user in `input` group) |
| ASR VRAM | Parakeet TDT 0.6B v3 (onnx-asr, fp32) uses **~3.48 GB** → ~3.69 GB with MT+VAD on a 4 GB card | Needs the **fp16** Parakeet to leave headroom |
| onnxruntime-gpu | the current `1.31.x` wheel is a **CUDA 13** build | pin **`onnxruntime-gpu==1.22.0`** (CUDA 12) |
| Overlay on-top | works under **xcb**; native **wayland** has no keep-above protocol (KWin ignores it) | launch overlay with `QT_QPA_PLATFORM=xcb` |

---

## 1. System

```console
$ cat /etc/os-release        # NAME / PRETTY_NAME
NAME="Manjaro Linux"
PRETTY_NAME="Manjaro Linux"

$ uname -r
6.12.68-1-MANJARO

$ echo "$XDG_SESSION_TYPE / $XDG_CURRENT_DESKTOP"; loginctl show-session <id> -p Type -p Desktop
wayland / KDE
Desktop=KDE
Type=wayland

$ pipewire --version
Compiled with libpipewire 1.4.10
Linked with libpipewire 1.4.10
$ wireplumber --version
Compiled with libwireplumber 0.5.13
$ pactl info | grep -i 'server'
Server Name: PulseAudio (on PipeWire 1.4.10)
Server Version: 15.0.0

$ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
NVIDIA GeForce GTX 1650 Ti, 575.64.05, 4096 MiB, 7.5

$ nvidia-smi | grep -i 'CUDA Version'
| NVIDIA-SMI 575.64.05   Driver Version: 575.64.05   CUDA Version: 12.9 |
```

- Distro: **Manjaro Linux**, kernel **6.12.68-1-MANJARO**.
- Session: **Wayland**, **KDE Plasma** (compositor: KWin).
- Audio: **PipeWire 1.4.10**, **WirePlumber 0.5.13**, PulseAudio compat (`pactl`) 15.0.0.
- GPU: **GTX 1650 Ti**, 4096 MiB, compute capability **7.5** (Turing).
- NVIDIA driver **575.64.05**, reports **CUDA 12.9** (driver max; forward-compatible with CUDA 12 runtime).

Note: the task brief said "GTX 1650" — the actual card is a **1650 Ti**, still 4 GB.

---

## 2. Audio routing (PipeWire, CLI only, fully reversible)

Everything is done with `pactl` loadable modules — no config files touched, cleared by a
reboot or by the teardown below. The single real output present during the test was the
built-in analog sink `alsa_output.pci-0000_00_1f.3.analog-stereo` (treated as "headphones";
a Bluetooth headset was configured as default but not connected).

Full lifecycle lives in [`scripts/linux/voztrad-audio.sh`](../../scripts/linux/voztrad-audio.sh)
(`setup` / `teardown` / `status` / `record-calls` / `app-volume`).

### 2.a — Calls null sink + loopback, and excluding our own output

```bash
# create the sink the call audio is routed into
pactl load-module module-null-sink sink_name=voztrad_calls \
      sink_properties=device.description=voztrad_calls
# make the user hear the call: loopback the sink's monitor to the real output
pactl load-module module-loopback source=voztrad_calls.monitor \
      sink=alsa_output.pci-0000_00_1f.3.analog-stereo latency_msec=60
```

Capturing the monitor is the subtle part. **`pw-record --target voztrad_calls.monitor`
silently binds to the default source (the microphone)** — proven with `pw-link -l`, which
showed `alsa_input...capture_FL -> pw-record:input_FL`. The correct invocation targets the
sink node and sets `stream.capture.sink=true`:

```bash
pw-record --target voztrad_calls -P '{ stream.capture.sink=true }' out.wav
# pw-link -l then confirms:  voztrad_calls:monitor_FL -> pw-record:input_FL
```

Exclusion test (max volume measured with `ffmpeg volumedetect`):

| Condition | monitor `max_volume` |
|---|---|
| nothing playing (baseline) | **−91.0 dB** (digital silence) |
| audio played **straight to the real sink** | **−91.0 dB** → not captured ✅ |
| audio played **into `voztrad_calls`** | **−3.4 dB** → captured ✅ |

→ Audio that goes directly to the headphones (e.g. our own translated TTS) does **not**
appear in the `voztrad_calls` monitor, so the engine never re-captures/re-translates its
own output. This is the mechanism to exclude our output.

### 2.b — Virtual microphone (null sink + remap-source)

```bash
pactl load-module module-null-sink sink_name=voztrad_mic \
      sink_properties=device.description=voztrad_mic
pactl load-module module-remap-source master=voztrad_mic.monitor \
      source_name=voztrad_mic_input \
      source_properties=device.description='Voztrad Virtual Mic'
```

`wpctl status` lists it as a capture source apps can pick:

```
 │     100. voztrad_mic_input                    [Audio/Source]
```

Feeding the sink and recording the remapped source proves an app reading the virtual mic
hears what we write: playing a sample into `voztrad_mic` and recording `voztrad_mic_input`
→ **−3.4 dB** (signal present).

### 2.c — Change one app's volume by application name

`pactl` identifies playback streams as "sink-inputs"; match on `application.name`:

```bash
# helper (see voztrad-audio.sh app-volume):
idx=$(pactl list sink-inputs | awk -v app="Teams" '
  /^Sink Input #/{i=substr($3,2)} /application\.name = /{if(index($0,app)){print i; exit}}')
pactl set-sink-input-volume "$idx" 40%
```

Verified with a stream tagged `application.name=VoztradDemoApp`:

```
--- BEFORE ---  Volume: mono: 65535 / 100% / -0,00 dB
--- AFTER  ---  Volume: mono: 26214 /  40% / -23,88 dB
```

### Teardown (reversible)

`scripts/linux/voztrad-audio.sh teardown` unloads exactly the modules it loaded (ids saved
to `$XDG_RUNTIME_DIR/voztrad/audio-modules.env`) with a name-based fallback. Verified: after
teardown `pactl list short sinks|sources` shows no `voztrad*` entries.

---

## 3. Global push-to-talk key under Wayland — evdev vs pynput

The app uses **F8** as a global PTT hotkey. Scripts:
[`global_hotkey_evdev.py`](../../scripts/linux/global_hotkey_evdev.py),
[`global_hotkey_pynput.py`](../../scripts/linux/global_hotkey_pynput.py). Both ran
concurrently (one physical key press feeds both).

`evdev` needs read access to `/dev/input/event*`, which are `root:input 0660`:

```bash
sudo usermod -aG input "$USER"      # then `sg input -c ...` picks it up without re-login
```

Results (user pressed F8 repeatedly during a 25 s window):

| Reader | Captured | Notes |
|---|---|---|
| **python-evdev** | **8/8 presses** | counts `value==1` (key-down only); sees the raw kernel event on `Evision RGB Keyboard` / `AT Translated Set 2` regardless of focus |
| **pynput** | **inconsistent** | 0 in one run, 1-of-2 in another, 68 in a third (X11 auto-repeat) — depends entirely on focus |

pynput loads the `pynput.keyboard._xorg` backend, i.e. it reads from **XWayland**. It only
sees a global key when the focused window is an **XWayland** client (X11/Electron apps); when
focus is on a native Wayland window it misses the key. The 68-vs-8 run was the user holding
F8 over an XWayland-focused window, so pynput also logged X11 key auto-repeat that evdev
(key-down only) correctly ignores.

**Conclusion:** the common belief "pynput can't read keys under Wayland" is too simple — it
works *only* while an XWayland surface has focus, which makes it unreliable for a global PTT.
**Use python-evdev** (user in the `input` group); it reads the kernel input stream and is
focus-independent. This contradicts the brief's "expected: not under Wayland" for pynput:
it *can* capture, just not dependably.

---

## 4. GPU / models

Environment: **uv** venv, **Python 3.11**. Reproduce with
[`scripts/linux/setup_recon_env.sh`](../../scripts/linux/setup_recon_env.sh), measure with
[`scripts/linux/gpu_bench.sh`](../../scripts/linux/gpu_bench.sh) (full script:
[`gpu_bench.py`](../../scripts/linux/gpu_bench.py)).

### Environment gotchas (important for the port)

1. **onnxruntime-gpu CUDA 12 build.** `onnx-asr[gpu]` pulls `onnxruntime-gpu==1.31.0`, which
   is a **CUDA 13** build and fails on this box:
   `libcublasLt.so.13: cannot open shared object file ... Require cuDNN 9.* and CUDA 13.*`.
   Even the Microsoft "cuda-12" feed served the CUDA 13 wheel for 1.31.0. Fix: pin the last
   CUDA 12 build, **`onnxruntime-gpu==1.22.0`** (`ldd` → `libcublasLt.so.12`).
2. **Runtime CUDA libs.** CTranslate2 needs only cuBLAS; onnxruntime-gpu also needs cuRAND
   and cuFFT. Install the pip CUDA-12 packages and put them on `LD_LIBRARY_PATH` **before**
   starting Python (they are `dlopen`-ed at session creation):
   `nvidia-cublas/cudnn/cuda-runtime/cuda-nvrtc/curand/cufft-cu12`.
3. **onnx-asr external-data.** ORT 1.31 rejected Parakeet's external-data file
   ("External data path escapes model directory"); ORT **1.22 loads it fine**.
4. **Silero** 6.x `read_audio` needs torchaudio/torchcodec; load the wav with `soundfile` and
   pass a 1-D float32 tensor to `get_speech_timestamps` instead.

### Measurements (CUDA 12, GTX 1650 Ti 4 GB)

| Model | Backend / precision | Latency (p50 / min–max) | VRAM (load Δ) | Device |
|---|---|---|---|---|
| **opus-mt en→es** | CTranslate2 `int8_float16` | **32.6 ms** / 19.3–36.8 | **~194 MiB** | CUDA ✅ |
| **Parakeet TDT 0.6B v3** | onnx-asr, `CUDAExecutionProvider` (fp32 ONNX) | **86.2 ms** / 34.7–98.8 (for 2–4 s clips) | **~3484 MiB** | CUDA ✅ |
| **Silero VAD** | onnxruntime (onnx=True) | **29.8 ms** / 18.9–34.1 | **~0 MiB** | CPU |

- MT output verified correct, e.g. `"Let's push the launch to next week." → "Vamos a empujar el lanzamiento a la próxima semana."`
- ASR output verified correct, e.g. `A_en03.wav → "Could you share your screen so we can look at the dashboard?"`
- Parakeet session providers confirmed: `['CPUExecutionProvider', 'CUDAExecutionProvider']`.

**⚠ VRAM budget (the binding constraint on this card):**

```
peak with everything loaded: process 3678 MiB / GPU total 3686 MiB  of 4096 MiB  (~90%)
```

fp32 Parakeet alone is ~3.5 GB, leaving only ~0.4 GB of headroom once MT is also resident.
The Windows build already ships `scripts/convert_parakeet_fp16.py` ("1.5 GB VRAM") — the
Linux port should use the **fp16** Parakeet, otherwise there is no room for the refiner
(TranslateGemma / llama.cpp) that the project also loads.

### Idle clock behaviour

`gpu_bench.py` samples `clocks.sm,clocks.gr,clocks.mem` (MHz) while busy and after a 3 s idle:

```
clocks WHILE BUSY    : 1890, 1890, 5000
clocks AFTER 3s idle : 1350, 1350, 5000
```

→ The GPU **does down-clock when idle** (SM 1890 → 1350 MHz after 3 s; drops further to the
~300 MHz idle floor observed with the GPU fully unused). Memory clock stays at 5000 MHz.
Practical note for latency consistency: the first inference after an idle gap pays a
clock-ramp cost — a tiny periodic keep-alive inference (or `nvidia-smi -lgc`) can avoid the
cold-clock penalty on the first phrase of a call.

---

## 5. PySide6 overlay — frameless always-on-top, xcb vs wayland

The overlay ([`src/voztrad/ui/overlay.py`](../../src/voztrad/ui/overlay.py)) uses
`Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus`
plus `WA_TranslucentBackground` / `WA_ShowWithoutActivating`.
Test: [`overlay_ontop_test.py`](../../scripts/linux/overlay_ontop_test.py) replicates those
flags and is probed with `wmctrl` / `xprop`.

| `QT_QPA_PLATFORM` | plugin | window maps | stays-on-top evidence |
|---|---|---|---|
| **xcb** | `xcb` loads | ✅ visible, frameless, geometry honored | `xprop _NET_WM_STATE` = `_NET_WM_STATE_ABOVE, _NET_WM_STATE_STAYS_ON_TOP` → KWin keeps it above ✅ |
| **wayland** | `wayland` loads | ✅ visible | not in `wmctrl`/`xprop` (native Wayland surface); **no standard protocol to request keep-above** — KWin ignores `WindowStaysOnTopHint` for ordinary clients ❌ |

```console
# xcb
platformName (loaded) : xcb
_NET_WM_STATE(ATOM) = _NET_WM_STATE_DEMANDS_ATTENTION, _NET_WM_STATE_ABOVE, _NET_WM_STATE_STAYS_ON_TOP
# wayland
platformName (loaded) : wayland
(not in wmctrl list)
```

**Conclusion:** a frameless window stays reliably on top **only under `QT_QPA_PLATFORM=xcb`**
(via XWayland, where KWin honors `_NET_WM_STATE_ABOVE`). Under native Wayland the window maps
but there is no client-side keep-above; the user would have to add a manual KWin window rule.
Recommendation: **run the overlay with `QT_QPA_PLATFORM=xcb`**. (Bonus: xcb also lets the
global-key choice and absolute window positioning behave like the Windows build; native
Wayland additionally forbids clients from positioning themselves.)

---

## 6. Latency & quality optimization (for the port)

Goal: lowest latency and best audio/text quality within the 4 GB VRAM budget. All numbers
measured on this machine with [`scripts/linux/gpu_bench.py`](../../scripts/linux/gpu_bench.py)
and the extra benches noted below.

### 6.1 MT — opus-mt en→es, CTranslate2 compute-type sweep

p50 over 5 call-style sentences, beam_size=1 unless noted; "quality" = sentences differing
from `float32` output:

| compute_type | p50 latency | VRAM Δ | quality vs float32 |
|---|---|---|---|
| **`int8`** | **19.0 ms** | **170 MiB** | identical (0/5) |
| `int8_float32` | 19.6 ms | 170 MiB | identical (0/5) |
| `float32` | 20.5 ms | 362 MiB | (reference) |
| `int8_float16` *(repo's current choice)* | 25.7 ms | 226 MiB | 1/5 (synonym) |
| `float16` | **75.1 ms** | 202 MiB | identical — but **3× slower on Turing** |
| `int8_float16`, beam=4 | 28.7 ms | — | 3/5 differ, and **worse** ("dashboard"→"salpicadero") |

**Recommendation: switch MT to `int8`, beam_size=1.** It is the fastest (19 ms), smallest
(170 MiB), and bit-identical to float32 here. Avoid `float16` (3× slower on this GPU — no
tensor-core win for this model) and avoid beam>1 (slower and lower quality for this domain).

### 6.2 ASR — Parakeet TDT 0.6B v3 precision tradeoff (onnx-asr, CUDA)

Per 2–4 s clip; accuracy = exact-match vs fp32 on the English sample set:

| precision | p50 latency | VRAM Δ | on-disk | accuracy | notes |
|---|---|---|---|---|---|
| **fp32** | **~85 ms** (34–99) | **3484 MiB** | ~3.6 GB | reference | fast, but nearly fills the card |
| **fp16** | ~257 ms (steady-state) | **2516 MiB** | 1312 MB | 8/8 identical | **~3× slower** here despite less VRAM |
| int8 | ~317 ms | ~294 MiB (GPU) | — | 8/8 identical | ORT runs int8 on **CPU** on Turing → slow |

Surprising result: **fp16 is ~3× slower than fp32** on the GTX 1650 Ti (257 ms steady-state on
a fixed input, so not per-shape autotuning). The repo's `convert_parakeet_fp16.py`
(onnxruntime float16 converter, `keep_io_types=True`) saves ~1 GB of VRAM but regresses
latency badly — likely Cast churn / no tensor-core path for this graph. **int8** is accurate
but not CUDA-accelerated on Turing (falls to CPU, ~317 ms).

→ **For latency, keep fp32**; for VRAM-constrained coexistence, fp16 is the only option but its
latency must be optimized first (investigate ORT fp16 graph optimization / `op_block_list`,
or a tensor-core-friendly export) before relying on it. Do **not** assume fp16/int8 are free wins.

### 6.3 VRAM budget on 4 GB — the real constraint ⚠

| Resident set | VRAM |
|---|---|
| fp32 Parakeet + opus-mt `int8` + Silero | ~3.65 GB → fits **alone**, ~0.4 GB free |
| fp16 Parakeet + opus-mt `int8` + Silero | ~2.7 GB → ~1.3 GB free |
| llama.cpp **TranslateGemma-4B** refiner (Q4_K_M) | ~2.5–3 GB on its own |

The two-tier refiner the project adds (TranslateGemma-4B via llama.cpp) **cannot coexist with
Parakeet on this 4 GB card** — even fp16 Parakeet leaves < 1.5 GB. On the XPS the realistic
options are: run the refiner **on CPU**, use a **much smaller** refiner, or **drop the refiner**
and ship opus-mt `int8` alone (19 ms, good quality). The original "4.46–4.69 GB of 6.1" figure
in the git history was on a 6 GB card; it does not transfer to 4 GB.

### 6.4 Audio latency (PipeWire)

- Global clock: **48 kHz, quantum 1024 (~21.3 ms)**, min-quantum 32 (~0.67 ms), max 2048.
- **Capture the call monitor directly at 16 kHz mono** (`pw-record --rate 16000 --channels 1`):
  PipeWire resamples 48k→16k for free, so no separate resampler stage before the ASR.
- A small capture quantum — `PIPEWIRE_LATENCY=256/48000` (~5.3 ms) — ran **without xruns**, well
  under the 21 ms default. Use it on the capture node to cut input latency.
- The loopback that lets you hear the raw call adds `latency_msec=60`. For a translator you may
  lower it or not monitor the raw call at all; it does not affect the ASR capture path.

### 6.5 GPU cold-clock penalty (and keep-alive)

The GPU down-clocks when idle; the first inference after a gap pays a ramp cost
(fp16 Parakeet, warm p50 257 ms):

| state | SM clock | first inference |
|---|---|---|
| warm (steady) | 1905 MHz | 257 ms |
| after 3 s idle | 1350 MHz | 301 ms (**+44 ms**) |
| after 8 s idle | 1350 MHz | 333 ms (**+75 ms**) |

This mirrors the project's own `src/voztrad/gpuwarm.py` note (Windows: P8 @ 210 MHz →
ASR 44 → ~280 ms). → Keep a **tiny periodic keep-alive inference** running between phrases, or
pin clocks with `nvidia-smi -lgc`, so the first word of each utterance isn't slow. Port
`gpuwarm.py` to Linux (it already drives a Silero session on CUDA to hold the clock up).

---

## Appendix — reproduce

```bash
# audio (reversible)
scripts/linux/voztrad-audio.sh setup
scripts/linux/voztrad-audio.sh record-calls /tmp/calls.wav 5
scripts/linux/voztrad-audio.sh app-volume "Teams" 40
scripts/linux/voztrad-audio.sh teardown

# global key (needs: sudo usermod -aG input $USER)
sg input -c '.venv-recon/bin/python scripts/linux/global_hotkey_evdev.py 20'
.venv-recon/bin/python scripts/linux/global_hotkey_pynput.py 20

# GPU benchmark
scripts/linux/setup_recon_env.sh
scripts/linux/gpu_bench.sh

# overlay
QT_QPA_PLATFORM=xcb     .venv-recon/bin/python scripts/linux/overlay_ontop_test.py 5
QT_QPA_PLATFORM=wayland .venv-recon/bin/python scripts/linux/overlay_ontop_test.py 5
```

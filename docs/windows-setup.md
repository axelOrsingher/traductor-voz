# Windows setup notes

CUDA/cuDNN redist layout and the Windows-specific fixes that make the engine run and stay fast.
(For the everyday flow see the [README](../README.md#windows-1011).)

## CUDA / cuDNN redist

The ONNX Runtime GPU build needs CUDA 12 + cuDNN 9 libraries at load time. Put the NVIDIA
redists under `~/tools/` (paths are resolved in [`src/voztrad/cuda_dlls.py`](../src/voztrad/cuda_dlls.py)):

- cudart ≥ 12.8 — e.g. `~/tools/cuda_cudart-12.9.79`
- cuDNN 9 (CUDA 12) — e.g. `~/tools/cudnn-9.27.0.42_cuda12/bin/x64`

Paths are overridable with `VOZTRAD_CUDART_BIN`, `VOZTRAD_CUDNN_BIN`, `VOZTRAD_CUDA_BIN`.

## What breaks on a fresh machine and how it is resolved

| Problem | Fix |
|---|---|
| PyPI `onnxruntime-gpu` 1.30 is a **CUDA 13** build → silently falls back to CPU | pin 1.29.0 from Microsoft's CUDA 12 feed (`[tool.uv.index] ort-cuda12` in `pyproject.toml`) |
| `piper-tts` drags in CPU `onnxruntime`, clobbering the GPU build | `override-dependencies` in `pyproject.toml` |
| ORT CUDA 12 needs cudart ≥ 12.8 symbols; installed toolkit is 12.6 | cudart 12.9 redist in `~/tools/cuda_cudart-12.9.79`, preloaded in `cuda_dlls.py` |
| cuDNN 9 missing | redist 9.27 in `~/tools/cudnn-9.27.0.42_cuda12/bin/x64` (used by ORT; CT2 ships its own dispatcher) |
| Windows 11 power-throttles windowless processes (EcoQoS): MT 45 → 190 ms after idle | `winperf.disable_throttling()` on startup |
| ORT spin-wait steals CPU from MT (900 ms spikes) | `allow_spinning=0` on ORT sessions |
| GPU drops to P8/210 MHz after ~2 s idle: ASR 2–6× slower | `gpuwarm.py` loads the GPU only while there is speech (~0.6 s is enough). Admin alternative: `nvidia-smi -lgc 1500,3105` |
| Installing VB-CABLE makes it the default output (you hear nothing) | `scripts\set_default_audio.ps1 -Match Speakers` |

On **Linux** these are handled differently: the CUDA 12 libraries are pip `nvidia-*-cu12`
packages pulled in by `uv sync` (no `~/tools` redist), the GPU idle-clock mitigation is the same
`gpuwarm.py`, and there is no EcoQoS/spin-wait tuning. See
[`docs/porting/linux-recon.md`](porting/linux-recon.md).

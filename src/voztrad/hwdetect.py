"""Hardware detection + profile selection.

Probes the GPU (name / VRAM / CUDA compute capability), CPU and audio stack, then
picks model backends and precisions that fit. The tiers encode what the Linux recon
measured on this class of hardware (docs/porting/linux-recon.md §6):

  - Turing (cc 7.5): MT `int8` beats `float16`; Parakeet int8 falls to CPU (slow),
    fp16 is ~3x slower than fp32 -> prefer fp32 when it fits.
  - Ampere+ (cc >= 8.0): float16 wins (tensor cores).
  - 4 GB card: Parakeet fp32 (~3.5 GB) fits alone, no room for the TranslateGemma
    refiner; 6-8 GB: fp16 Parakeet + refiner; >= 8 GB: fp32 + refiner.

User flags always override the chosen profile (see __main__.cmd_run)."""

from __future__ import annotations

import os
import subprocess
import sys

from .config import VOICE_REFS


def _nvidia() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        if not out:
            return None
        name, mem, cc = (x.strip() for x in out.splitlines()[0].split(","))
        return {"name": name, "vram_mib": int(float(mem)), "compute_cap": float(cc)}
    except Exception:
        return None


def _ram_gb() -> float:
    try:
        import psutil  # os.sysconf doesn't exist on Windows
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        return 0.0


def _audio_server() -> str:
    try:
        info = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=3).stdout
        for ln in info.splitlines():
            if ln.startswith("Server Name:"):
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _llama_server() -> str | None:
    from .refine import find_server
    exe = find_server()
    return str(exe) if exe else None


def detect() -> dict:
    gpu = _nvidia()
    return {
        "platform": sys.platform,
        "gpu": gpu,
        "has_cuda": gpu is not None,
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": _ram_gb(),
        "audio_server": _audio_server(),
        "has_voice_ref": any(p.exists() for p in VOICE_REFS),
        "llama_server": _llama_server(),
    }


def choose_profile(hw: dict) -> dict:
    """Return Config-shaped overrides + human-readable notes.

    Smart Turn (ONNX, CPU, ~46 ms per decision) is on in every profile: it closes a
    phrase after ~200 ms of silence instead of the 500 ms hangover (~250 ms less
    end-to-end, measured on Windows)."""
    notes: list[str] = []
    linux = hw["platform"] != "win32"
    tts = "pocket" if hw["has_voice_ref"] else "piper"

    if not hw["has_cuda"]:
        notes.append("no NVIDIA GPU: everything on CPU (slow ASR, ~300 ms+). A GPU is recommended.")
        return {"asr": "parakeet", "parakeet_quant": "int8", "mt_compute_type": "int8",
                "refine": False, "smart_turn": True, "gpu_warm": False, "tts": tts,
                "notes": notes}

    g = hw["gpu"]
    cc, vram = g["compute_cap"], g["vram_mib"]
    ampere = cc >= 8.0
    mt_ct = "float16" if ampere else "int8"
    notes.append(f"GPU {g['name']} ({vram} MiB, cc {cc}) -> MT {mt_ct}"
                 + ("" if ampere else " (int8 is faster than float16 on Turing)"))
    can_refine = vram >= 6000 and hw.get("llama_server") is not None

    if vram >= 8000:
        quant = None  # fp32
        refine = can_refine
        notes.append(">=8 GB: Parakeet fp32" + (" + refiner" if refine else ""))
    elif vram >= 6000:
        quant = "fp16"
        refine = can_refine
        notes.append("6-8 GB: Parakeet fp16" + (" + refiner" if refine else ""))
    else:
        quant = None  # fp32 ~3.5 GB fits alone and is the fastest on this card
        refine = False
        notes.append(f"<6 GB: Parakeet fp32 alone, no refiner (doesn't fit in {vram} MiB)")
        if linux:
            notes.append("fp16 would save VRAM but is ~3x slower here (see recon §6)")
    if vram >= 6000 and not can_refine:
        notes.append("refiner off: llama-server not found (scripts/fetch_models.py, or set "
                     "VOZTRAD_LLAMA_SERVER)")

    return {"asr": "parakeet", "parakeet_quant": quant, "mt_compute_type": mt_ct,
            "refine": refine, "smart_turn": True, "gpu_warm": True, "tts": tts,
            "notes": notes}

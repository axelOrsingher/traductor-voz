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
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
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
    }


def choose_profile(hw: dict) -> dict:
    """Return Config-shaped overrides + human-readable notes."""
    notes: list[str] = []
    linux = hw["platform"] != "win32"
    tts = "pocket" if hw["has_voice_ref"] else "piper"

    if not hw["has_cuda"]:
        notes.append("sin GPU NVIDIA: todo en CPU (ASR lento, ~300 ms+). Mejor con GPU.")
        return {"asr": "parakeet", "parakeet_quant": "int8", "mt_compute_type": "int8",
                "refine": False, "smart_turn": False, "gpu_warm": False, "tts": tts,
                "notes": notes}

    g = hw["gpu"]
    cc, vram = g["compute_cap"], g["vram_mib"]
    ampere = cc >= 8.0
    mt_ct = "float16" if ampere else "int8"
    notes.append(f"GPU {g['name']} ({vram} MiB, cc {cc}) -> MT {mt_ct}"
                 + ("" if ampere else " (int8 es más rápido que float16 en Turing)"))
    # the TranslateGemma refiner (llama.cpp) is only wired on Windows for now
    can_refine = (not linux) and vram >= 6000

    if vram >= 8000:
        quant = None  # fp32
        refine = can_refine
        notes.append("≥8 GB: Parakeet fp32" + (" + refinador" if refine else ""))
    elif vram >= 6000:
        quant = "fp16"
        refine = can_refine
        notes.append("6-8 GB: Parakeet fp16" + (" + refinador" if refine else ""))
    else:
        quant = None  # fp32 ~3.5 GB fits alone and is the fastest on this card
        refine = False
        notes.append(f"<6 GB: Parakeet fp32 solo, sin refinador (no entra en {vram} MiB)")
        if linux:
            notes.append("fp16 ahorraría VRAM pero es ~3x más lento acá (ver recon §6)")

    return {"asr": "parakeet", "parakeet_quant": quant, "mt_compute_type": mt_ct,
            "refine": refine, "smart_turn": False, "gpu_warm": True, "tts": tts,
            "notes": notes}

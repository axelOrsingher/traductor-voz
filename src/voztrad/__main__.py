"""CLI del motor.

  python -m voztrad run    [--asr whisper|parakeet] [--b-mode ptt|vad] [--ptt-key F8] ...
  python -m voztrad bench  DIR [DIR...] [--asr whisper,parakeet]
  python -m voztrad report LOG.jsonl [...]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from .config import ROOT, SR, Config  # noqa: E402


def cmd_run(a):
    from .engine import Engine, setup_console
    setup_console()
    if _port_open(a.port):
        sys.exit(f"ya hay un motor corriendo en el puerto {a.port}: cerralo primero (o usá --port)")
    cfg = Config(asr=a.asr, b_mode=a.b_mode, ptt_key=a.ptt_key, mic=a.mic, loopback=a.loopback,
                 cable=a.cable, normalize_voseo=not a.no_norm, parakeet_quant=a.parakeet_quant,
                 enable_a=not a.no_a, enable_b=not a.no_b, port=a.port,
                 gpu_warm=not a.no_gpu_warm, tts=a.tts,
                 voice_ref=Path(a.voice) if a.voice else None, listen=not a.no_listen)
    if a.listen_voice:
        cfg.listen_voice = Path(a.listen_voice)
    cfg.seg.hangover_ms = a.hangover
    Engine(cfg, save_segments=a.save_segments).run()


def cmd_voice(a):
    from . import voice
    from .engine import setup_console
    setup_console()
    if a.action == "use":
        if not a.which:
            sys.exit("uso: python -m voztrad voice use es|en|mix")
        voice.use(a.which)
    elif a.action == "preview":
        voice.previews()
    else:
        voice.record(a.mic)


def _port_open(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


_mutex = None


def _single_instance(name: str) -> bool:
    """Mutex con nombre de Windows: False si otra instancia ya lo tiene."""
    global _mutex
    if sys.platform != "win32":
        return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    _mutex = k32.CreateMutexW(None, False, f"Local\\{name}")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def cmd_app(a):
    """Motor en un subproceso (sin consola) + overlay en este proceso."""
    import socket

    if not _single_instance("voztrad-overlay"):
        sys.exit(0)  # ya hay un overlay abierto: no duplicar
    from .ui.overlay import main as ui_main
    if _port_open(a.port):
        # ya hay un motor corriendo (p.ej. quedó de antes): solo abrir el overlay
        sys.exit(ui_main(port=a.port))
    run_argv = [x for x in sys.argv[sys.argv.index("app") + 1:]]
    (ROOT / "logs").mkdir(exist_ok=True)
    out = open(ROOT / "logs" / "engine-console.txt", "w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    exe = sys.executable.replace("pythonw.exe", "python.exe")
    eng = subprocess.Popen([exe, "-m", "voztrad", "run", *run_argv], cwd=ROOT, stdout=out,
                           stderr=subprocess.STDOUT, creationflags=flags,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        code = ui_main(port=a.port)
    finally:
        try:  # apagado prolijo: el motor imprime el resumen en logs/engine-console.txt
            with socket.create_connection(("127.0.0.1", a.port), timeout=1) as s:
                s.sendall(b'{"cmd": "stop"}\n')
            eng.wait(timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            eng.kill()
        out.close()
    sys.exit(code)


def _vram() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def cmd_bench(a):
    """Mismo audio por ambos backends. Archivos A_*.wav -> inglés, B_*.wav -> español."""
    import soundfile as sf
    import soxr

    from . import asr as asr_mod
    from .engine import setup_console
    from .mt import Translator
    from .voseo import normalize
    setup_console()

    wavs = sorted(p for d in a.dirs for p in Path(d).glob("*.wav") if p.name[:2] in ("A_", "B_"))
    if not wavs:
        sys.exit("no hay A_*.wav / B_*.wav en " + ", ".join(a.dirs))
    clips = []
    for p in wavs:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = soxr.resample(x, sr, SR).astype("float32")
        clips.append((p, "en" if p.name.startswith("A_") else "es", x))

    mt = {"en": Translator("en", "es"), "es": Translator("es", "en")}
    results = {}
    for kind in a.asr.split(","):
        base = _vram()
        t = time.perf_counter()
        model = asr_mod.load(kind, parakeet_quant=a.parakeet_quant)
        load_s, vram = time.perf_counter() - t, _vram() - base
        print(f"\n### {model.name}  (carga {load_s:.1f}s, VRAM +{vram} MiB)")
        rows = []
        for p, lang, x in clips:
            ms = []
            for _ in range(a.repeat):
                t = time.perf_counter()
                text = model.transcribe(x, lang)
                ms.append((time.perf_counter() - t) * 1000)
            src = normalize(text) if lang == "es" else text
            t = time.perf_counter()
            tr = mt[lang].translate(src)
            mt_ms = (time.perf_counter() - t) * 1000
            rows.append({"file": p.name, "lang": lang, "audio_s": round(len(x) / SR, 2),
                         "asr_ms": round(statistics.median(ms), 1), "mt_ms": round(mt_ms, 1),
                         "text": text, "translation": tr})
            print(f"  {p.name:14} {len(x) / SR:5.2f}s  asr {statistics.median(ms):6.0f} ms  "
                  f"mt {mt_ms:4.0f} ms | {text}\n{'':38}-> {tr}")
        results[model.name] = {"load_s": load_s, "vram_mib": vram, "rows": rows}
        del model
        import gc
        gc.collect()

    print("\n" + "=" * 72 + "\nASR ms por backend y dirección (p50 / p95 / max)")
    for name, r in results.items():
        for lang in ("en", "es"):
            v = [x["asr_ms"] for x in r["rows"] if x["lang"] == lang]
            if v:
                v.sort()
                p95 = v[min(len(v) - 1, round(0.95 * (len(v) - 1)))]
                print(f"  {name:32} {lang}  n={len(v):3}  {statistics.median(v):6.0f} / {p95:6.0f} / {max(v):6.0f}")
    out = ROOT / "out" / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


def cmd_report(a):
    from .engine import setup_console, summary
    setup_console()
    rows = []
    for f in a.logs:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("type") == "phrase":
                rows.append(ev)
    for asr_name in sorted({r["asr"] for r in rows}):
        print(summary([r for r in rows if r["asr"] == asr_name]))


def main():
    ap = argparse.ArgumentParser(prog="voztrad")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="motor en vivo (headless)")
    _run_args(r)
    r.set_defaults(fn=cmd_run)

    ap_app = sub.add_parser("app", help="motor + overlay de subtítulos (cerrar el overlay apaga todo)")
    _run_args(ap_app)
    ap_app.set_defaults(fn=cmd_app)

    u = sub.add_parser("ui", help="solo el overlay (se conecta a un motor ya corriendo)")
    u.add_argument("--port", type=int, default=8765)
    u.set_defaults(fn=lambda a: sys.exit(__import__("voztrad.ui.overlay", fromlist=["main"]).main(port=a.port)))

    v = sub.add_parser("voice", help="grabar tu voz para clonarla / elegir referencia")
    v.add_argument("action", nargs="?", default="record", choices=["record", "use", "preview"])
    v.add_argument("which", nargs="?", choices=["es", "en", "mix"])
    v.add_argument("--mic", default=None)
    v.set_defaults(fn=cmd_voice)

    b = sub.add_parser("bench", help="comparar backends de ASR sobre los mismos wav")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--asr", default="whisper,parakeet")
    b.add_argument("--parakeet-quant", choices=["int8"], default=None)
    b.add_argument("--repeat", type=int, default=3)
    b.set_defaults(fn=cmd_bench)

    rp = sub.add_parser("report", help="p50/p95 por etapa desde logs/*.jsonl")
    rp.add_argument("logs", nargs="+")
    rp.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    from .winperf import disable_throttling
    disable_throttling()
    a.fn(a)


def _run_args(r):
    r.add_argument("--asr", choices=["whisper", "parakeet"], default="parakeet")
    r.add_argument("--parakeet-quant", choices=["int8"], default=None)
    r.add_argument("--b-mode", choices=["ptt", "vad", "interp"], default="ptt",
                   help="ptt = mantener tecla; vad = manos libres; interp = manos libres que va traduciendo en las pausas")
    r.add_argument("--ptt-key", default="F8")
    r.add_argument("--hangover", type=int, default=500, help="ms de silencio para cerrar frase")
    r.add_argument("--mic", default=None)
    r.add_argument("--loopback", default=None)
    r.add_argument("--cable", default="CABLE Input")
    r.add_argument("--no-norm", action="store_true", help="no normalizar rioplatense antes del MT")
    r.add_argument("--no-a", action="store_true", help="desactivar dirección A (ellos -> yo)")
    r.add_argument("--no-b", action="store_true", help="desactivar dirección B (yo -> ellos)")
    r.add_argument("--save-segments", action="store_true", help="guardar cada segmento en out/segments/")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-gpu-warm", action="store_true", help="no calentar la GPU durante la voz")
    r.add_argument("--tts", choices=["pocket", "piper"], default="pocket",
                   help="pocket = tu voz clonada (models/voice/me.wav); piper = voz genérica")
    r.add_argument("--voice", default=None, help="wav de referencia para clonar (default models/voice/me.wav)")
    r.add_argument("--no-listen", action="store_true", help="no reproducir la traducción al castellano")
    r.add_argument("--listen-voice", default=None, help="voz Piper para el castellano (default es_MX-ald-medium)")


if __name__ == "__main__":
    main()

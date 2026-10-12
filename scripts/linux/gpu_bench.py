#!/usr/bin/env python3
"""GPU recon: VRAM + latency for opus-mt (CT2 int8_float16), Parakeet TDT 0.6B v3
(onnx-asr, CUDA 12) and Silero VAD. Also logs GPU clocks busy vs after 3s idle."""
import glob, os, subprocess, statistics, time, traceback

import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
SAMPLES = sorted(glob.glob(f"{ROOT}/samples/synth/*.wav"))
PID = os.getpid()

def nvsmi(a): return subprocess.run(["nvidia-smi"]+a, capture_output=True, text=True).stdout
def proc_vram():
    for ln in nvsmi(["--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"]).splitlines():
        p=[x.strip() for x in ln.split(",")]
        if len(p)==2 and p[0].isdigit() and int(p[0])==PID: return int(p[1])
    return 0
def total_vram():
    o=nvsmi(["--query-gpu=memory.used","--format=csv,noheader,nounits"]).strip().splitlines()
    return int(o[0]) if o else -1
def gclocks():
    return nvsmi(["--query-gpu=clocks.sm,clocks.gr,clocks.mem","--format=csv,noheader,nounits"]).strip()
def section(n): print(f"\n{'='*60}\n### {n}\n{'='*60}", flush=True)

print(f"PID={PID}  samples={len(SAMPLES)}", flush=True)
print("baseline proc VRAM:", proc_vram(), "MiB  total GPU used:", total_vram(), "MiB", flush=True)

ct2_translate = None   # keep a callable for the clock stress test

# ---------------------------------------------------------------- opus-mt CT2
section("opus-mt en->es  CTranslate2 int8_float16 (CUDA)")
try:
    import ctranslate2
    from transformers import AutoTokenizer
    ct2_dir = os.environ.get("CT2_DIR", "/tmp/opus-mt-en-es-ct2")
    if not os.path.isdir(ct2_dir):
        print("converting Helsinki-NLP/opus-mt-en-es -> CT2 int8_float16 ...", flush=True)
        subprocess.run(["ct2-transformers-converter","--model","Helsinki-NLP/opus-mt-en-es",
                        "--output_dir",ct2_dir,"--quantization","int8_float16"], check=True)
    v0 = proc_vram()
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tr = ctranslate2.Translator(ct2_dir, device="cuda", compute_type="int8_float16")
    sents = ["Can you hear me okay?", "Let's push the launch to next week.",
             "I think we should review the budget before deciding.",
             "The audio quality on this call is excellent today.",
             "Could you share your screen so we can look at the dashboard?"]
    def translate(s):
        src = tok.convert_ids_to_tokens(tok.encode(s))
        res = tr.translate_batch([src], max_batch_size=1)
        return tok.decode(tok.convert_tokens_to_ids(res[0].hypotheses[0]), skip_special_tokens=True)
    ct2_translate = lambda: translate(sents[0])
    print("warmup:", translate(sents[0]), flush=True)
    time.sleep(0.2); v1 = proc_vram()
    lat=[]
    for s in sents:
        t=time.perf_counter(); out=translate(s); lat.append((time.perf_counter()-t)*1000)
        print(f"  [{lat[-1]:6.1f} ms] {s!r} -> {out!r}", flush=True)
    print(f"opus-mt latency ms: p50={statistics.median(lat):.1f} min={min(lat):.1f} max={max(lat):.1f}", flush=True)
    print(f"opus-mt VRAM: load delta ~{v1-v0} MiB (proc total {v1} MiB)", flush=True)
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- Parakeet
section("Parakeet TDT 0.6B v3  (onnx-asr, CUDAExecutionProvider, CUDA 12)")
try:
    import onnx_asr, soundfile as sf, onnxruntime
    vP0 = proc_vram()
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",
                                providers=["CUDAExecutionProvider"])
    # deep-scan the adapter for the real onnxruntime sessions and their providers
    provs=set()
    def _scan(o, seen=set()):
        if id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, onnxruntime.InferenceSession): provs.update(o.get_providers()); return
        for a in getattr(o,'__dict__',{}).values():
            try: _scan(a, seen)
            except Exception: pass
    _scan(model)
    print("loaded parakeet v3; ACTUAL session providers:", sorted(provs), flush=True)
    print("warmup:", model.recognize(SAMPLES[0]), flush=True)
    time.sleep(0.2); vP1 = proc_vram()
    lat=[]
    for w in SAMPLES[:8]:
        dur = len(sf.read(w)[0])/sf.read(w)[1]
        t=time.perf_counter(); txt=model.recognize(w); ms=(time.perf_counter()-t)*1000
        lat.append(ms); print(f"  [{ms:7.1f} ms | {dur:4.2f}s audio] {os.path.basename(w)} -> {txt!r}", flush=True)
    print(f"parakeet latency ms: p50={statistics.median(lat):.1f} min={min(lat):.1f} max={max(lat):.1f}", flush=True)
    print(f"parakeet VRAM: load delta ~{vP1-vP0} MiB (proc total {vP1} MiB)", flush=True)
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- Silero VAD
section("Silero VAD (onnx)")
try:
    import numpy as np, torch, soundfile as sf
    from silero_vad import load_silero_vad, get_speech_timestamps
    vS0 = proc_vram()
    vad = load_silero_vad(onnx=True)
    vS1 = proc_vram()
    lat=[]
    for w in SAMPLES[:8]:
        data, sr = sf.read(w, dtype="float32")
        if data.ndim>1: data = data.mean(axis=1)
        audio = torch.from_numpy(data)
        t=time.perf_counter(); ts=get_speech_timestamps(audio, vad, sampling_rate=16000); ms=(time.perf_counter()-t)*1000
        lat.append(ms); print(f"  [{ms:7.2f} ms] {os.path.basename(w)} -> {len(ts)} speech segs", flush=True)
    print(f"silero latency ms: p50={statistics.median(lat):.2f} min={min(lat):.2f} max={max(lat):.2f}", flush=True)
    print(f"silero VRAM: load delta ~{vS1-vS0} MiB (onnxruntime CPU EP by default)", flush=True)
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- peak VRAM
section("peak VRAM with everything loaded")
print(f"process VRAM: {proc_vram()} MiB   GPU total used: {total_vram()} MiB / 4096 MiB", flush=True)

# ---------------------------------------------------------------- idle clocks
section("clock behaviour: busy vs after 3s idle (sm,gr,mem MHz)")
try:
    if ct2_translate:
        print("clocks AT IDLE (before stress):", gclocks(), flush=True)
        end=time.time()+2.5
        while time.time()<end: ct2_translate()   # keep GPU busy ~2.5s
        print("clocks WHILE BUSY:", gclocks(), flush=True)
        time.sleep(3)
        print("clocks AFTER 3s idle:", gclocks(), flush=True)
    else:
        print("no CT2 workload available; raw before/after:")
        print("before:", gclocks()); time.sleep(3); print("after 3s:", gclocks())
except Exception:
    traceback.print_exc()
print("\nALL_DONE_MARKER", flush=True)

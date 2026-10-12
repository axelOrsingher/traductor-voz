#!/usr/bin/env python3
"""Convert Parakeet TDT 0.6B v3 encoder to fp16 (repo method) and benchmark it on
CUDA: VRAM, latency, accuracy vs fp32, plus a proper GPU cold-clock penalty test."""
import os, glob, shutil, statistics, subprocess, time, traceback
from pathlib import Path
import pathlib
ROOT=str(pathlib.Path(__file__).resolve().parents[2])
SAMPLES=sorted(glob.glob(f"{ROOT}/samples/synth/A_en*.wav"))
PID=os.getpid()
def vram():
    for ln in subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                             capture_output=True,text=True).stdout.splitlines():
        p=[x.strip() for x in ln.split(",")]
        if len(p)==2 and p[0].isdigit() and int(p[0])==PID: return int(p[1])
    return 0
def smhz():
    return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader,nounits"],
                          capture_output=True,text=True).stdout.strip().splitlines()[0]
FP32={"A_en01.wav":"Can you hear me okay?","A_en02.wav":"I think we should push the release to Monday.",
 "A_en03.wav":"Could you share your screen so we can look at the dashboard?",
 "A_en04.wav":"The latency numbers look much better after the last deploy.",
 "A_en05.wav":"Sorry, you were on mute for a second.","A_en06.wav":"Let's take this offline and sync tomorrow morning.",
 "A_en07.wav":"Who owns the database migration?","A_en08.wav":"We still have three open tickets from the last sprint."}

OUT=Path("/tmp/parakeet-tdt-0.6b-v3-fp16")
try:
    if not (OUT/"encoder-model.onnx").exists():
        print("converting encoder to fp16 ...",flush=True)
        import onnx
        from huggingface_hub import snapshot_download
        from onnxruntime.transformers.float16 import convert_float_to_float16
        snap=Path(snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx",
                 allow_patterns=["encoder-model.onnx*","decoder_joint-model.onnx","config.json","vocab.txt"]))
        # HF stores big files as symlinks to blobs; onnx.load refuses symlinked external data.
        # Copy real bytes (shutil.copy follows symlinks) into a plain dir first.
        src=Path("/tmp/parakeet-src-real"); src.mkdir(parents=True,exist_ok=True)
        for f in ("encoder-model.onnx","encoder-model.onnx.data","decoder_joint-model.onnx","config.json","vocab.txt"):
            shutil.copy(snap/f, src/f)
        OUT.mkdir(parents=True,exist_ok=True)
        for f in ("decoder_joint-model.onnx","config.json","vocab.txt"): shutil.copy(src/f,OUT/f)
        m16=convert_float_to_float16(onnx.load(src/"encoder-model.onnx"),keep_io_types=True,disable_shape_infer=True)
        onnx.save_model(m16,OUT/"encoder-model.onnx",save_as_external_data=True,
                        all_tensors_to_one_file=True,location="encoder-model.onnx.data")
        print("converted ->",OUT,flush=True)
    sz=sum(f.stat().st_size for f in OUT.glob("*"))/1e6
    print(f"fp16 model on-disk size: {sz:.0f} MB",flush=True)

    import onnx_asr
    v0=vram()
    m=onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",path=str(OUT),providers=["CUDAExecutionProvider"])
    m.recognize(SAMPLES[0]); time.sleep(0.2); v1=vram()
    print(f"\nfp16 VRAM load delta ~{v1-v0} MiB (proc total {v1} MiB)",flush=True)
    lat=[]; agree=0
    for w in SAMPLES:
        t=time.perf_counter(); txt=m.recognize(w); ms=(time.perf_counter()-t)*1000; lat.append(ms)
        b=os.path.basename(w); exp=FP32.get(b)
        tag="" if exp is None else ("  OK" if txt.strip()==exp.strip() else "  != "+repr(exp))
        if exp is not None and txt.strip()==exp.strip(): agree+=1
        print(f"  [{ms:6.1f} ms] {b} -> {txt!r}{tag}",flush=True)
    print(f"fp16 latency ms: p50={statistics.median(lat):.1f} min={min(lat):.1f} max={max(lat):.1f}",flush=True)
    print(f"fp16 vs fp32 exact-match: {agree}/{len(SAMPLES)}",flush=True)

    # ---- steady-state on a SINGLE fixed input (isolates per-shape cuDNN autotuning) ----
    print("\n--- steady-state, same input x8 (same tensor shape) ---",flush=True)
    rep=[]
    for _ in range(8):
        t=time.perf_counter(); m.recognize(SAMPLES[0]); rep.append((time.perf_counter()-t)*1000)
    print("  per-call ms:", " ".join(f"{x:.0f}" for x in rep),flush=True)
    print(f"  steady-state p50={statistics.median(rep):.1f} ms (first={rep[0]:.1f})",flush=True)
    # ---- proper GPU cold-clock penalty: idle long enough for the card to down-clock ----
    print("\n--- GPU cold-clock penalty (fp16, GPU-resident) ---",flush=True)
    warm=rep[-4:]
    wp=statistics.median(warm)
    print(f"warm SM={smhz()}MHz  warm p50={wp:.1f} ms",flush=True)
    for idle in (3,8):
        time.sleep(idle)
        sm_before=smhz()
        t=time.perf_counter(); m.recognize(SAMPLES[0]); cold=(time.perf_counter()-t)*1000
        print(f"after {idle}s idle: SM(before infer)={sm_before}MHz  first-infer={cold:.1f} ms  penalty=+{cold-wp:.1f} ms",flush=True)
except Exception: traceback.print_exc()
print("\nALL_DONE_MARKER",flush=True)

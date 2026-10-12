#!/usr/bin/env python3
"""Optimization bench: MT compute-type sweep, Parakeet int8 vs fp32 (VRAM/latency/
accuracy), and cold-vs-warm clock penalty. One process; models loaded/freed in turn."""
import glob, os, gc, subprocess, statistics, time, traceback
import pathlib
ROOT=str(pathlib.Path(__file__).resolve().parents[2])
SAMPLES=sorted(glob.glob(f"{ROOT}/samples/synth/A_en*.wav"))
ES=sorted(glob.glob(f"{ROOT}/samples/synth/B_rio*.wav"))
PID=os.getpid()
def vram():
    for ln in subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                             capture_output=True,text=True).stdout.splitlines():
        p=[x.strip() for x in ln.split(",")]
        if len(p)==2 and p[0].isdigit() and int(p[0])==PID: return int(p[1])
    return 0
def sec(n): print(f"\n{'='*64}\n### {n}\n{'='*64}",flush=True)

SENTS=["Can you hear me okay?","Let's push the launch to next week.",
       "I think we should review the budget before deciding.",
       "The audio quality on this call is excellent today.",
       "Could you share your screen so we can look at the dashboard?"]

# -------------------------------------------------- MT compute-type sweep
sec("opus-mt en->es  CTranslate2 compute-type sweep")
try:
    import ctranslate2
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    base="/tmp/opus-mt-en-es"
    # build one fp32 CT2 model, let Translator cast per compute_type
    if not os.path.isdir(base):
        subprocess.run(["ct2-transformers-converter","--model","Helsinki-NLP/opus-mt-en-es",
                        "--output_dir",base],check=True)
    def run_ct(ct, beam=1):
        v0=vram()
        tr=ctranslate2.Translator(base,device="cuda",compute_type=ct)
        def tr1(s):
            src=tok.convert_ids_to_tokens(tok.encode(s))
            r=tr.translate_batch([src],beam_size=beam,max_batch_size=1)
            return tok.decode(tok.convert_tokens_to_ids(r[0].hypotheses[0]),skip_special_tokens=True)
        tr1(SENTS[0]); time.sleep(0.1); v1=vram()
        lat=[]; outs=[]
        for s in SENTS:
            t=time.perf_counter(); o=tr1(s); lat.append((time.perf_counter()-t)*1000); outs.append(o)
        print(f"[{ct:14s} beam={beam}] p50={statistics.median(lat):5.1f}ms  min={min(lat):4.1f}  VRAMΔ={v1-v0}MiB",flush=True)
        for s,o in zip(SENTS,outs): print(f"      {o}",flush=True)
        del tr; gc.collect()
        return outs
    outs={}
    for ct in ["int8_float16","float16","int8","int8_float32","float32"]:
        try: outs[ct]=run_ct(ct,1)
        except Exception as e: print(f"[{ct}] FAILED {e}")
    outs["beam4"]=run_ct("int8_float16",4)
    # quality: how often does each cheaper type differ from float32 beam1?
    ref=outs.get("float32")
    if ref:
        for ct,o in outs.items():
            diff=sum(1 for a,b in zip(o,ref) if a!=b)
            print(f"  diff vs float32/beam1: {ct:14s} -> {diff}/{len(ref)} sentences differ")
except Exception: traceback.print_exc()

# -------------------------------------------------- Parakeet int8
sec("Parakeet TDT 0.6B v3  int8  (VRAM/latency/accuracy)")
FP32={  # from the fp32 run (gpu_bench) for accuracy comparison
 "A_en01.wav":"Can you hear me okay?",
 "A_en02.wav":"I think we should push the release to Monday.",
 "A_en03.wav":"Could you share your screen so we can look at the dashboard?",
 "A_en04.wav":"The latency numbers look much better after the last deploy.",
 "A_en05.wav":"Sorry, you were on mute for a second.",
 "A_en06.wav":"Let's take this offline and sync tomorrow morning.",
 "A_en07.wav":"Who owns the database migration?",
 "A_en08.wav":"We still have three open tickets from the last sprint."}
try:
    import onnx_asr
    v0=vram()
    m=onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",quantization="int8",providers=["CUDAExecutionProvider"])
    m.recognize(SAMPLES[0]); time.sleep(0.2); v1=vram()
    lat=[]; agree=0; tot=0
    for w in SAMPLES:
        t=time.perf_counter(); txt=m.recognize(w); ms=(time.perf_counter()-t)*1000; lat.append(ms)
        b=os.path.basename(w); exp=FP32.get(b)
        tag=""
        if exp is not None:
            tot+=1; ok=(txt.strip()==exp.strip()); agree+=ok; tag="  OK" if ok else f"  != fp32: {exp!r}"
        print(f"  [{ms:7.1f} ms] {b} -> {txt!r}{tag}",flush=True)
    print(f"parakeet int8 latency ms: p50={statistics.median(lat):.1f} min={min(lat):.1f} max={max(lat):.1f}",flush=True)
    print(f"parakeet int8 VRAM load delta ~{v1-v0} MiB",flush=True)
    print(f"int8 vs fp32 exact-match: {agree}/{tot} English samples",flush=True)
    sec("Parakeet int8 — Spanish (es) samples, for the es->en path")
    for w in ES:
        t=time.perf_counter(); txt=m.recognize(w); ms=(time.perf_counter()-t)*1000
        print(f"  [{ms:7.1f} ms] {os.path.basename(w)} -> {txt!r}",flush=True)
    # ---- cold vs warm (after this model is resident) ----
    sec("cold-clock penalty: first recognize after 3s idle vs warm")
    import onnx_asr as _
    time.sleep(3)
    t=time.perf_counter(); m.recognize(SAMPLES[0]); cold=(time.perf_counter()-t)*1000
    warm=[]
    for _i in range(5):
        t=time.perf_counter(); m.recognize(SAMPLES[0]); warm.append((time.perf_counter()-t)*1000)
    print(f"cold (after 3s idle)= {cold:.1f} ms   warm p50= {statistics.median(warm):.1f} ms   penalty= {cold-statistics.median(warm):.1f} ms",flush=True)
except Exception: traceback.print_exc()
print("\nALL_DONE_MARKER",flush=True)

"""TTS con clonación de voz: Kyutai Pocket TTS, export ONNX (ottomate/pocket-tts-ONNX).

ONNX puro en CPU, sin PyTorch. Port del runtime de referencia del exportador
(github.com/alfredomariamilano/pocket-tts-onnx-export, dist/compare_onnx_node.js):

  voz:   wav (24 kHz) -> mimi_encoder -> latentes [1,F,1024] -> flow_lm_main (prompt)
         => estado KV de la voz. Se calcula UNA vez y se copia por frase.
  texto: tokens -> text_conditioner -> flow_lm_main (prompt)
  loop:  flow_lm_main(latente previo) -> conditioning + eos
         flow_lm_flow (Euler, lsd_steps) -> latente -> mimi_decoder -> 80 ms de audio

Modelos CC-BY-4.0 (Kyutai). Uso permitido solo con consentimiento de la persona
cuya voz se clona: acá es la voz del propio usuario.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

REPO = "ottomate/pocket-tts-ONNX"
SR = 24000
LATENT = 32
DIM = 1024
SAMPLES_PER_LATENT = 1920        # 80 ms a 24 kHz
MIMI_STEPS_PER_LATENT = 16
EOS_THRESHOLD = -4.0
MIN_FRAMES_BEFORE_EOS = 6       # constante del original (pocket_tts/models/tts_model.py)
MAX_TOKENS_PER_CHUNK = 50       # idem default_parameters.py
SILENCE_RMS = 0.01              # trozos iniciales más bajos que esto no se mandan
MAX_LEADING_SKIP = 6            # como mucho ~0,5 s de silencio inicial salteado
MAX_PAUSE_FRAMES = 4            # pausa interna máxima: 4 x 80 ms


def _sent_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in (x.strip() for x in parts) if p]


def _prepare(text: str) -> tuple[str, int]:
    t = re.sub(r"\s+", " ", text.strip())
    frames_after_eos = (3 if len(t.split()) <= 4 else 1) + 2
    t = t[0].upper() + t[1:]
    if t[-1].isalnum():
        t += "."
    if len(t.split()) < 5:
        t = " " * 8 + t  # el modelo rinde mejor con prompts no tan cortos (igual que la referencia)
    return t, frames_after_eos


class PocketTTS:
    def __init__(self, voice_wav: Path, quant: str = "int8", threads: int = 4,
                 temperature: float = 0.3, lsd_steps: int = 1, max_text_tokens: int = 160):
        root = Path(snapshot_download(REPO, allow_patterns=[
            "onnx/flow_lm_main*", "onnx/flow_lm_flow*", "onnx/mimi_decoder*",
            "onnx/mimi_encoder.onnx", "onnx/text_conditioner.onnx", "tokenizer*"]))
        sfx = "" if quant == "fp32" else f"_{quant}"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads   # matmuls chicos y secuenciales: más hilos empeora
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        cpu = ["CPUExecutionProvider"]

        def sess(name):
            return ort.InferenceSession(str(root / "onnx" / name), so, providers=cpu)

        self.text_cond = sess("text_conditioner.onnx")
        self.main = sess(f"flow_lm_main{sfx}.onnx")
        self.flow = sess(f"flow_lm_flow{sfx}.onnx")
        self.mimi = sess(f"mimi_decoder{sfx}.onnx")
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.temperature, self.lsd_steps = temperature, lsd_steps
        self.sr = SR
        self._rng = np.random.default_rng()

        self._main_states = [i for i in self.main.get_inputs() if i.name.startswith("state_")]
        self._mimi_states = [i for i in self.mimi.get_inputs() if i.name.startswith("state_")]
        kv = next(i for i in self.mimi.get_inputs() if i.name == "state_19")
        self.max_latents = kv.shape[2] // MIMI_STEPS_PER_LATENT  # 2048/16 = 128 (~10 s)
        self._mimi_init = {i.name: np.zeros(i.shape, dtype=_np_type(i.type)) for i in self._mimi_states}

        # --- voz: encoder + prompt, una sola vez
        enc = ort.InferenceSession(str(root / "onnx" / "mimi_encoder.onnx"), so, providers=cpu)
        audio = _load_mono(voice_wav, SR)
        latents = enc.run(None, {"audio": audio[None, None, :]})[0].astype(np.float32)
        del enc
        self.voice_frames = latents.shape[1]
        self._cache_len = self.voice_frames + max_text_tokens + self.max_latents + 8
        state = self._init_flow_state(self._cache_len)
        state = self._run_main(np.empty((1, 0, LATENT), np.float32), latents, state)[2]
        self._voice_state = state
        self.voice_seconds = len(audio) / SR

    # ------------------------------------------------------------------ estado
    def _init_flow_state(self, cache_len: int) -> dict[str, np.ndarray]:
        st = {}
        for i in self._main_states:
            if i.type == "tensor(float)":
                shape = [i.shape[0], i.shape[1], cache_len, i.shape[3], i.shape[4]]
                st[i.name] = np.full(shape, np.nan, dtype=np.float32)
            else:
                st[i.name] = np.zeros([d if isinstance(d, int) else 1 for d in i.shape], _np_type(i.type))
        return st

    def _run_main(self, seq, text_emb, state):
        out = self.main.run(None, {"sequence": seq, "text_embeddings": text_emb, **state})
        names = [o.name for o in self.main.get_outputs()]
        new_state = {n.replace("out_", "", 1): v for n, v in zip(names, out) if n.startswith("out_state_")}
        return out[0], out[1], new_state

    # ------------------------------------------------------------------ síntesis
    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Audio float32 @24 kHz en trozos de 80 ms, por bloques de oraciones.

        Oraciones cortas se agrupan (hasta MAX_TOKENS_PER_CHUNK, como el original):
        un prompt muy corto rinde peor y obliga a rellenar con espacios.
        """
        first_block = True
        quiet_run = 0
        for block in self._chunks(text):
            for i, audio in enumerate(self._stream_sentence(block)):
                silent = float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS
                # el modelo arranca con ~300 ms de silencio: no hacer esperar a nadie por eso
                if first_block and i < MAX_LEADING_SKIP and silent:
                    continue
                first_block = False
                # y a veces deja pausas de 1-2 s entre oraciones: se recortan a MAX_PAUSE_FRAMES
                quiet_run = quiet_run + 1 if silent else 0
                if quiet_run > MAX_PAUSE_FRAMES:
                    continue
                yield audio

    def _chunks(self, text: str) -> list[str]:
        blocks, cur, cur_tok = [], [], 0
        for sent in _sent_split(text):
            n = len(self.tok.encode(sent, add_special_tokens=False).ids)
            if cur and cur_tok + n > MAX_TOKENS_PER_CHUNK:
                blocks.append(" ".join(cur))
                cur, cur_tok = [], 0
            cur.append(sent)
            cur_tok += n
        if cur:
            blocks.append(" ".join(cur))
        return blocks

    def _stream_sentence(self, text: str) -> Iterator[np.ndarray]:
        prepared, frames_after_eos = _prepare(text)
        ids = self.tok.encode(prepared, add_special_tokens=False).ids
        ids = np.asarray([ids], dtype=np.int64)
        max_gen = min(math.ceil((ids.shape[1] / 3.0 + 2.0) * 12.5), self.max_latents)
        # OJO: el runtime JS de referencia usa max(8, tokens*1.5): con el relleno de espacios
        # obliga a seguir hablando después de terminar y el modelo REPITE la frase (medido).
        min_steps = MIN_FRAMES_BEFORE_EOS

        state = {k: v.copy() for k, v in self._voice_state.items()}
        emb = self.text_cond.run(None, {"token_ids": ids})[0].astype(np.float32)
        _, _, state = self._run_main(np.empty((1, 0, LATENT), np.float32), emb, state)

        mimi = {k: v.copy() for k, v in self._mimi_init.items()}
        mimi_names = [o.name for o in self.mimi.get_outputs()]
        prev = np.full((1, 1, LATENT), np.nan, dtype=np.float32)
        no_text = np.empty((1, 0, DIM), np.float32)
        std = math.sqrt(self.temperature)
        dt = 1.0 / self.lsd_steps
        eos_step = None
        for step in range(max_gen):
            cond, eos_logit, state = self._run_main(prev, no_text, state)
            x = (self._rng.standard_normal((1, LATENT)) * std).astype(np.float32)
            for k in range(self.lsd_steps):
                s = np.array([[k * dt]], np.float32)
                flow_dir = self.flow.run(None, {"c": cond.astype(np.float32), "s": s,
                                                "t": s + dt, "x": x})[0]
                x = x + flow_dir * dt
            if step >= min_steps and float(np.asarray(eos_logit).reshape(-1)[0]) > EOS_THRESHOLD and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            latent = x[:, None, :]
            out = self.mimi.run(None, {"latent": latent, **mimi})
            mimi = {n.replace("out_", "", 1): v for n, v in zip(mimi_names, out) if n.startswith("out_state_")}
            prev = latent
            yield out[0].reshape(-1).astype(np.float32)


def _np_type(t: str):
    return {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}[t]


def _load_mono(path: Path, sr: int) -> np.ndarray:
    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if s != sr:
        x = soxr.resample(x, s, sr).astype(np.float32)
    peak = float(np.abs(x).max()) or 1.0
    return (x / peak * 0.9).astype(np.float32)

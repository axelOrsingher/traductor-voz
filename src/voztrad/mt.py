"""Traducción automática: opus-mt (Marian) vía CTranslate2, int8 en CPU."""

from __future__ import annotations

import os
import re
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

# Modelos opus-mt ya convertidos a CTranslate2 (sin PyTorch).
REPOS = {
    ("en", "es"): "michaelfeil/ct2fast-opus-mt-en-es",
    ("es", "en"): "michaelfeil/ct2fast-opus-mt-es-en",
}


class Translator:
    def __init__(self, src: str, tgt: str, device: str = "cpu", compute_type: str = "int8",
                 threads: int | None = None):
        path = Path(snapshot_download(REPOS[(src, tgt)]))
        self.model = ctranslate2.Translator(
            str(path), device=device, compute_type=compute_type,
            inter_threads=1, intra_threads=threads or min(4, os.cpu_count() or 4))
        self.sp_src = spm.SentencePieceProcessor(model_file=str(path / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(path / "target.spm"))

    def translate(self, text: str) -> str:
        # opus-mt traduce UNA oración: con varias, se come las de más (medido: "That makes
        # sense. Go ahead." -> "Eso tiene sentido."). Se parte y se traduce en un solo batch.
        sents = split_sentences(text)
        if not sents:
            return ""
        batch = [self.sp_src.encode(s, out_type=str) + ["</s>"] for s in sents]
        res = self.model.translate_batch(batch, beam_size=1, max_decoding_length=256)
        return " ".join(self.sp_tgt.decode(r.hypotheses[0]) for r in res)


_SENT = re.compile(r"(?<=[.!?…])\s+(?=[¿¡\"'(\w])")


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT.split(text.strip())) if s]

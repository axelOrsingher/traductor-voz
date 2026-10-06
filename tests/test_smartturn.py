"""Smart Turn: las features numpy tienen que ser las de WhisperFeatureExtractor (transformers).

La referencia se generó una vez con transformers 5.18 en un venv aparte (sin torch) y se guarda
en tests/data para no depender de transformers en el proyecto.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voztrad.smartturn import SmartTurn

REF = Path(__file__).parent / "data" / "smartturn_features_ref.npz"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not REF.exists(), reason="sin referencia")
def test_features_identicas_a_transformers():
    st = SmartTurn()
    ref = np.load(REF)
    for k in ref.files:
        x, _ = sf.read(ROOT / k.replace("|", "/"), dtype="float32")
        assert np.abs(st.features(x) - ref[k]).max() < 1e-4, k

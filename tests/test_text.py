"""Procesamiento de texto: corte en oraciones para el MT, detección de idioma, voseo."""

import re

from voztrad.asr import _GLUED
from voztrad.langid import detect
from voztrad.mt import split_sentences
from voztrad.voseo import normalize


def test_split_no_pierde_oraciones():
    assert split_sentences("That makes sense. Go ahead.") == ["That makes sense.", "Go ahead."]
    assert split_sentences("Me alegro. ¿Qué es de tu día hoy?") == ["Me alegro.", "¿Qué es de tu día hoy?"]
    assert split_sentences("  ") == []
    assert split_sentences("3.5 años") == ["3.5 años"]  # el punto decimal no corta


def test_langid_frases_reales_de_la_llamada():
    assert detect("sí te escucho, tenemos de mujer igual") == "es"
    assert detect("Y tarda un poquito para mi, o sea, yo te hablo, te contesto.") == "es"
    assert detect("Can you share your screen so we can look at the dashboard?") == "en"
    assert detect("Il était chapel, t'as le pote voleur, tout ça.") == "fr"
    assert detect("Mm.") is None  # ambiguo: se traduce como venía


def test_voseo_y_lexico():
    assert normalize("Che, ¿vos tenés el informe?") == "Oye, ¿tú tienes el informe?"
    assert normalize("Dale, después te aviso") == "De acuerdo, después te aviso"  # "después" intacto
    assert normalize("el equipo de infra") == "el equipo de infraestructura"


def test_parakeet_puntuacion_pegada():
    assert _GLUED.sub(r"\1 \2", "Hola, amor.¿estás solo?") == "Hola, amor. ¿estás solo?"
    assert re.sub(_GLUED, r"\1 \2", "3.5") == "3.5"


def test_parakeet_signo_de_apertura():
    from voztrad.asr import _OPEN_SPACE
    assert _OPEN_SPACE.sub(r"\1", "Hola, ¿ cómo andan? ¡ Dale!") == "Hola, ¿cómo andan? ¡Dale!"

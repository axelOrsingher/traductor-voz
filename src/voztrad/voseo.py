"""Normalización rioplatense -> español neutro antes del MT (opcional).

Diccionario explícito de formas, no reglas genéricas sobre -ás/-és: esas
romperían "después", "además", "inglés", "atrás", "interés", etc.
"""

from __future__ import annotations

import re

_FORMS = {
    # pronombres / muletillas
    "vos": "tú", "con vos": "contigo", "a vos": "a ti", "para vos": "para ti",
    "che": "oye",
    # presente indicativo (con diptongo / irregulares)
    "sos": "eres", "tenés": "tienes", "querés": "quieres", "podés": "puedes",
    "sabés": "sabes", "venís": "vienes", "decís": "dices", "hacés": "haces",
    "pensás": "piensas", "entendés": "entiendes", "preferís": "prefieres",
    "volvés": "vuelves", "seguís": "sigues", "sentís": "sientes", "conocés": "conoces",
    "estás": "estás", "vas": "vas", "das": "das",
    "mandás": "mandas", "hablás": "hablas", "trabajás": "trabajas", "necesitás": "necesitas",
    "llegás": "llegas", "mirás": "miras", "creés": "crees", "ves": "ves", "leés": "lees",
    "escribís": "escribes", "vivís": "vives", "abrís": "abres", "subís": "subes",
    "chequeás": "chequeas", "arrancás": "arrancas", "avisás": "avisas", "pasás": "pasas",
    "tomás": "tomas", "usás": "usas", "probás": "pruebas", "contás": "cuentas",
    "encontrás": "encuentras", "recordás": "recuerdas", "acordás": "acuerdas",
    # imperativos
    "mirá": "mira", "fijate": "fíjate", "decime": "dime", "decile": "dile", "vení": "ven",
    "tené": "ten", "poné": "pon", "hacé": "haz", "andá": "ve", "salí": "sal", "pasame": "pásame",
    "mandame": "mándame", "avisame": "avísame", "contame": "cuéntame", "esperá": "espera",
    "escuchá": "escucha", "pará": "para", "probá": "prueba", "chequeá": "chequea",
    "mandá": "manda", "avisá": "avisa", "decí": "di", "pensá": "piensa", "acordate": "acuérdate",
    "fijense": "fíjense", "dejame": "déjame", "llamame": "llámame", "escribime": "escríbeme",
    "mostrame": "muéstrame", "compartí": "comparte", "compartime": "compárteme",
    # léxico rioplatense / jerga de oficina que opus-mt traduce mal (medido)
    "dale": "de acuerdo", "está bárbaro": "está muy bien", "bárbaro": "excelente",
    "re bien": "muy bien", "buenísimo": "muy bien", "joya": "perfecto",
    "infra": "infraestructura", "chequear": "verificar", "chequearlo": "verificarlo",
    "chequeá": "verifica", "fijarme": "verificar", "fijarse": "verificar", "fijate": "verifica",
    "laburo": "trabajo", "laburar": "trabajar", "posta": "en serio",
    "al toque": "enseguida", "un toque": "un poco", "ni idea": "no tengo idea",
}

# Frases multi-palabra primero; \b no sirve con vocales acentuadas al borde, se usa lookaround.
_KEYS = sorted(_FORMS, key=len, reverse=True)
_RE = re.compile(r"(?<![\wáéíóúñü])(" + "|".join(map(re.escape, _KEYS)) + r")(?![\wáéíóúñü])",
                 re.IGNORECASE)


def _sub(m: re.Match) -> str:
    src = m.group(0)
    out = _FORMS[src.lower()]
    return out[0].upper() + out[1:] if src[0].isupper() else out


def normalize(text: str) -> str:
    return _RE.sub(_sub, text)

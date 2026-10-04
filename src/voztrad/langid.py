"""Detección de idioma liviana sobre el TEXTO del ASR (sin modelos).

Parakeet no deja forzar el idioma y devuelve lo que oye: si del otro lado hablan
castellano, la dirección A recibe castellano (medido en la 1ª llamada real), y a
veces alucina en otro idioma (francés). Con stopwords alcanza para frases de >=3
palabras, que es lo que importa.
"""

from __future__ import annotations

import re

_SW = {
    "es": set("""de la que el en y a los se del las un por con no una su para es al lo como más
        pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde
        quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí
        antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual
        poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros
        vos che dale bueno sea tenés sos estás querés podés sabés mirá acá allá está son tiene
        hace fue era ser tengo voy vamos digo dije creo bien mal ahora después entonces osea re
        boludo igual nomás capaz""".split()),
    "en": set("""the of and to in is you that it he was for on are as with his they i at be this
        have from or one had by but not what all were we when your can said there use an each which
        she do how their if will up other about out many then them these so some her would make like
        him into time has look two more go see no way could my than first been call who its now
        find down did get come made may part okay yeah yes hey hi thanks thank please sorry just
        don't i'm it's that's let's we're you're can't think know want need should""".split()),
    "fr": set("""le les des est et un une du que qui dans pas pour sur avec ce cette sont mais ou
        nous vous ils elle était être avoir fait tout très bien je tu il on c'est t'as ça""".split()),
    "pt": set("""não uma com para os das dos isso você vocês muito também então está são tem foi
        ele ela eu nós mas como mais bem aqui obrigado""".split()),
    "it": set("""il della che è di per non una sono gli anche come questo sei molto grazie ciao
        perché però io tu lui lei noi voi""".split()),
}
_WORD = re.compile(r"[a-záéíóúñüàâçèêëîïôûœ']+")


def detect(text: str) -> str | None:
    """'es' | 'en' | 'fr' | 'pt' | 'it' | None (no hay señal suficiente)."""
    words = _WORD.findall(text.lower())
    if not words:
        return None
    scores = {lang: sum(w in sw for w in words) for lang, sw in _SW.items()}
    if any(c in text.lower() for c in "ñ¿¡"):
        scores["es"] += 2
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or (ranked[0] == ranked[1]):
        return None
    if len(words) <= 2 and ranked[0] < 2:
        return None  # "Mm.", "Okay." -> ambiguo
    return best

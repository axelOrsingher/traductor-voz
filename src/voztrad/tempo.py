"""Cambio de tempo sin cambiar el tono (WSOLA), en streaming.

Para qué: tu inglés clonado habla más lento que vos en castellano. En el modo intérprete,
en monólogos largos se acumula cola: en la llamada del 2026-10-05 el 18% de los
fragmentos le llegó al otro con >2 s de atraso. Si hay cola, el próximo fragmento se
reproduce a 1.15-1.3x.

WSOLA: ventanas Hann de 30 ms con 50% de solape; cada ventana de entrada se elige en
±10 ms alrededor de la posición ideal maximizando la correlación con la continuación
natural de la anterior (así no hay "fases" rotas ni efecto ardilla).
"""

from __future__ import annotations

import numpy as np


class Wsola:
    def __init__(self, rate: float, sr: int, frame_ms: float = 30.0, search_ms: float = 10.0):
        self.rate = rate
        self.n = int(sr * frame_ms / 1000) // 2 * 2
        self.hs = self.n // 2
        self.ha = self.hs * rate
        self.tol = int(sr * search_ms / 1000)
        self.win = np.hanning(self.n + 1)[:-1].astype(np.float32)  # periódica: suma 1 con 50% solape
        self.inp = np.zeros(self.tol, np.float32)  # margen para buscar hacia atrás en el 1er frame
        self.base = -self.tol                      # índice absoluto de inp[0]
        self.k = 0
        self.prev = 0                              # inicio absoluto del último segmento elegido
        self.acc = np.zeros(self.n, np.float32)

    def _seg(self, pos: int) -> np.ndarray:
        i = pos - self.base
        return self.inp[i:i + self.n]

    def push(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        self.inp = np.concatenate((self.inp, x.astype(np.float32)))
        if last:  # completar para poder cerrar las últimas ventanas
            self.inp = np.concatenate((self.inp, np.zeros(self.n + 2 * self.tol + int(self.ha) + 1, np.float32)))
        end = self.base + len(self.inp)
        out = []
        while True:
            target = int(round(self.k * self.ha))
            natural = self.prev + self.hs
            if max(target + self.tol, natural) + self.n > end:
                break
            if self.k == 0:
                pos = 0
            else:
                ref = self._seg(natural)
                lo = max(target - self.tol, self.base)
                region = self.inp[lo - self.base: target + self.tol + self.n - self.base]
                corr = np.correlate(region, ref, mode="valid")
                pos = lo + int(np.argmax(corr))
            self.acc += self._seg(pos) * self.win
            out.append(self.acc[:self.hs].copy())
            self.acc = np.concatenate((self.acc[self.hs:], np.zeros(self.hs, np.float32)))
            self.prev = pos
            self.k += 1
        # descartar entrada que ya no se va a usar (dejando margen de búsqueda)
        keep_from = min(int(round(self.k * self.ha)) - self.tol, self.prev + self.hs) - self.base
        if keep_from > 0:
            self.inp = self.inp[keep_from:]
            self.base += keep_from
        if last:
            out.append(self.acc[:self.hs].copy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def rate_for_backlog(queued_s: float) -> float:
    """Tempo según cuánto inglés hay en cola esperando sonar."""
    if queued_s < 0.8:
        return 1.0
    if queued_s < 2.5:
        return 1.15
    return 1.3

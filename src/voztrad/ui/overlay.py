"""Overlay de subtítulos (PySide6). Proceso aparte: habla con el motor SOLO por el
transporte JSON-lines (127.0.0.1:8765), nunca por llamadas directas.

Dos columnas — ELLOS (lo que dicen, traducido al castellano) y VOS (lo que dijiste,
como salió en inglés). La traducción va grande, el original chico debajo.
Angosto -> se apila en filas. Arrastrar desde la barra superior (con snap nativo de
Windows); redimensionar desde la esquina. Clic derecho en la barra: opciones.
"""

from __future__ import annotations

import json
import sys

from PySide6.QtCore import QByteArray, QPoint, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QPainter, QPainterPath
from PySide6.QtNetwork import QAbstractSocket, QTcpSocket
from PySide6.QtWidgets import (QApplication, QBoxLayout, QFrame, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QScrollArea, QSizeGrip, QVBoxLayout, QWidget)

# ------------------------------------------------------------------ tokens
BG = QColor(16, 18, 24)
BORDER = QColor(255, 255, 255, 22)
TEXT = "#F2F4F8"
MUTED = "#8B93A7"
FAINT = "#5A6172"
THEM = "#5CC8FF"   # ellos
ME = "#FFB454"     # vos
LIVE = "#FF5C5C"   # transmitiendo
OK = "#4ADE80"
WARN = "#FBBF24"
FONT = "Segoe UI Variable Text" if sys.platform == "win32" else "Inter"
MAX_CARDS = 14
STACK_BELOW_PX = 620


def _qss_btn(color=MUTED):
    return (f"QPushButton{{color:{color};background:transparent;border:none;padding:2px 6px;"
            f"font:600 11px '{FONT}';}}QPushButton:hover{{color:{TEXT};}}")


class Card(QFrame):
    """Una frase: traducción (grande) + original (chico) + latencia."""

    def __init__(self, ev: dict, accent: str, scale: float, show_src: bool, show_lat: bool):
        super().__init__()
        self.ev, self.accent = ev, accent
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.bar = QFrame()
        self.bar.setFixedWidth(3)
        outer.addWidget(self.bar)
        lay = QVBoxLayout()
        lay.setContentsMargins(12, 8, 10, 8)
        lay.setSpacing(2)
        outer.addLayout(lay, 1)
        self.main = QLabel(ev["tgt"])
        self.main.setWordWrap(True)
        self.main.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sub = QLabel(ev["src"])
        self.sub.setWordWrap(True)
        self.sub.setVisible(show_src and not ev.get("passthrough"))
        meta = []
        if ev.get("passthrough"):
            meta.append("sin traducir" if ev["dir"] == "A" else "dicho tal cual")
        total = (ev.get("lat") or {}).get("total")
        if show_lat and total is not None:
            meta.append(f"{total / 1000:.2f} s")
        self.meta = QLabel(" · ".join(meta))
        self.meta.setVisible(bool(meta))
        lay.addWidget(self.main)
        lay.addWidget(self.sub)
        lay.addWidget(self.meta)
        self.restyle(scale, age=0)

    def restyle(self, scale: float, age: int):
        # las frases viejas se atenúan, la última va a pleno
        a = max(0.38, 1.0 - 0.16 * age)
        main_c = QColor(TEXT)
        main_c.setAlphaF(a)
        sub_c = QColor(MUTED)
        sub_c.setAlphaF(a)
        self.main.setStyleSheet(f"color:{main_c.name(QColor.HexArgb)};"
                                f"font:{600 if age == 0 else 500} {round(19 * scale)}px '{FONT}';")
        self.sub.setStyleSheet(f"color:{sub_c.name(QColor.HexArgb)};font:{round(13 * scale)}px '{FONT}';")
        self.meta.setStyleSheet(f"color:{FAINT};font:{round(10.5 * scale)}px '{FONT}';")
        bar = QColor(self.accent)
        bar.setAlphaF(0.95 if age == 0 else 0.22)
        self.bar.setStyleSheet(f"background:{bar.name(QColor.HexArgb)};border:none;"
                               "border-top-left-radius:2px;border-bottom-left-radius:2px;")
        self.setStyleSheet(f"Card{{background:{'rgba(255,255,255,0.045)' if age == 0 else 'transparent'};"
                           "border:none;border-radius:6px;}")


class Column(QWidget):
    def __init__(self, title: str, hint: str, accent: str):
        super().__init__()
        self.accent = accent
        self.cards: list[Card] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        head = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setStyleSheet(f"color:{accent};font:700 11px '{FONT}';letter-spacing:1.5px;")
        self.hint = QLabel(hint)
        self.hint.setStyleSheet(f"color:{FAINT};font:11px '{FONT}';")
        self.live = QLabel("")
        self.live.setStyleSheet(f"color:{accent};font:600 11px '{FONT}';")
        head.addWidget(self.title)
        head.addWidget(self.hint)
        head.addStretch()
        head.addWidget(self.live)
        lay.addLayout(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;}"
            "QScrollBar:vertical{width:6px;background:transparent;margin:0;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.15);border-radius:3px;min-height:24px;}"
            "QScrollBar::add-line,QScrollBar::sub-line{height:0;}"
            "QScrollBar::add-page,QScrollBar::sub-page{background:none;}")
        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        self.list = QVBoxLayout(inner)
        self.list.setContentsMargins(0, 0, 4, 0)
        self.list.setSpacing(6)
        self.list.addStretch()  # las frases se apilan desde abajo, como subtítulos
        self.typing = QLabel("")
        self.typing.setStyleSheet(f"color:{accent};font:700 20px '{FONT}';padding-left:14px;")
        self.typing.hide()
        self.list.addWidget(self.typing)
        self.scroll.setWidget(inner)
        lay.addWidget(self.scroll, 1)

        self.empty = QLabel("")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(f"color:{FAINT};font:13px '{FONT}';padding:6px 2px;")
        self.list.insertWidget(1, self.empty)

        self._dots = 0
        self._timer = QTimer(self, interval=350, timeout=self._tick)

    def set_empty_text(self, text: str):
        self.empty.setText(text)
        self.empty.setVisible(not self.cards and bool(text))

    def add(self, ev: dict, scale: float, show_src: bool, show_lat: bool):
        bar = self.scroll.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 30
        card = Card(ev, self.accent, scale, show_src, show_lat)
        self.list.insertWidget(self.list.count() - 1, card)  # antes del indicador "…"
        self.cards.append(card)
        while len(self.cards) > MAX_CARDS:
            old = self.cards.pop(0)
            old.setParent(None)
            old.deleteLater()
        self.empty.hide()
        self.restyle(scale)
        if at_bottom:
            QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def restyle(self, scale: float):
        for age, c in enumerate(reversed(self.cards)):
            c.restyle(scale, age)

    def set_active(self, active: bool, label: str = "escuchando…", color: str | None = None):
        self.live.setText(f"● {label}" if active else "")
        self.live.setStyleSheet(f"color:{color or self.accent};font:700 11px '{FONT}';")
        if active:
            self._timer.start()
            self.typing.show()
            self._tick()
            bar = self.scroll.verticalScrollBar()
            QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))
        else:
            self._timer.stop()
            self.typing.hide()

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self.typing.setText("•" * max(1, self._dots))


class Overlay(QWidget):
    def __init__(self, host="127.0.0.1", port=8765):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("voztrad")
        self.settings = QSettings("voztrad", "overlay")
        self.scale = float(self.settings.value("scale", 1.0))
        self.show_src = self.settings.value("show_src", "true") == "true"
        self.show_lat = self.settings.value("show_lat", "true") == "true"
        self.bg_alpha = int(self.settings.value("bg_alpha", 236))
        self.state: dict = {}
        self.host, self.port = host, port

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 10)
        root.setSpacing(8)

        # --- barra superior (drag handle)
        self.bar = QWidget()
        self.bar.setCursor(Qt.SizeAllCursor)
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        self.dot = QLabel("●")
        self.status = QLabel("conectando…")
        self.status.setStyleSheet(f"color:{MUTED};font:11px '{FONT}';")
        self.mode_btn = QPushButton("")
        self.mode_btn.setCursor(Qt.PointingHandCursor)
        self.mode_btn.clicked.connect(self.toggle_mode)
        self.listen_btn = QPushButton("")
        self.listen_btn.setCursor(Qt.PointingHandCursor)
        self.listen_btn.setToolTip("Escuchar a los otros traducidos al castellano en tus auriculares\n"
                                   "(Teams baja de volumen mientras suena la traducción)")
        self.listen_btn.clicked.connect(self.toggle_listen)
        minus = QPushButton("A−", clicked=lambda: self.set_scale(self.scale - 0.1))
        plus = QPushButton("A+", clicked=lambda: self.set_scale(self.scale + 0.1))
        close = QPushButton("✕", clicked=self.close)
        for b in (minus, plus, close):
            b.setStyleSheet(_qss_btn())
            b.setCursor(Qt.PointingHandCursor)
        bl.addWidget(self.dot)
        bl.addWidget(self.status)
        bl.addStretch()
        bl.addWidget(self.listen_btn)
        bl.addWidget(self.mode_btn)
        bl.addSpacing(6)
        bl.addWidget(minus)
        bl.addWidget(plus)
        bl.addWidget(close)
        root.addWidget(self.bar)

        # --- columnas
        self.them = Column("ELLOS", "→ castellano", THEM)
        self.me = Column("VOS", "→ inglés", ME)
        self.cols = QBoxLayout(QBoxLayout.LeftToRight)
        self.cols.setSpacing(18)
        self.cols.addWidget(self.them, 1)
        self.cols.addWidget(self.me, 1)
        root.addLayout(self.cols, 1)

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip_row.addWidget(grip)
        root.addLayout(grip_row)

        self.setMinimumSize(340, 200)
        geo = self.settings.value("geometry")
        if isinstance(geo, QByteArray) and self.restoreGeometry(geo):
            pass
        else:
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.resize(min(980, scr.width() - 80), 300)
            self.move(scr.center().x() - self.width() // 2, scr.bottom() - self.height() - 40)

        self._set_connected(False)
        self._render_state()

        # --- transporte
        self.buf = b""
        self.sock = QTcpSocket(self)
        self.sock.connected.connect(self._on_connected)
        self.sock.disconnected.connect(lambda: self._set_connected(False))
        self.sock.readyRead.connect(self._on_ready)
        self.sock.errorOccurred.connect(lambda _e: self._set_connected(False))
        self._retry = QTimer(self, interval=1000, timeout=self._connect)
        self._retry.start()
        self._connect()

    # ------------------------------------------------------------ transporte
    def _connect(self):
        if self.sock.state() == QAbstractSocket.UnconnectedState:
            self.sock.connectToHost(self.host, self.port)

    def _on_connected(self):
        self._set_connected(True)
        self.send({"cmd": "state"})

    def send(self, cmd: dict):
        if self.sock.state() == QAbstractSocket.ConnectedState:
            self.sock.write((json.dumps(cmd) + "\n").encode())

    def _on_ready(self):
        self.buf += bytes(self.sock.readAll())
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.strip():
                try:
                    self.on_event(json.loads(line))
                except ValueError:
                    pass

    def on_event(self, ev: dict):
        t = ev.get("type")
        if t == "phrase":
            col = self.them if ev["dir"] == "A" else self.me
            col.set_active(False)
            col.add(ev, self.scale, self.show_src, self.show_lat)
        elif t == "voice":
            col = self.them if ev["dir"] == "A" else self.me
            if ev["dir"] == "B" and self.state.get("b_mode") == "ptt":
                return  # en PTT el indicador lo maneja el evento "ptt"
            col.set_active(ev["active"], "hablan…" if ev["dir"] == "A" else "hablás…")
        elif t == "ptt":
            self.state["ptt"] = ev["down"]
            self.me.set_active(ev["down"], "TRANSMITIENDO", LIVE)
            self._render_state()
        elif t == "state":
            if not ev.get("ready", True):
                self.status.setText("cargando modelos…")
                return
            self.state = ev
            self._render_state()
            self.status.setText(f"en vivo · {(ev.get('asr') or '').split('-')[0]}")
        elif t == "status":
            msg = ev.get("msg", "")
            if "escuchando" in msg:
                self.send({"cmd": "state"})
            elif not self.state:
                self.status.setText(msg)

    # ------------------------------------------------------------ estado UI
    def _set_connected(self, ok: bool):
        self.dot.setStyleSheet(f"color:{OK if ok else WARN};font:10px;")
        if not ok:
            self.state = {}
            self.status.setText("esperando al motor…")
            self._render_state()
        hint_a = "" if ok else "Acá aparece lo que dicen, traducido."
        self.them.set_empty_text(hint_a)
        self.me.set_empty_text("" if ok else "Acá aparece lo que decís, como sale en inglés.")

    def _render_state(self):
        mode = self.state.get("b_mode")
        key = self.state.get("ptt_key", "F8")
        if not mode:
            self.mode_btn.hide()
            self.listen_btn.hide()
            return
        self.mode_btn.show()
        self.listen_btn.setVisible(bool(self.state.get("listen_available")))
        on = bool(self.state.get("listen"))
        self.listen_btn.setText("🔊 ESCUCHAR ES" if on else "🔇 ESCUCHAR ES")
        self.listen_btn.setStyleSheet(_qss_btn(THEM if on else FAINT))
        label, hint = {
            "ptt": (f"PTT · {key}", f"→ inglés · mantené {key} para hablar"),
            "vad": ("MANOS LIBRES", "→ inglés · traduce cuando terminás la frase"),
            "interp": ("INTÉRPRETE", "→ inglés · va traduciendo en cada pausa"),
        }.get(mode, (mode, ""))
        self.mode_btn.setText(label)
        self.mode_btn.setToolTip("Cambiar modo: push-to-talk → manos libres → intérprete\n"
                                 "(manos libres / intérprete: usá auriculares)")
        self.mode_btn.setStyleSheet(_qss_btn(ME))
        self.me.hint.setText(hint)

    def toggle_listen(self):
        self.send({"cmd": "listen", "value": not self.state.get("listen")})

    def toggle_mode(self):
        order = ["ptt", "vad", "interp"]
        cur = self.state.get("b_mode", "ptt")
        self.send({"cmd": "b_mode", "value": order[(order.index(cur) + 1) % 3 if cur in order else 0]})

    def set_scale(self, s: float):
        self.scale = round(min(2.0, max(0.7, s)), 2)
        self.settings.setValue("scale", self.scale)
        self.them.restyle(self.scale)
        self.me.restyle(self.scale)

    # ------------------------------------------------------------ ventana
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        bg = QColor(BG)
        bg.setAlpha(self.bg_alpha)
        p.fillPath(path, bg)
        p.setPen(BORDER)
        p.drawPath(path)

    def resizeEvent(self, e):
        self.cols.setDirection(QBoxLayout.TopToBottom if self.width() < STACK_BELOW_PX
                               else QBoxLayout.LeftToRight)
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.bar.geometry().contains(e.position().toPoint()):
            self.windowHandle().startSystemMove()  # drag nativo: Aero Snap incluido
        elif e.button() == Qt.RightButton:
            self._menu(e.globalPosition().toPoint())

    def _menu(self, pos: QPoint):
        m = QMenu(self)
        for label, attr in (("Mostrar original", "show_src"), ("Mostrar latencia", "show_lat")):
            act = QAction(label, m, checkable=True, checked=getattr(self, attr))
            act.toggled.connect(lambda v, a=attr: self._toggle(a, v))
            m.addAction(act)
        op = m.addMenu("Opacidad del fondo")
        for pct in (100, 92, 80, 65):
            op.addAction(f"{pct}%", lambda p=pct: self._set_alpha(round(255 * p / 100)))
        m.addSeparator()
        m.addAction("Limpiar", self._clear)
        m.addAction("Cerrar", self.close)
        m.exec(pos)

    def _toggle(self, attr: str, v: bool):
        setattr(self, attr, v)
        self.settings.setValue(attr, "true" if v else "false")
        for col in (self.them, self.me):
            for c in col.cards:
                c.sub.setVisible(self.show_src and not c.ev.get("passthrough"))
                total = (c.ev.get("lat") or {}).get("total")
                c.meta.setVisible(bool(c.meta.text()) and (self.show_lat or c.ev.get("passthrough"))
                                  and total is not None or bool(c.ev.get("passthrough")))

    def _set_alpha(self, a: int):
        self.bg_alpha = a
        self.settings.setValue("bg_alpha", a)
        self.update()

    def _clear(self):
        for col in (self.them, self.me):
            for c in col.cards:
                c.setParent(None)
                c.deleteLater()
            col.cards.clear()

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)
        # Qt no cuenta las ventanas Qt.Tool para "última ventana cerrada": salir explícito,
        # si no, `voztrad app` deja el motor corriendo huérfano.
        QApplication.quit()


def main(host="127.0.0.1", port=8765) -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont(FONT, 10))
    w = Overlay(host, port)
    w.show()
    # Pruebas: VOZTRAD_UI_SNAPSHOT="ruta.png;segundos" guarda una captura y sale.
    snap = __import__("os").environ.get("VOZTRAD_UI_SNAPSHOT")
    if snap:
        path, secs = snap.rsplit(";", 1)
        QTimer.singleShot(int(float(secs) * 1000), lambda: (w.grab().save(path), w.close()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

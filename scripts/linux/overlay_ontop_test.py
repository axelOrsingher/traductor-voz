#!/usr/bin/env python3
"""Replicate the voice-bridge overlay window flags and report, for the current
QT_QPA_PLATFORM, the actual platform plugin Qt loaded, whether the frameless
always-on-top window maps, and its geometry. Stays alive SECONDS then quits so
an external prober (wmctrl/xprop) can inspect it."""
import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
app = QApplication(sys.argv)

w = QWidget(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus)
w.setAttribute(Qt.WA_TranslucentBackground)
w.setAttribute(Qt.WA_ShowWithoutActivating)
w.setWindowTitle("VOZTRAD_OVERLAY_TEST")
lay = QVBoxLayout(w); lay.addWidget(QLabel("voice-bridge overlay on-top test"))
w.setGeometry(200, 200, 420, 90)
w.show(); w.raise_()

def report():
    print("QT_QPA_PLATFORM env     :", sys.argv[0] and __import__("os").environ.get("QT_QPA_PLATFORM"))
    print("platformName (loaded)   :", QGuiApplication.platformName())
    print("window isVisible        :", w.isVisible())
    print("window winId            :", int(w.winId()))
    print("frameGeometry           :", w.frameGeometry().getRect())
    print("flags StaysOnTop set    :", bool(w.windowFlags() & Qt.WindowStaysOnTopHint))
    print("READY_FOR_PROBE", flush=True)

QTimer.singleShot(400, report)
QTimer.singleShot(int(SECONDS * 1000), app.quit)
sys.exit(app.exec())

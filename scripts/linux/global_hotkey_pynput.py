#!/usr/bin/env python3
"""Try to read F8 globally via pynput. Under Wayland pynput falls back to the
Xlib backend, which only sees events delivered through XWayland (focused X11
windows) and generally cannot capture a global hotkey. Expected: no capture."""
import sys, time
from pynput import keyboard

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
print(f"pynput backend module: {keyboard.Listener.__module__}")
count = 0

def on_press(key):
    global count
    if key == keyboard.Key.f8:
        count += 1
        print(f"[{time.strftime('%H:%M:%S')}] F8 seen by pynput (#{count})", flush=True)

print(f"pynput listening {TIMEOUT:.0f}s ...")
print(">>> PRESS F8 a few times now <<<", flush=True)
listener = keyboard.Listener(on_press=on_press)
listener.start()
time.sleep(TIMEOUT)
listener.stop()
print(f"RESULT: pynput captured {count} F8 press(es) -> {'OK' if count else 'NONE (expected under Wayland)'}")

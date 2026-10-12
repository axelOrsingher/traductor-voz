#!/usr/bin/env python3
"""Read F8 globally via python-evdev (needs read access to /dev/input/event*,
i.e. the user in the 'input' group). Listens for TIMEOUT seconds across all
keyboards and reports every F8 press. This works regardless of Wayland/X11
because it reads raw kernel input events."""
import sys, time, select
from evdev import InputDevice, list_devices, ecodes

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

kbds = []
for path in list_devices():
    try:
        d = InputDevice(path)
        caps = d.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_F8 in caps:
            kbds.append(d)
    except Exception as e:
        print(f"skip {path}: {e}")

if not kbds:
    print("NO readable keyboard devices with KEY_F8 (missing 'input' group?)")
    sys.exit(2)

print(f"evdev listening {TIMEOUT:.0f}s on: {[d.path+' '+d.name for d in kbds]}")
print(">>> PRESS F8 a few times now <<<", flush=True)
fdmap = {d.fd: d for d in kbds}
end = time.time() + TIMEOUT
count = 0
while time.time() < end:
    r, _, _ = select.select(fdmap, [], [], 0.5)
    for fd in r:
        for ev in fdmap[fd].read():
            if ev.type == ecodes.EV_KEY and ev.code == ecodes.KEY_F8 and ev.value == 1:
                count += 1
                print(f"[{time.strftime('%H:%M:%S')}] F8 DOWN detected (#{count}) on {fdmap[fd].name}", flush=True)
print(f"RESULT: evdev captured {count} F8 press(es) -> {'OK' if count else 'NONE'}")

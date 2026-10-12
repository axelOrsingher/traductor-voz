"""Global push-to-talk key. Windows polls GetAsyncKeyState; Linux reads evdev."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from .ptt_win import PushToTalkKey  # noqa: F401
else:
    from .ptt_linux import PushToTalkKey  # noqa: F401

"""Lower the call app's volume while the Spanish translation plays. Windows uses
pycaw/WASAPI sessions; Linux uses pactl sink-input volumes."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from .ducker_win import Ducker  # noqa: F401
else:
    from .ducker_linux import Ducker  # noqa: F401

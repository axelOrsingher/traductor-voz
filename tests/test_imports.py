"""Every module must import: a syntax error in the UI isn't caught by any other test
(it happened: the shortcut opened nothing and nobody noticed until they launched it).

Modules for other platforms can't be imported here (their native deps are missing), so they
are only compiled: that still catches syntax errors on every platform."""

import importlib
import pkgutil
import py_compile
import sys
from pathlib import Path

import voztrad

_PLATFORM_SUFFIX = {"win32": "_win", "linux": "_linux", "darwin": "_mac"}
_SUFFIXES = tuple(_PLATFORM_SUFFIX.values())
_WINDOWS_ONLY = {"voztrad.procloop"}  # WASAPI process loopback via COM


def _native_here(name: str) -> bool:
    if name in _WINDOWS_ONLY:
        return sys.platform == "win32"
    if name.endswith(_SUFFIXES):
        mine = _PLATFORM_SUFFIX.get(sys.platform)
        return mine is not None and name.endswith(mine)
    return True


def test_all_modules_import():
    for m in pkgutil.walk_packages(voztrad.__path__, "voztrad."):
        if m.name == "voztrad.__main__":
            continue
        if _native_here(m.name):
            importlib.import_module(m.name)


def test_all_modules_compile():
    root = Path(voztrad.__file__).parent
    for f in root.rglob("*.py"):
        py_compile.compile(str(f), doraise=True)


def test_config_serializable_to_log():
    import json

    from voztrad.config import Config
    from voztrad.engine import _cfg_dict
    json.dumps(_cfg_dict(Config()))  # any new Path in Config used to break startup

"""Compatibility shim so the project can be imported directly from the repo root.

When the package is not installed in editable mode, running commands from the
repository root should still allow imports like ``crash_kedro.api.app``. We do
this by extending the package search path to include ``src/crash_kedro``.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

_package_path = list(extend_path(list(globals().get("__path__", [])), __name__))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _PROJECT_ROOT / "src" / "crash_kedro"
if _SRC_PACKAGE.is_dir():
    src_path = str(_SRC_PACKAGE)
    if src_path not in _package_path:
        _package_path.append(src_path)

__path__ = _package_path
__version__ = "0.1"



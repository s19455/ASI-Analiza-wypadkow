"""Ensure the local `src/` directory is importable when running from the project root.

This keeps commands like `uvicorn crash_kedro.api.app:app --reload` working even
before the package is installed in editable mode. Python imports `sitecustomize`
automatically during startup, so this is a lightweight compatibility shim.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"

if _SRC_DIR.is_dir():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


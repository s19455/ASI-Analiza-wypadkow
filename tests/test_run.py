"""Testy rejestru pipeline'ow."""

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crash_kedro.pipeline_registry import register_pipelines  # noqa: E402


def test_register_pipelines_has_defaults():
    """Rejestr powinien zawierac domyslne pipeline'y."""
    pipelines = register_pipelines()

    assert "__default__" in pipelines
    assert "data_processing" in pipelines
    assert "modeling" in pipelines


def test_default_pipeline_has_nodes():
    """Domyslny pipeline nie powinien byc pusty."""
    pipelines = register_pipelines()

    assert len(pipelines["__default__"].nodes) > 0

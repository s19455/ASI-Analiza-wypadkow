"""Tests for encoder utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crash_kedro.utils.encoders import (  # noqa: E402
    fit_encoders,
    load_encoders,
    save_encoders,
    transform_with_encoders,
)


def test_fit_encoders_creates_encoders_for_text_columns() -> None:
    """fit_encoders should create mappings for object columns only."""

    df = pd.DataFrame(
        {
            "city": ["Warszawa", "Krakow", "Warszawa"],
            "segment": ["A", "B", "A"],
            "age": [21, 34, 29],
            "ignored_text": ["x", "y", "z"],
        }
    )

    encoders = fit_encoders(df, ignore_columns=["ignored_text"])

    assert set(encoders) == {"city", "segment"}
    assert encoders["city"] == {"Warszawa": 0, "Krakow": 1}
    assert encoders["segment"] == {"A": 0, "B": 1}


def test_transform_with_encoders_handles_known_and_unknown_categories() -> None:
    """Known values should be encoded and unseen values should map to -1."""

    train_df = pd.DataFrame(
        {
            "city": ["Warszawa", "Krakow", "Gdansk"],
            "age": [10, 20, 30],
        }
    )
    test_df = pd.DataFrame(
        {
            "city": ["Warszawa", "Poznan", None],
            "age": [40, 50, 60],
        }
    )

    encoders = fit_encoders(train_df)
    transformed_df = transform_with_encoders(test_df, encoders)

    assert transformed_df["city"].tolist() == [0, -1, -1]
    assert transformed_df["age"].tolist() == [40, 50, 60]
    assert test_df["city"].tolist() == ["Warszawa", "Poznan", None]


def test_save_and_load_encoders_roundtrip(tmp_path: Path) -> None:
    """Saved encoders should load back to the same structure."""

    df = pd.DataFrame(
        {
            "city": ["Warszawa", "Krakow", "Warszawa"],
            "age": [1, 2, 3],
        }
    )
    encoders = fit_encoders(df)
    file_path = tmp_path / "encoders.pkl"

    save_encoders(encoders, file_path)
    loaded_encoders = load_encoders(file_path)

    assert loaded_encoders == encoders


"""Tests for the FastAPI prediction helpers."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crash_kedro.api.app import (  # noqa: E402
    CrashInput,
    build_prediction_frame,
    health_status,
)


def test_build_prediction_frame_uses_training_column_order() -> None:
    """Prediction frame should follow the training feature order."""

    frame = build_prediction_frame(CrashInput())

    expected_columns = [
        "Route Type",
        "Road Name",
        "Cross-Street Type",
        "Cross-Street Name",
        "Municipality",
        "Collision Type",
        "Weather",
        "Surface Condition",
        "Light",
        "Traffic Control",
        "Driver Substance Abuse",
        "Driver At Fault",
        "Circumstance",
        "Driver Distracted By",
        "Drivers License State",
        "Vehicle Damage Extent",
        "Vehicle First Impact Location",
        "Vehicle Second Impact Location",
        "Vehicle Body Type",
        "Vehicle Movement",
        "Vehicle Continuing Dir",
        "Vehicle Going Dir",
        "Speed Limit",
        "Driverless Vehicle",
        "Parked Vehicle",
        "Vehicle Year",
        "Vehicle Make",
        "Vehicle Model",
        "Equipment Problems",
        "Latitude",
        "Longitude",
        "crash_hour",
        "crash_dayofweek",
        "crash_month",
        "crash_year",
        "is_night",
        "is_bad_weather",
        "is_wet_surface",
        "vehicle_age",
    ]

    assert list(frame.columns) == expected_columns
    assert pd.api.types.is_integer_dtype(frame["Weather"])
    assert pd.api.types.is_integer_dtype(frame["Light"])
    assert frame.loc[0, "Speed Limit"] == 35
    assert frame.loc[0, "crash_year"] == dt.datetime.now().year


def test_health_status_reports_encoder_availability() -> None:
    """Health payload should report model and encoder status."""

    status = health_status()

    # encoders.pkl i model siedza w data/ (gitignore), wiec po swiezym klonie
    # moze ich nie byc - sprawdzamy tylko, ze health zwraca poprawna strukture
    assert "model_loaded" in status
    assert "encoders_loaded" in status
    assert isinstance(status["encoders_loaded"], bool)


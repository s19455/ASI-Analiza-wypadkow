"""FastAPI - serwowanie modelu predykcji stopnia obrazen."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from crash_kedro.utils.encoders import (
    UNKNOWN_CATEGORY_CODE,
    load_encoders,
    transform_with_encoders,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = PROJECT_ROOT / "logs"
ENCODERS_PATH = PROJECT_ROOT / "data" / "06_models" / "encoders.pkl"

TRAINING_FEATURE_ORDER = [
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

MODEL_CANDIDATES = [
    Path(os.getenv("MODEL_PATH", "")) if os.getenv("MODEL_PATH") else None,
    PROJECT_ROOT / "data" / "06_models" / "model.pkl",
    PROJECT_ROOT / "data" / "06_models" / "tuned_model.pkl",
    PROJECT_ROOT / "data" / "06_models" / "best_comparison_model.pkl",
    PROJECT_ROOT / "data" / "06_models" / "grid_random_model.pkl",
]

_MODEL_CACHE: dict[str, Any | None] = {"path": None, "obj": None}
_ENCODERS_CACHE: dict[str, Any | None] = {"path": None, "obj": None}


app = FastAPI(
    title="Crash Severity Prediction API",
    description="Predykcja stopnia obrazen w wypadkach drogowych",
    version="1.0.0",
)


class CrashInput(BaseModel):
    """Dane wejsciowe przyjmowane przez endpoint predykcji."""

    model_config = ConfigDict(extra="forbid")

    weather: str = "CLEAR"
    light: str = "DAYLIGHT"
    collision_type: str = "SAME DIR REAR END"
    surface_condition: str = "DRY"
    traffic_control: str = "NO CONTROLS"
    driver_substance_abuse: str = "NONE DETECTED"
    driver_distracted_by: str = "NOT DISTRACTED"
    vehicle_body_type: str = "PASSENGER CAR"
    vehicle_damage_extent: str = "FUNCTIONAL"
    vehicle_movement: str = "MOVING CONSTANT"
    speed_limit: int = 35
    driver_at_fault: str = "Yes"
    driverless_vehicle: str = "No"
    parked_vehicle: str = "No"
    vehicle_year: int = 2020
    crash_hour: int = 12
    crash_dayofweek: int = 2
    crash_month: int = 6
    crash_year: int = Field(default_factory=lambda: dt.datetime.now().year)


class PredictionOutput(BaseModel):
    """Odpowiedz zwracana przez endpoint predykcji."""

    severity: str
    probabilities: dict[str, float]
    timestamp: str


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as input_stream:
        return pickle.load(input_stream)


def _unique_paths(paths: list[Path | None]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in paths:
        if candidate is None:
            continue
        key = str(candidate.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate.expanduser())
    return unique


def get_model_candidates() -> list[Path]:
    """Zwraca mozliwe lokalizacje modelu w kolejnosci priorytetu."""

    return _unique_paths(MODEL_CANDIDATES)


def get_model() -> Any | None:
    """Wczytuje i cache'uje pierwszy dostepny model."""

    cached_path = _MODEL_CACHE["path"]
    cached_model = _MODEL_CACHE["obj"]
    if isinstance(cached_path, Path) and cached_path.is_file() and cached_model is not None:
        return cached_model

    for candidate_path in get_model_candidates():
        if not candidate_path.is_file():
            continue

        try:
            loaded_model = _load_pickle(candidate_path)
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
            LOGGER.warning("Nie udalo sie wczytac modelu z %s: %s", candidate_path, exc)
            continue

        _MODEL_CACHE["path"] = candidate_path
        _MODEL_CACHE["obj"] = loaded_model
        return loaded_model

    _MODEL_CACHE["path"] = None
    _MODEL_CACHE["obj"] = None
    return None


def get_encoders() -> dict[str, object] | None:
    """Wczytuje i cache'uje enkodery, jesli plik istnieje."""

    cached_path = _ENCODERS_CACHE["path"]
    cached_encoders = _ENCODERS_CACHE["obj"]
    if (
        isinstance(cached_path, Path)
        and cached_path.is_file()
        and isinstance(cached_encoders, dict)
    ):
        return cached_encoders

    if not ENCODERS_PATH.is_file():
        _ENCODERS_CACHE["path"] = None
        _ENCODERS_CACHE["obj"] = None
        return None

    try:
        loaded_encoders = load_encoders(ENCODERS_PATH)
    except ValueError as exc:
        LOGGER.warning("Nie udalo sie wczytac encoderow: %s", exc)
        _ENCODERS_CACHE["path"] = None
        _ENCODERS_CACHE["obj"] = None
        return None

    _ENCODERS_CACHE["path"] = ENCODERS_PATH
    _ENCODERS_CACHE["obj"] = loaded_encoders
    return loaded_encoders


def build_prediction_frame(input_data: CrashInput) -> pd.DataFrame:
    """Buduje ramke cech zgodna z pipeline'em treningowym."""

    current_year = int(input_data.crash_year)
    feature_row: dict[str, Any] = {column: None for column in TRAINING_FEATURE_ORDER}
    feature_row.update(
        {
            "Weather": input_data.weather,
            "Light": input_data.light,
            "Collision Type": input_data.collision_type,
            "Surface Condition": input_data.surface_condition,
            "Traffic Control": input_data.traffic_control,
            "Driver Substance Abuse": input_data.driver_substance_abuse,
            "Driver Distracted By": input_data.driver_distracted_by,
            "Vehicle Body Type": input_data.vehicle_body_type,
            "Vehicle Damage Extent": input_data.vehicle_damage_extent,
            "Vehicle Movement": input_data.vehicle_movement,
            "Speed Limit": int(input_data.speed_limit),
            "Driver At Fault": input_data.driver_at_fault,
            "Driverless Vehicle": input_data.driverless_vehicle,
            "Parked Vehicle": input_data.parked_vehicle,
            "Vehicle Year": int(input_data.vehicle_year),
            "crash_hour": int(input_data.crash_hour),
            "crash_dayofweek": int(input_data.crash_dayofweek),
            "crash_month": int(input_data.crash_month),
            "crash_year": current_year,
            "is_night": int("DARK" in input_data.light.upper()),
            "is_bad_weather": int(input_data.weather not in {"CLEAR", "CLOUDY"}),
            "is_wet_surface": int(input_data.surface_condition != "DRY"),
            "vehicle_age": max(current_year - int(input_data.vehicle_year), 0),
            "Latitude": 0.0,
            "Longitude": 0.0,
        }
    )

    frame = pd.DataFrame([feature_row], columns=TRAINING_FEATURE_ORDER)
    encoders = get_encoders()
    if encoders is not None:
        frame = transform_with_encoders(frame, encoders)
    else:
        for column_name in frame.select_dtypes(include=["object"]).columns:
            frame[column_name] = UNKNOWN_CATEGORY_CODE

    return frame.loc[:, TRAINING_FEATURE_ORDER]


def _extract_probabilities(model: Any, features: pd.DataFrame) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(features)
    if hasattr(probabilities, "iloc"):
        probability_row = probabilities.iloc[0].tolist()
    else:
        probability_row = list(probabilities[0])

    class_labels = getattr(model, "classes_", range(len(probability_row)))
    return {
        str(label): round(float(probability), 4)
        for label, probability in zip(class_labels, probability_row, strict=False)
    }


def _append_prediction_log(
    input_data: CrashInput,
    prediction: str,
    probabilities: dict[str, float],
    timestamp: str,
) -> None:
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": timestamp,
        "input": input_data.model_dump(),
        "prediction": prediction,
        "probabilities": probabilities,
    }

    try:
        with (LOG_PATH / "predictions.jsonl").open("a", encoding="utf-8") as output_stream:
            output_stream.write(json.dumps(log_entry) + "\n")
    except OSError as exc:  # pragma: no cover - logging should not break the API
        LOGGER.warning("Nie udalo sie zapisac logu predykcji: %s", exc)


def predict_crash_severity(input_data: CrashInput) -> PredictionOutput:
    """Uruchamia predykcje modelu i zwraca ustrukturyzowana odpowiedz."""

    timestamp = dt.datetime.now().isoformat()
    model = get_model()
    features = build_prediction_frame(input_data)

    if model is None:
        output = PredictionOutput(
            severity="UNKNOWN",
            probabilities={},
            timestamp=timestamp,
        )
        _append_prediction_log(input_data, output.severity, output.probabilities, timestamp)
        return output

    try:
        prediction = model.predict(features)[0]
        probabilities = _extract_probabilities(model, features)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        LOGGER.warning("Blad predykcji: %s", exc)
        prediction = "UNKNOWN"
        probabilities = {}

    output = PredictionOutput(
        severity=str(prediction),
        probabilities=probabilities,
        timestamp=timestamp,
    )
    _append_prediction_log(input_data, output.severity, output.probabilities, timestamp)
    return output


def health_status() -> dict[str, Any]:
    """Zwraca prosty status gotowosci API."""

    model = get_model()
    encoders_loaded = get_encoders() is not None
    model_path = _MODEL_CACHE["path"]

    return {
        "status": "ok" if model is not None else "no model loaded",
        "model_loaded": model is not None,
        "encoders_loaded": encoders_loaded,
        "model_path": str(model_path) if isinstance(model_path, Path) else None,
    }


def get_recent_predictions(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Czyta ostatnie predykcje z pliku logu JSONL."""

    log_file = LOG_PATH / "predictions.jsonl"
    if limit <= 0 or not log_file.exists():
        return {"predictions": []}

    try:
        lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:  # pragma: no cover - log access should not fail the API
        LOGGER.warning("Nie udalo sie odczytac logu predykcji: %s", exc)
        return {"predictions": []}

    recent_entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            recent_entries.append(json.loads(line))
        except json.JSONDecodeError:
            LOGGER.warning("Pominięto uszkodzony wpis w logu predykcji.")

    return {"predictions": recent_entries}


@app.get("/health")
def health() -> dict[str, Any]:
    """Endpoint sprawdzajacy gotowosc aplikacji."""

    return health_status()


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: CrashInput) -> PredictionOutput:
    """Przewiduje stopien obrazen dla jednego zdarzenia."""

    return predict_crash_severity(input_data)


@app.get("/predictions/recent")
def recent_predictions(n: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Zwraca ostatnie zapisane predykcje."""

    return get_recent_predictions(limit=n)

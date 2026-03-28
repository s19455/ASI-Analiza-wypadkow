"""FastAPI - serwowanie modelu predykcji stopnia obrazen."""

import datetime
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Crash Severity Prediction API",
    description="Predykcja stopnia obrazen w wypadkach drogowych",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).parent.parent.parent.parent / "data" / "06_models" / "model.pkl"
LOG_PATH = Path(__file__).parent.parent.parent.parent / "logs"


class CrashInput(BaseModel):
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


class PredictionOutput(BaseModel):
    severity: str
    probabilities: dict
    timestamp: str
    latency_ms: float


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    total_latency_ms: float
    avg_latency_ms: float
    n_predictions: int


def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


_model_cache: dict = {}


def get_model():
    if "model" not in _model_cache:
        _model_cache["model"] = load_model()
    return _model_cache["model"]


def _prepare_features(input_data: CrashInput) -> pd.DataFrame:
    features = pd.DataFrame([input_data.model_dump()])
    for col in features.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
    features["is_night"] = 1 if "DARK" in input_data.light.upper() else 0
    features["is_bad_weather"] = 0 if input_data.weather in ["CLEAR", "CLOUDY"] else 1
    features["is_wet_surface"] = 0 if input_data.surface_condition == "DRY" else 1
    features["vehicle_age"] = 2026 - input_data.vehicle_year
    return features


def _predict_one(model, input_data: CrashInput) -> PredictionOutput:
    start = time.perf_counter()
    features = _prepare_features(input_data)

    try:
        prediction = model.predict(features)[0]
        probas = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            for cls, p in zip(model.classes_, proba, strict=False):
                probas[str(cls)] = round(float(p), 4)
    except Exception:
        prediction = "UNKNOWN"
        probas = {}

    latency_ms = (time.perf_counter() - start) * 1000
    timestamp = datetime.datetime.now().isoformat()

    log_prediction(input_data.model_dump(), prediction, probas, timestamp, latency_ms)

    return PredictionOutput(
        severity=str(prediction),
        probabilities=probas,
        timestamp=timestamp,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "ok" if model is not None else "no model loaded",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: CrashInput):
    """Real-time prediction (single sample)."""
    model = get_model()
    if model is None:
        return PredictionOutput(
            severity="UNKNOWN", probabilities={},
            timestamp=datetime.datetime.now().isoformat(), latency_ms=0.0,
        )
    return _predict_one(model, input_data)


@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(inputs: list[CrashInput]):
    """Batch prediction - przyjmuje liste wypadkow, zwraca liste predykcji.
    Optymalizacja: efektywniejsze niz wielokrotne /predict (jeden HTTP request).
    """
    model = get_model()
    start = time.perf_counter()

    predictions = [_predict_one(model, inp) for inp in inputs] if model else []
    total_latency = (time.perf_counter() - start) * 1000
    avg_latency = total_latency / len(inputs) if inputs else 0.0

    return BatchPredictionOutput(
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        avg_latency_ms=round(avg_latency, 2),
        n_predictions=len(predictions),
    )


def log_prediction(input_data, prediction, probabilities, timestamp, latency_ms=0.0):
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": timestamp,
        "input": input_data,
        "prediction": str(prediction),
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 2),
    }
    with open(LOG_PATH / "predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/predictions/recent")
def recent_predictions(n: int = 10):
    log_file = LOG_PATH / "predictions.jsonl"
    if not log_file.exists():
        return {"predictions": []}
    lines = log_file.read_text().strip().split("\n")
    recent = [json.loads(line) for line in lines[-n:]]
    return {"predictions": recent}


@app.get("/predictions/stats")
def predictions_stats():
    """Statystyki latency - ile predykcji, srednia, p50, p95, p99."""
    log_file = LOG_PATH / "predictions.jsonl"
    if not log_file.exists():
        return {"n": 0}
    lines = log_file.read_text().strip().split("\n")
    latencies = [json.loads(line).get("latency_ms", 0) for line in lines if line]
    if not latencies:
        return {"n": 0}
    latencies.sort()
    n = len(latencies)
    return {
        "n": n,
        "avg_ms": round(sum(latencies) / n, 2),
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[int(n * 0.95)], 2),
        "p99_ms": round(latencies[int(n * 0.99)], 2),
        "max_ms": round(max(latencies), 2),
    }

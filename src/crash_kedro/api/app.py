"""FastAPI - serwowanie modelu predykcji stopnia obrazen."""

import datetime
import json
import pickle
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


# cache modelu zeby nie wczytywal sie przy kazdym requeście
_model = {"obj": None}


def get_model():
    if _model["obj"] is None and MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model["obj"] = pickle.load(f)
    return _model["obj"]


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "ok" if model is not None else "no model loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: CrashInput):
    model = get_model()
    timestamp = datetime.datetime.now().isoformat()

    if model is None:
        return PredictionOutput(severity="UNKNOWN", probabilities={}, timestamp=timestamp)

    # przygotowanie cech - musi byc spojne z preprocessing'iem z pipeline'u
    features = pd.DataFrame([input_data.model_dump()])
    for col in features.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))

    features["is_night"] = 1 if "DARK" in input_data.light.upper() else 0
    features["is_bad_weather"] = 0 if input_data.weather in ["CLEAR", "CLOUDY"] else 1
    features["is_wet_surface"] = 0 if input_data.surface_condition == "DRY" else 1
    features["vehicle_age"] = 2026 - input_data.vehicle_year

    try:
        prediction = model.predict(features)[0]
        probas = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            for cls, p in zip(model.classes_, proba, strict=False):
                probas[str(cls)] = round(float(p), 4)
    except Exception as e:
        # TODO: lepiej obsluzyc bledy - na razie wystarczy
        print(f"Blad predykcji: {e}")
        prediction = "UNKNOWN"
        probas = {}

    # logowanie predykcji
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": timestamp,
        "input": input_data.model_dump(),
        "prediction": str(prediction),
        "probabilities": probas,
    }
    with open(LOG_PATH / "predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return PredictionOutput(severity=str(prediction), probabilities=probas, timestamp=timestamp)


@app.get("/predictions/recent")
def recent_predictions(n: int = 10):
    log_file = LOG_PATH / "predictions.jsonl"
    if not log_file.exists():
        return {"predictions": []}
    lines = log_file.read_text().strip().split("\n")
    return {"predictions": [json.loads(line) for line in lines[-n:]]}

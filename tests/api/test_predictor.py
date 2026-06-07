"""Tests for the prediction helpers used by the Streamlit app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crash_kedro.api import predictor  # noqa: E402


class _FakeHeaders:
    def get_content_charset(self, default: str = "utf-8") -> str:
        return default


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.headers = _FakeHeaders()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_build_prediction_payload_coerces_and_fills_missing_values() -> None:
    """Payload builder should coerce data types and use defaults when needed."""

    payload = predictor.build_prediction_payload(
        {
            "weather": " rain ",
            "light": None,
            "collision_type": "HEAD ON",
            "surface_condition": "wet",
            "traffic_control": "",
            "driver_substance_abuse": "ALCOHOL",
            "driver_distracted_by": "PHONE",
            "vehicle_body_type": "SUV",
            "vehicle_damage_extent": "SEVERE",
            "vehicle_movement": "STOPPED",
            "speed_limit": "50",
            "driver_at_fault": "No",
            "driverless_vehicle": "Yes",
            "parked_vehicle": False,
            "vehicle_year": "2022",
            "crash_hour": "7",
            "crash_dayofweek": "1",
            "crash_month": "12",
            "crash_year": "2026",
        },
        reference_year=2026,
    )

    assert payload["weather"] == "rain"
    assert payload["light"] == "DAYLIGHT"
    assert payload["traffic_control"] == "NO CONTROLS"
    assert payload["speed_limit"] == 50
    assert payload["driverless_vehicle"] == "Yes"
    assert payload["parked_vehicle"] == "No"
    assert payload["crash_year"] == 2026


def test_describe_severity_maps_known_and_unknown_labels() -> None:
    """Severity labels should be translated into a friendly Polish description."""

    no_injury = predictor.describe_severity("no_injury")
    unknown = predictor.describe_severity("unexpected_label")

    assert no_injury["label"] == "Brak obrażeń"
    assert no_injury["injury_detected"] is False
    assert unknown["label"] == "unexpected_label"
    assert unknown["injury_detected"] is None


def test_sorted_probabilities_orders_values_descending() -> None:
    """Probability values should be ordered from the highest to the lowest."""

    rows = predictor.sorted_probabilities({"SERIOUS": 0.01, "MINOR": 0.17, "NO_INJURY": 0.82})

    assert [row["label"] for row in rows] == ["NO_INJURY", "MINOR", "SERIOUS"]
    assert rows[0]["probability"] == 0.82


def test_check_api_health_uses_normalized_base_url(monkeypatch) -> None:
    """Health checks should request the /health endpoint on the normalized base URL."""

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        return _FakeResponse({"status": "ok", "model_loaded": True})

    monkeypatch.setattr(predictor.request, "urlopen", fake_urlopen)

    result = predictor.check_api_health("http://localhost:8000/predict")

    assert result["status"] == "ok"
    assert captured["url"] == "http://localhost:8000/health"
    assert captured["method"] == "GET"


def test_predict_via_api_posts_payload(monkeypatch) -> None:
    """Predictions should be sent to /predict as a JSON POST request."""

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "severity": "NO_INJURY",
                "probabilities": {"NO_INJURY": 0.82, "MINOR": 0.17, "SERIOUS": 0.01},
                "timestamp": "2026-05-24T12:00:00",
            }
        )

    monkeypatch.setattr(predictor.request, "urlopen", fake_urlopen)

    payload = {"weather": "CLEAR", "speed_limit": 35}
    result = predictor.predict_via_api("http://localhost:8000", payload)

    assert result["severity"] == "NO_INJURY"
    assert captured["url"] == "http://localhost:8000/predict"
    assert captured["method"] == "POST"
    assert captured["body"] == payload



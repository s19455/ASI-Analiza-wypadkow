"""Funkcje pomocnicze wspoldzielone przez frontend Streamlit i klienta API."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from typing import Any
from urllib import error, request

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 10.0

SEVERITY_LABELS = {
    "NO_INJURY": {
        "label": "Brak obrażeń",
        "description": "Model nie przewiduje obrażeń u uczestników zdarzenia.",
        "injury_detected": False,
    },
    "MINOR": {
        "label": "Drobne obrażenia",
        "description": "Model przewiduje lekkie obrażenia.",
        "injury_detected": True,
    },
    "SERIOUS": {
        "label": "Poważne obrażenia / zgon",
        "description": "Model przewiduje poważne obrażenia albo ofiarę śmiertelną.",
        "injury_detected": True,
    },
    "UNKNOWN": {
        "label": "Nieznany wynik",
        "description": "Model nie zwrócił jednoznacznej klasy.",
        "injury_detected": None,
    },
}

DEFAULT_FORM_VALUES = {
    "weather": "CLEAR",
    "light": "DAYLIGHT",
    "collision_type": "SAME DIR REAR END",
    "surface_condition": "DRY",
    "traffic_control": "NO CONTROLS",
    "driver_substance_abuse": "NONE DETECTED",
    "driver_distracted_by": "NOT DISTRACTED",
    "vehicle_body_type": "PASSENGER CAR",
    "vehicle_damage_extent": "FUNCTIONAL",
    "vehicle_movement": "MOVING CONSTANT SPEED",
    "speed_limit": 35,
    "driver_at_fault": "Yes",
    "driverless_vehicle": "No",
    "parked_vehicle": "No",
    "vehicle_year": 2020,
    "crash_hour": 12,
    "crash_dayofweek": 2,
    "crash_month": 6,
    "crash_year": dt.datetime.now().year,
}


class PredictionAPIError(RuntimeError):
    """Rzucany, gdy API jest nieosiagalne albo zwraca niepoprawne dane."""


def normalize_api_base_url(base_url: str | None) -> str:
    """Zwraca znormalizowany adres bazowy API (bez koncowek endpointow)."""

    if not base_url:
        return DEFAULT_API_URL

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/predict", "/health", "/predictions/recent"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized.rstrip("/") or DEFAULT_API_URL


def default_form_values(reference_year: int | None = None) -> dict[str, Any]:
    """Zwraca domyslne wartosci formularza wypadku."""

    current_year = dt.datetime.now().year if reference_year is None else reference_year
    defaults = dict(DEFAULT_FORM_VALUES)
    defaults["crash_year"] = current_year
    return defaults


def _coerce_text(value: Any, default: str) -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text or default


def _coerce_yes_no(value: Any, default: str) -> str:
    if value is None:
        return default

    if isinstance(value, bool):
        return "Yes" if value else "No"

    text = str(value).strip()
    if not text:
        return default

    lowered = text.lower()
    if lowered in {"yes", "y", "true", "1"}:
        return "Yes"
    if lowered in {"no", "n", "false", "0"}:
        return "No"
    return text


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_prediction_payload(
    form_values: Mapping[str, Any],
    reference_year: int | None = None,
) -> dict[str, Any]:
    """Zamienia dane z formularza na payload JSON oczekiwany przez API."""

    defaults = default_form_values(reference_year=reference_year)
    current_year = defaults["crash_year"]

    return {
        "weather": _coerce_text(form_values.get("weather"), defaults["weather"]),
        "light": _coerce_text(form_values.get("light"), defaults["light"]),
        "collision_type": _coerce_text(
            form_values.get("collision_type"), defaults["collision_type"]
        ),
        "surface_condition": _coerce_text(
            form_values.get("surface_condition"), defaults["surface_condition"]
        ),
        "traffic_control": _coerce_text(
            form_values.get("traffic_control"), defaults["traffic_control"]
        ),
        "driver_substance_abuse": _coerce_text(
            form_values.get("driver_substance_abuse"),
            defaults["driver_substance_abuse"],
        ),
        "driver_distracted_by": _coerce_text(
            form_values.get("driver_distracted_by"), defaults["driver_distracted_by"]
        ),
        "vehicle_body_type": _coerce_text(
            form_values.get("vehicle_body_type"), defaults["vehicle_body_type"]
        ),
        "vehicle_damage_extent": _coerce_text(
            form_values.get("vehicle_damage_extent"), defaults["vehicle_damage_extent"]
        ),
        "vehicle_movement": _coerce_text(
            form_values.get("vehicle_movement"), defaults["vehicle_movement"]
        ),
        "speed_limit": _coerce_int(
            form_values.get("speed_limit"), defaults["speed_limit"]
        ),
        "driver_at_fault": _coerce_yes_no(
            form_values.get("driver_at_fault"), defaults["driver_at_fault"]
        ),
        "driverless_vehicle": _coerce_yes_no(
            form_values.get("driverless_vehicle"), defaults["driverless_vehicle"]
        ),
        "parked_vehicle": _coerce_yes_no(
            form_values.get("parked_vehicle"), defaults["parked_vehicle"]
        ),
        "vehicle_year": _coerce_int(
            form_values.get("vehicle_year"), defaults["vehicle_year"]
        ),
        "crash_hour": _coerce_int(
            form_values.get("crash_hour"), defaults["crash_hour"]
        ),
        "crash_dayofweek": _coerce_int(
            form_values.get("crash_dayofweek"), defaults["crash_dayofweek"]
        ),
        "crash_month": _coerce_int(
            form_values.get("crash_month"), defaults["crash_month"]
        ),
        "crash_year": _coerce_int(form_values.get("crash_year"), current_year),
    }


def describe_severity(severity: Any) -> dict[str, Any]:
    """Tlumaczy etykiete modelu na czytelny opis po polsku."""

    severity_key = str(severity).strip().upper()
    details = SEVERITY_LABELS.get(severity_key)
    if details is not None:
        return {
            "severity": severity_key,
            "label": details["label"],
            "description": details["description"],
            "injury_detected": details["injury_detected"],
        }

    return {
        "severity": str(severity),
        "label": str(severity),
        "description": "Model zwrócił etykietę spoza oczekiwanego słownika.",
        "injury_detected": None,
    }


def sorted_probabilities(probabilities: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Zwraca prawdopodobienstwa posortowane od najwiekszego do najmniejszego."""

    normalized: list[dict[str, Any]] = []
    for label, value in probabilities.items():
        try:
            probability = float(value)
        except (TypeError, ValueError):
            continue
        normalized.append({"label": str(label), "probability": round(probability, 4)})

    normalized.sort(key=lambda item: item["probability"], reverse=True)
    return normalized


def _json_request(
    url: str,
    *,
    method: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_body = response.read().decode(charset)
    except error.HTTPError as exc:
        error_body = ""
        if exc.fp is not None:
            charset = exc.headers.get_content_charset() or "utf-8"
            error_body = exc.read().decode(charset, errors="replace")
        message = error_body.strip() or exc.reason or "brak treści odpowiedzi"
        raise PredictionAPIError(f"API zwróciło błąd {exc.code}: {message}") from exc
    except error.URLError as exc:
        raise PredictionAPIError(
            f"Nie udało się połączyć z API pod adresem {url}: {exc.reason}"
        ) from exc

    if not raw_body.strip():
        return {}

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise PredictionAPIError(
            f"API pod adresem {url} zwróciło niepoprawny JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise PredictionAPIError(
            f"API pod adresem {url} zwróciło nieoczekiwany format danych."
        )

    return parsed


def check_api_health(
    base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Wywoluje endpoint /health API."""

    normalized_base_url = normalize_api_base_url(base_url)
    return _json_request(
        f"{normalized_base_url}/health",
        method="GET",
        payload=None,
        timeout=timeout,
    )


def predict_via_api(
    base_url: str | None,
    payload: Mapping[str, Any],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wysyla zadanie predykcji do API i zwraca odpowiedz JSON."""

    normalized_base_url = normalize_api_base_url(base_url)
    return _json_request(
        f"{normalized_base_url}/predict",
        method="POST",
        payload=payload,
        timeout=timeout,
    )

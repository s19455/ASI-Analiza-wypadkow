"""Streamlit front-end for crash severity prediction."""

from __future__ import annotations

import datetime as dt
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from crash_kedro.api.predictor import (
    DEFAULT_API_URL,
    PredictionAPIError,
    build_prediction_payload,
    check_api_health,
    default_form_values,
    describe_severity,
    predict_via_api,
    sorted_probabilities,
)

DAY_NAMES = {
    0: "Poniedziałek",
    1: "Wtorek",
    2: "Środa",
    3: "Czwartek",
    4: "Piątek",
    5: "Sobota",
    6: "Niedziela",
}
MONTH_NAMES = {
    1: "Styczeń",
    2: "Luty",
    3: "Marzec",
    4: "Kwiecień",
    5: "Maj",
    6: "Czerwiec",
    7: "Lipiec",
    8: "Sierpień",
    9: "Wrzesień",
    10: "Październik",
    11: "Listopad",
    12: "Grudzień",
}

# pola formularza (selectbox) -> nazwy kolumn z modelu/enkoderow
CATEGORY_COLUMNS = {
    "weather": "Weather",
    "light": "Light",
    "collision_type": "Collision Type",
    "surface_condition": "Surface Condition",
    "traffic_control": "Traffic Control",
    "driver_substance_abuse": "Driver Substance Abuse",
    "driver_distracted_by": "Driver Distracted By",
    "vehicle_body_type": "Vehicle Body Type",
    "vehicle_damage_extent": "Vehicle Damage Extent",
    "vehicle_movement": "Vehicle Movement",
}

ENCODERS_PATH = Path(__file__).resolve().parents[3] / "data" / "06_models" / "encoders.pkl"


def load_category_options(encoders_path: Path = ENCODERS_PATH) -> dict[str, list[str]]:
    """Read possible category values for each form field from the encoders file.

    Returns an empty dict if the file is missing - the form then falls back to
    just the default value for each field.
    """

    try:
        with open(encoders_path, "rb") as stream:
            encoders = pickle.load(stream)
    except (OSError, pickle.PickleError):
        return {}

    if not isinstance(encoders, dict):
        return {}

    options: dict[str, list[str]] = {}
    for field_name, column_name in CATEGORY_COLUMNS.items():
        values = encoders.get(column_name)
        if isinstance(values, dict) and values:
            options[field_name] = sorted(str(key) for key in values)
    return options


def _render_select(
    streamlit: Any,
    label: str,
    options: list[str],
    default: str,
    help_text: str,
) -> str:
    """Render a selectbox, making sure the default value is always selectable."""

    choices = list(options)
    if default not in choices:
        choices = [default, *choices]
    return streamlit.selectbox(label, choices, index=choices.index(default), help=help_text)


def _render_binary_choice(streamlit: Any, label: str, default: str) -> str:
    options = ["Yes", "No"]
    return streamlit.selectbox(label, options, index=0 if default == "Yes" else 1)


def _render_form(
    streamlit: Any,
    defaults: Mapping[str, Any],
    options: Mapping[str, list[str]],
) -> dict[str, Any]:
    current_year = dt.datetime.now().year
    with streamlit.form("crash_severity_form"):
        left_column, right_column = streamlit.columns(2)

        with left_column:
            streamlit.subheader("🌧️ Warunki zdarzenia")
            weather = _render_select(
                streamlit,
                "Pogoda",
                options.get("weather", []),
                str(defaults["weather"]),
                "Warunki pogodowe w chwili zdarzenia.",
            )
            light = _render_select(
                streamlit,
                "Warunki oświetlenia",
                options.get("light", []),
                str(defaults["light"]),
                "Pora dnia / oświetlenie drogi.",
            )
            collision_type = _render_select(
                streamlit,
                "Typ kolizji",
                options.get("collision_type", []),
                str(defaults["collision_type"]),
                "Rodzaj zderzenia pojazdów.",
            )
            surface_condition = _render_select(
                streamlit,
                "Stan nawierzchni",
                options.get("surface_condition", []),
                str(defaults["surface_condition"]),
                "Stan nawierzchni drogi.",
            )
            traffic_control = _render_select(
                streamlit,
                "Kontrola ruchu",
                options.get("traffic_control", []),
                str(defaults["traffic_control"]),
                "Rodzaj sygnalizacji / oznakowania.",
            )
            driver_substance_abuse = _render_select(
                streamlit,
                "Użycie substancji przez kierowcę",
                options.get("driver_substance_abuse", []),
                str(defaults["driver_substance_abuse"]),
                "Czy wykryto substancje u kierowcy.",
            )
            driver_distracted_by = _render_select(
                streamlit,
                "Rozproszenie kierowcy",
                options.get("driver_distracted_by", []),
                str(defaults["driver_distracted_by"]),
                "Co rozpraszało kierowcę.",
            )

        with right_column:
            streamlit.subheader("🚗 Pojazd i czas")
            vehicle_body_type = _render_select(
                streamlit,
                "Typ pojazdu",
                options.get("vehicle_body_type", []),
                str(defaults["vehicle_body_type"]),
                "Rodzaj nadwozia pojazdu.",
            )
            vehicle_damage_extent = _render_select(
                streamlit,
                "Zakres uszkodzeń pojazdu",
                options.get("vehicle_damage_extent", []),
                str(defaults["vehicle_damage_extent"]),
                "Jak bardzo pojazd został uszkodzony.",
            )
            vehicle_movement = _render_select(
                streamlit,
                "Ruch pojazdu",
                options.get("vehicle_movement", []),
                str(defaults["vehicle_movement"]),
                "Co robił pojazd w chwili zdarzenia.",
            )
            speed_limit = streamlit.number_input(
                "Ograniczenie prędkości",
                min_value=0,
                max_value=150,
                value=int(defaults["speed_limit"]),
                step=5,
                help="Limit drogi w km/h.",
            )
            driver_at_fault = _render_binary_choice(
                streamlit,
                "Kierowca był winny?",
                str(defaults["driver_at_fault"]),
            )
            driverless_vehicle = _render_binary_choice(
                streamlit,
                "Pojazd autonomiczny?",
                str(defaults["driverless_vehicle"]),
            )
            parked_vehicle = _render_binary_choice(
                streamlit,
                "Pojazd zaparkowany?",
                str(defaults["parked_vehicle"]),
            )
            vehicle_year = streamlit.number_input(
                "Rok produkcji pojazdu",
                min_value=1980,
                max_value=current_year + 1,
                value=int(defaults["vehicle_year"]),
                step=1,
                help="Rok produkcji pojazdu uczestniczącego w zdarzeniu.",
            )
            crash_hour = streamlit.number_input(
                "Godzina wypadku",
                min_value=0,
                max_value=23,
                value=int(defaults["crash_hour"]),
                step=1,
            )
            crash_dayofweek = streamlit.selectbox(
                "Dzień tygodnia",
                options=list(DAY_NAMES),
                format_func=lambda day: f"{day} - {DAY_NAMES[day]}",
                index=int(defaults["crash_dayofweek"]),
            )
            crash_month = streamlit.selectbox(
                "Miesiąc",
                options=list(MONTH_NAMES),
                format_func=lambda month: f"{month} - {MONTH_NAMES[month]}",
                index=int(defaults["crash_month"]) - 1,
            )
            crash_year = streamlit.number_input(
                "Rok wypadku",
                min_value=2000,
                max_value=current_year + 1,
                value=int(defaults["crash_year"]),
                step=1,
            )

        submitted = streamlit.form_submit_button("🔮 Oblicz prawdopodobieństwo obrażeń")

    return {
        "submitted": submitted,
        "form_values": {
            "weather": weather,
            "light": light,
            "collision_type": collision_type,
            "surface_condition": surface_condition,
            "traffic_control": traffic_control,
            "driver_substance_abuse": driver_substance_abuse,
            "driver_distracted_by": driver_distracted_by,
            "vehicle_body_type": vehicle_body_type,
            "vehicle_damage_extent": vehicle_damage_extent,
            "vehicle_movement": vehicle_movement,
            "speed_limit": speed_limit,
            "driver_at_fault": driver_at_fault,
            "driverless_vehicle": driverless_vehicle,
            "parked_vehicle": parked_vehicle,
            "vehicle_year": vehicle_year,
            "crash_hour": crash_hour,
            "crash_dayofweek": crash_dayofweek,
            "crash_month": crash_month,
            "crash_year": crash_year,
        },
    }


def _render_class_legend(streamlit: Any) -> None:
    """Show a short legend describing the three severity classes."""

    streamlit.markdown(
        "**Klasy obrażeń:** "
        "🟢 `NO_INJURY` - brak obrażeń &nbsp;|&nbsp; "
        "🟡 `MINOR` - drobne obrażenia &nbsp;|&nbsp; "
        "🔴 `SERIOUS` - poważne obrażenia / zgon"
    )


def main() -> None:
    """Run the Streamlit application."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - only happens without optional dependency
        raise RuntimeError(
            "Streamlit nie jest zainstalowany. Zainstaluj zależności projektu, aby uruchomić UI."
        ) from exc

    st.set_page_config(page_title="Crash Severity Predictor", page_icon="🚗", layout="wide")
    st.title("🚗 Analiza skutków wypadku drogowego")
    st.caption(
        "Wprowadź dane zdarzenia i pobierz prognozę klasy obrażeń z istniejącego API projektu."
    )
    _render_class_legend(st)

    defaults = default_form_values()
    options = load_category_options()
    if not options:
        st.warning(
            "Nie znaleziono pliku `encoders.pkl` - listy wartości są ograniczone. "
            "Uruchom `kedro run`, aby wygenerować enkodery."
        )

    api_url = st.sidebar.text_input(
        "Adres API predykcji",
        value=os.getenv("PREDICTION_API_URL", DEFAULT_API_URL),
        help="Domyślnie: http://localhost:8000",
    )
    st.sidebar.markdown(
        """
        **Jak uruchomić?**
        1. Wystartuj backend FastAPI.
        2. Otwórz tę aplikację Streamlit.
        3. Uzupełnij formularz i kliknij przycisk predykcji.
        """
    )

    # automatyczne sprawdzenie polaczenia z API przy starcie
    try:
        health = check_api_health(api_url)
    except PredictionAPIError as exc:
        st.sidebar.error("API niedostępne - uruchom backend FastAPI.")
        st.sidebar.caption(str(exc))
    else:
        if health.get("model_loaded"):
            st.sidebar.success("API działa, model załadowany.")
        else:
            st.sidebar.warning("API działa, ale brak modelu (wynik może być UNKNOWN).")
        with st.sidebar.expander("Szczegóły API"):
            st.json(health)

    form_state = _render_form(st, defaults, options)
    if not form_state["submitted"]:
        return

    payload = build_prediction_payload(form_state["form_values"], reference_year=defaults["crash_year"])

    with st.expander("Dane wysyłane do API", expanded=False):
        st.json(payload)

    try:
        with st.spinner("Wysyłam dane do API i pobieram wynik..."):
            response = predict_via_api(api_url, payload)
    except PredictionAPIError as exc:
        st.error(
            "Nie udało się pobrać predykcji. Sprawdź, czy backend jest uruchomiony i czy URL jest poprawny."
        )
        st.exception(exc)
        return

    severity_info = describe_severity(response.get("severity"))
    probabilities = sorted_probabilities(response.get("probabilities", {}))

    st.subheader("Wynik predykcji")
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Klasa", severity_info["label"])
    metric_middle.metric(
        "Czy doszło do obrażeń?",
        "Tak" if severity_info["injury_detected"] is True else "Nie" if severity_info["injury_detected"] is False else "Brak danych",
    )
    metric_right.metric("Timestamp", str(response.get("timestamp", "-")))

    if severity_info["injury_detected"] is True:
        st.warning(severity_info["description"])
    elif severity_info["injury_detected"] is False:
        st.success(severity_info["description"])
    else:
        st.info(severity_info["description"])

    if probabilities:
        probability_frame = pd.DataFrame(probabilities)
        st.subheader("Prawdopodobieństwa klas")
        st.bar_chart(probability_frame.set_index("label"))
        st.dataframe(probability_frame, use_container_width=True, hide_index=True)
    else:
        st.info("Model nie zwrócił prawdopodobieństw dla tej predykcji.")

    with st.expander("Surowa odpowiedź API", expanded=False):
        st.json(response)


if __name__ == "__main__":
    main()

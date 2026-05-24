"""Streamlit front-end for crash severity prediction."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
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


def _render_text_input(streamlit: Any, label: str, default: str, help_text: str) -> str:
    return streamlit.text_input(label, value=default, help=help_text)


def _render_binary_choice(streamlit: Any, label: str, default: str) -> str:
    options = ["Yes", "No"]
    return streamlit.selectbox(label, options, index=0 if default == "Yes" else 1)


def _render_form(streamlit: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    current_year = dt.datetime.now().year
    with streamlit.form("crash_severity_form"):
        left_column, right_column = streamlit.columns(2)

        with left_column:
            streamlit.subheader("Warunki zdarzenia")
            weather = _render_text_input(
                streamlit,
                "Pogoda",
                str(defaults["weather"]),
                "Przykłady: CLEAR, RAIN, CLOUDY, FOG.",
            )
            light = _render_text_input(
                streamlit,
                "Warunki oświetlenia",
                str(defaults["light"]),
                "Przykłady: DAYLIGHT, DARK, DUSK, DAWN.",
            )
            collision_type = _render_text_input(
                streamlit,
                "Typ kolizji",
                str(defaults["collision_type"]),
                "Przykłady: SAME DIR REAR END, HEAD ON, ANGLE.",
            )
            surface_condition = _render_text_input(
                streamlit,
                "Stan nawierzchni",
                str(defaults["surface_condition"]),
                "Przykłady: DRY, WET, SNOW, ICE.",
            )
            traffic_control = _render_text_input(
                streamlit,
                "Kontrola ruchu",
                str(defaults["traffic_control"]),
                "Przykłady: NO CONTROLS, TRAFFIC SIGNAL, STOP SIGN.",
            )
            driver_substance_abuse = _render_text_input(
                streamlit,
                "Użycie substancji przez kierowcę",
                str(defaults["driver_substance_abuse"]),
                "Przykłady: NONE DETECTED, ALCOHOL, DRUGS.",
            )
            driver_distracted_by = _render_text_input(
                streamlit,
                "Rozproszenie kierowcy",
                str(defaults["driver_distracted_by"]),
                "Przykłady: NOT DISTRACTED, CELL PHONE, PASSENGER.",
            )

        with right_column:
            streamlit.subheader("Pojazd i czas")
            vehicle_body_type = _render_text_input(
                streamlit,
                "Typ pojazdu",
                str(defaults["vehicle_body_type"]),
                "Przykłady: PASSENGER CAR, SUV, TRUCK, MOTORCYCLE.",
            )
            vehicle_damage_extent = _render_text_input(
                streamlit,
                "Zakres uszkodzeń pojazdu",
                str(defaults["vehicle_damage_extent"]),
                "Przykłady: FUNCTIONAL, DISABLING, MINIMAL.",
            )
            vehicle_movement = _render_text_input(
                streamlit,
                "Ruch pojazdu",
                str(defaults["vehicle_movement"]),
                "Przykłady: MOVING CONSTANT, STOPPED IN TRAFFIC, LEFT TURN.",
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

        submitted = streamlit.form_submit_button("Oblicz prawdopodobieństwo obrażeń")

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

    defaults = default_form_values()
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

    if st.sidebar.button("Sprawdź połączenie z API"):
        try:
            health = check_api_health(api_url)
        except PredictionAPIError as exc:
            st.sidebar.error(str(exc))
        else:
            st.sidebar.success("API działa poprawnie.")
            st.sidebar.json(health)

    st.info(
        "Jeśli API nie ma załadowanego modelu, wynik może zwrócić `UNKNOWN`. "
        "To aplikacja front-endowa korzystająca z istniejącego endpointu `/predict`."
    )

    form_state = _render_form(st, defaults)
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


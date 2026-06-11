"""Frontend Streamlit do predykcji stopnia obrazen w wypadkach."""

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
    fetch_recent_predictions,
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

# Pola formularza (selectbox) -> nazwy kolumn z modelu/enkoderów.
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
FORM_STATE_PREFIX = "crash_form_"

SCENARIO_PRESETS = {
    "Typowy bezpieczny wypadek": {
        "description": "Dzień, dobra pogoda, sucha nawierzchnia i niskie uszkodzenia pojazdu.",
        "values": {
            "weather": "CLEAR",
            "light": "DAYLIGHT",
            "collision_type": "SAME DIR REAR END",
            "surface_condition": "DRY",
            "traffic_control": "NO CONTROLS",
            "driver_substance_abuse": "NONE DETECTED",
            "driver_distracted_by": "NOT DISTRACTED",
            "vehicle_damage_extent": "FUNCTIONAL",
            "vehicle_movement": "MOVING CONSTANT SPEED",
            "speed_limit": 35,
            "driver_at_fault": "Yes",
            "driverless_vehicle": "No",
            "parked_vehicle": "No",
            "crash_hour": 12,
            "crash_dayofweek": 2,
            "crash_month": 6,
        },
    },
    "Noc i mokra nawierzchnia": {
        "description": "Ciemno, deszczowo i mokro - szybki test gorszych warunków drogowych.",
        "values": {
            "weather": "RAINING",
            "light": "DARK LIGHTS ON",
            "collision_type": "SAME DIR REAR END",
            "surface_condition": "WET",
            "traffic_control": "TRAFFIC SIGNAL",
            "driver_substance_abuse": "NONE DETECTED",
            "driver_distracted_by": "NOT DISTRACTED",
            "vehicle_damage_extent": "FUNCTIONAL",
            "vehicle_movement": "MOVING CONSTANT SPEED",
            "speed_limit": 45,
            "driver_at_fault": "Yes",
            "driverless_vehicle": "No",
            "parked_vehicle": "No",
            "crash_hour": 22,
            "crash_dayofweek": 5,
            "crash_month": 11,
        },
    },
    "Wysokie uszkodzenia pojazdu": {
        "description": "Większe uszkodzenia i wyższy limit prędkości przy standardowych warunkach.",
        "values": {
            "weather": "CLEAR",
            "light": "DAYLIGHT",
            "collision_type": "ANGLE MEETS LEFT TURN",
            "surface_condition": "DRY",
            "traffic_control": "TRAFFIC SIGNAL",
            "driver_substance_abuse": "NONE DETECTED",
            "driver_distracted_by": "NOT DISTRACTED",
            "vehicle_damage_extent": "DISABLING",
            "vehicle_movement": "MOVING CONSTANT SPEED",
            "speed_limit": 55,
            "driver_at_fault": "Yes",
            "driverless_vehicle": "No",
            "parked_vehicle": "No",
            "crash_hour": 16,
            "crash_dayofweek": 4,
            "crash_month": 7,
        },
    },
    "Potencjalnie poważne zdarzenie": {
        "description": "Noc, mokra nawierzchnia, wyższa prędkość i poważniejsze uszkodzenia.",
        "values": {
            "weather": "RAINING",
            "light": "DARK NO LIGHTS",
            "collision_type": "HEAD ON",
            "surface_condition": "WET",
            "traffic_control": "NO CONTROLS",
            "driver_substance_abuse": "ALCOHOL PRESENT",
            "driver_distracted_by": "UNKNOWN",
            "vehicle_damage_extent": "DESTROYED",
            "vehicle_movement": "MOVING CONSTANT SPEED",
            "speed_limit": 65,
            "driver_at_fault": "Yes",
            "driverless_vehicle": "No",
            "parked_vehicle": "No",
            "crash_hour": 2,
            "crash_dayofweek": 6,
            "crash_month": 12,
        },
    },
}


def load_category_options(encoders_path: Path = ENCODERS_PATH) -> dict[str, list[str]]:
    """Wczytuje mozliwe wartosci pol formularza z pliku enkoderow.

    Gdy plik nie istnieje, zwraca pusty slownik - wtedy formularz korzysta
    tylko z wartosci domyslnych.
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


def _form_key(field_name: str) -> str:
    return f"{FORM_STATE_PREFIX}{field_name}"


def _render_select(
    streamlit: Any,
    label: str,
    options: list[str],
    default: str,
    help_text: str,
    key: str,
) -> str:
    """Rysuje selectbox tak, zeby wartosc domyslna zawsze byla na liscie."""

    choices = list(options)
    if default not in choices:
        choices = [default, *choices]
    current_value = streamlit.session_state.get(key)
    if current_value is not None and current_value not in choices:
        choices = [str(current_value), *choices]
    return streamlit.selectbox(
        label,
        choices,
        index=choices.index(default),
        help=help_text,
        key=key,
    )


def _render_binary_choice(streamlit: Any, label: str, default: str, key: str) -> str:
    options = ["Yes", "No"]
    return streamlit.selectbox(label, options, index=0 if default == "Yes" else 1, key=key)


def _render_scenario_picker(streamlit: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Pokazuje szybkie scenariusze testowe i wczytuje wybrany do formularza."""

    selected_scenario = streamlit.sidebar.selectbox(
        "Szybki scenariusz testowy",
        options=list(SCENARIO_PRESETS),
        help="Wypełnia formularz przykładowymi danymi do szybkiego porównania predykcji.",
    )
    preset = SCENARIO_PRESETS[selected_scenario]
    streamlit.sidebar.caption(str(preset["description"]))

    values = dict(defaults)
    if streamlit.sidebar.button("Wczytaj scenariusz", use_container_width=True):
        values.update(preset["values"])
        for field_name, value in preset["values"].items():
            streamlit.session_state[_form_key(field_name)] = value
        streamlit.sidebar.success(f"Wczytano: {selected_scenario}")

    return values


def _render_form(
    streamlit: Any,
    defaults: Mapping[str, Any],
    options: Mapping[str, list[str]],
) -> dict[str, Any]:
    current_year = dt.datetime.now().year
    with streamlit.form("crash_severity_form"):
        left_column, right_column = streamlit.columns(2)

        with left_column:
            streamlit.subheader("Warunki zdarzenia")
            weather = _render_select(
                streamlit,
                "Pogoda",
                options.get("weather", []),
                str(defaults["weather"]),
                "Warunki pogodowe w chwili zdarzenia.",
                _form_key("weather"),
            )
            light = _render_select(
                streamlit,
                "Warunki oświetlenia",
                options.get("light", []),
                str(defaults["light"]),
                "Pora dnia / oświetlenie drogi.",
                _form_key("light"),
            )
            collision_type = _render_select(
                streamlit,
                "Typ kolizji",
                options.get("collision_type", []),
                str(defaults["collision_type"]),
                "Rodzaj zderzenia pojazdów.",
                _form_key("collision_type"),
            )
            surface_condition = _render_select(
                streamlit,
                "Stan nawierzchni",
                options.get("surface_condition", []),
                str(defaults["surface_condition"]),
                "Stan nawierzchni drogi.",
                _form_key("surface_condition"),
            )
            traffic_control = _render_select(
                streamlit,
                "Kontrola ruchu",
                options.get("traffic_control", []),
                str(defaults["traffic_control"]),
                "Rodzaj sygnalizacji / oznakowania.",
                _form_key("traffic_control"),
            )
            driver_substance_abuse = _render_select(
                streamlit,
                "Użycie substancji przez kierowcę",
                options.get("driver_substance_abuse", []),
                str(defaults["driver_substance_abuse"]),
                "Czy wykryto substancje u kierowcy.",
                _form_key("driver_substance_abuse"),
            )
            driver_distracted_by = _render_select(
                streamlit,
                "Rozproszenie kierowcy",
                options.get("driver_distracted_by", []),
                str(defaults["driver_distracted_by"]),
                "Co rozpraszało kierowcę.",
                _form_key("driver_distracted_by"),
            )

        with right_column:
            streamlit.subheader("Pojazd i czas")
            vehicle_body_type = _render_select(
                streamlit,
                "Typ pojazdu",
                options.get("vehicle_body_type", []),
                str(defaults["vehicle_body_type"]),
                "Rodzaj nadwozia pojazdu.",
                _form_key("vehicle_body_type"),
            )
            vehicle_damage_extent = _render_select(
                streamlit,
                "Zakres uszkodzeń pojazdu",
                options.get("vehicle_damage_extent", []),
                str(defaults["vehicle_damage_extent"]),
                "Jak bardzo pojazd został uszkodzony.",
                _form_key("vehicle_damage_extent"),
            )
            vehicle_movement = _render_select(
                streamlit,
                "Ruch pojazdu",
                options.get("vehicle_movement", []),
                str(defaults["vehicle_movement"]),
                "Co robił pojazd w chwili zdarzenia.",
                _form_key("vehicle_movement"),
            )
            speed_limit = streamlit.number_input(
                "Ograniczenie prędkości",
                min_value=0,
                max_value=150,
                value=int(defaults["speed_limit"]),
                step=5,
                help="Limit drogi w km/h.",
                key=_form_key("speed_limit"),
            )
            driver_at_fault = _render_binary_choice(
                streamlit,
                "Kierowca był winny?",
                str(defaults["driver_at_fault"]),
                _form_key("driver_at_fault"),
            )
            driverless_vehicle = _render_binary_choice(
                streamlit,
                "Pojazd autonomiczny?",
                str(defaults["driverless_vehicle"]),
                _form_key("driverless_vehicle"),
            )
            parked_vehicle = _render_binary_choice(
                streamlit,
                "Pojazd zaparkowany?",
                str(defaults["parked_vehicle"]),
                _form_key("parked_vehicle"),
            )
            vehicle_year = streamlit.number_input(
                "Rok produkcji pojazdu",
                min_value=1980,
                max_value=current_year + 1,
                value=int(defaults["vehicle_year"]),
                step=1,
                help="Rok produkcji pojazdu uczestniczącego w zdarzeniu.",
                key=_form_key("vehicle_year"),
            )
            crash_hour = streamlit.number_input(
                "Godzina wypadku",
                min_value=0,
                max_value=23,
                value=int(defaults["crash_hour"]),
                step=1,
                key=_form_key("crash_hour"),
            )
            crash_dayofweek = streamlit.selectbox(
                "Dzień tygodnia",
                options=list(DAY_NAMES),
                format_func=lambda day: f"{day} - {DAY_NAMES[day]}",
                index=int(defaults["crash_dayofweek"]),
                key=_form_key("crash_dayofweek"),
            )
            crash_month = streamlit.selectbox(
                "Miesiąc",
                options=list(MONTH_NAMES),
                format_func=lambda month: f"{month} - {MONTH_NAMES[month]}",
                index=int(defaults["crash_month"]) - 1,
                key=_form_key("crash_month"),
            )
            crash_year = streamlit.number_input(
                "Rok wypadku",
                min_value=2000,
                max_value=current_year + 1,
                value=int(defaults["crash_year"]),
                step=1,
                key=_form_key("crash_year"),
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


def _render_class_legend(streamlit: Any) -> None:
    """Pokazuje krotka legende trzech klas obrazen."""

    streamlit.markdown(
        "**Klasy obrażeń:** "
        "`NO_INJURY` - brak obrażeń &nbsp;|&nbsp; "
        "`MINOR` - drobne obrażenia &nbsp;|&nbsp; "
        "`SERIOUS` - poważne obrażenia / zgon"
    )


def _format_prediction_history(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Zamienia surowe logi predykcji na wiersze do tabeli Streamlit."""

    rows: list[dict[str, Any]] = []
    for entry in reversed(predictions):
        if not isinstance(entry, dict):
            continue

        input_data = entry.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}

        probabilities = sorted_probabilities(entry.get("probabilities", {}))
        top_probability = probabilities[0]["probability"] if probabilities else None
        severity_info = describe_severity(entry.get("prediction"))

        rows.append(
            {
                "Czas": entry.get("timestamp", "-"),
                "Wynik": severity_info["label"],
                "Pewność": None if top_probability is None else f"{top_probability:.0%}",
                "Pogoda": input_data.get("weather", "-"),
                "Światło": input_data.get("light", "-"),
                "Limit": input_data.get("speed_limit", "-"),
            }
        )

    return rows


def _render_prediction_history(streamlit: Any, api_url: str) -> None:
    """Pokazuje w sidebarze ostatnie predykcje pobrane z API."""

    with streamlit.sidebar.expander("Historia predykcji", expanded=False):
        history_limit = streamlit.number_input(
            "Liczba wpisów",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="prediction_history_limit",
        )

        try:
            recent_payload = fetch_recent_predictions(api_url, limit=int(history_limit))
        except PredictionAPIError as exc:
            streamlit.caption("Nie udało się pobrać historii predykcji.")
            streamlit.caption(str(exc))
            return

        predictions = recent_payload.get("predictions", [])
        if not isinstance(predictions, list) or not predictions:
            streamlit.caption("Brak zapisanych predykcji.")
            return

        history_rows = _format_prediction_history(predictions)
        streamlit.dataframe(
            pd.DataFrame(history_rows),
            use_container_width=True,
            hide_index=True,
        )


def _render_api_status(streamlit: Any, api_url: str) -> None:
    """Pokazuje w sidebarze status API."""

    try:
        health = check_api_health(api_url)
    except PredictionAPIError as exc:
        streamlit.sidebar.error("API niedostępne - uruchom backend FastAPI.")
        streamlit.sidebar.caption(str(exc))
    else:
        if health.get("model_loaded"):
            streamlit.sidebar.success("API działa, model załadowany.")
        else:
            streamlit.sidebar.warning("API działa, ale brak modelu (wynik może być UNKNOWN).")
        with streamlit.sidebar.expander("Szczegóły API"):
            streamlit.json(health)


def _render_prediction_result(
    streamlit: Any,
    api_url: str,
    form_values: Mapping[str, Any],
    reference_year: int,
) -> None:
    """Wysyla dane formularza do API i pokazuje wynik predykcji."""

    payload = build_prediction_payload(form_values, reference_year=reference_year)

    with streamlit.expander("Dane wysyłane do API", expanded=False):
        streamlit.json(payload)

    try:
        with streamlit.spinner("Wysyłam dane do API i pobieram wynik..."):
            response = predict_via_api(api_url, payload)
    except PredictionAPIError as exc:
        streamlit.error(
            "Nie udało się pobrać predykcji. Sprawdź, czy backend jest uruchomiony i czy URL jest poprawny."
        )
        streamlit.exception(exc)
        return

    severity_info = describe_severity(response.get("severity"))
    probabilities = sorted_probabilities(response.get("probabilities", {}))

    streamlit.subheader("Wynik predykcji")
    metric_left, metric_middle, metric_right = streamlit.columns(3)
    metric_left.metric("Klasa", severity_info["label"])
    metric_middle.metric(
        "Czy doszło do obrażeń?",
        "Tak" if severity_info["injury_detected"] is True else "Nie" if severity_info["injury_detected"] is False else "Brak danych",
    )
    metric_right.metric("Timestamp", str(response.get("timestamp", "-")))

    if severity_info["injury_detected"] is True:
        streamlit.warning(severity_info["description"])
    elif severity_info["injury_detected"] is False:
        streamlit.success(severity_info["description"])
    else:
        streamlit.info(severity_info["description"])

    if probabilities:
        probability_frame = pd.DataFrame(probabilities)
        streamlit.subheader("Prawdopodobieństwa klas")
        streamlit.bar_chart(probability_frame.set_index("label"))
        streamlit.dataframe(probability_frame, use_container_width=True, hide_index=True)
    else:
        streamlit.info("Model nie zwrócił prawdopodobieństw dla tej predykcji.")

    with streamlit.expander("Surowa odpowiedź API", expanded=False):
        streamlit.json(response)


def main() -> None:
    """Uruchamia aplikacje Streamlit."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - only happens without optional dependency
        raise RuntimeError(
            "Streamlit nie jest zainstalowany. Zainstaluj zależności projektu, aby uruchomić UI."
        ) from exc

    st.set_page_config(page_title="Crash Severity Predictor", page_icon="🚗", layout="wide")
    st.title("Analiza skutków wypadku drogowego")
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

    defaults = _render_scenario_picker(st, defaults)

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

    form_state = _render_form(st, defaults, options)
    if form_state["submitted"]:
        _render_prediction_result(
            st,
            api_url,
            form_state["form_values"],
            reference_year=defaults["crash_year"],
        )

    _render_api_status(st, api_url)
    _render_prediction_history(st, api_url)


if __name__ == "__main__":
    main()

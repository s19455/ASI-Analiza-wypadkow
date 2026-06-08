"""Proste enkodery kategoryczne (label encoding) zapisywane jako slowniki.

Trzymamy mapowania kategoria -> liczba w zwyklych dict, dzieki czemu latwo je
zapisac picklem i wczytac przy inferencji bez dodatkowych zaleznosci.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
UNKNOWN_CATEGORY_CODE = -1

__all__ = [
    "fit_encoders",
    "transform_with_encoders",
    "save_encoders",
    "load_encoders",
]


def fit_encoders(
    df: pd.DataFrame,
    ignore_columns: list[str] | None = None,
) -> dict[str, object]:
    """Dopasowuje enkodery dla kolumn tekstowych (dtype object).

    Zwraca slownik {nazwa_kolumny: {kategoria: kod}}. Kolumny z ignore_columns
    sa pomijane (np. kolumna z targetem).
    """
    _validate_dataframe(df)
    ignore_set = _normalize_ignore_columns(ignore_columns)

    encoders: dict[str, object] = {}
    text_columns = df.select_dtypes(include=["object"]).columns

    for column_name in text_columns:
        if column_name in ignore_set:
            continue

        encoder = _build_encoder_mapping(df[column_name])
        encoders[column_name] = encoder
        LOGGER.debug("Enkoder dla '%s': %d kategorii", column_name, len(encoder))

    return encoders


def transform_with_encoders(
    df: pd.DataFrame,
    encoders: dict[str, object],
) -> pd.DataFrame:
    """Koduje kolumny DataFrame'u za pomoca dopasowanych enkoderow.

    Nie modyfikuje oryginalu. Kolumny spoza ``encoders`` zostaja bez zmian.
    Nieznane wartosci dostaja kod -1.
    """
    _validate_dataframe(df)
    validated_encoders = _validate_encoders_structure(encoders)

    transformed_df = df.copy(deep=True)
    for column_name, mapping in validated_encoders.items():
        if column_name not in transformed_df.columns:
            raise ValueError(f"Brak kolumny '{column_name}' w DataFrame.")

        transformed_df[column_name] = transformed_df[column_name].map(
            lambda value, column_mapping=mapping: _encode_single_value(
                value,
                column_mapping,
            )
        )

    return transformed_df


def save_encoders(encoders: dict[str, object], path: Path) -> None:
    """Zapisuje enkodery do pliku pickle."""
    validated_encoders = _validate_encoders_structure(encoders)
    target_path = Path(path)

    if not str(target_path):
        raise ValueError("Sciezka nie moze byc pusta.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as output_stream:
        pickle.dump(validated_encoders, output_stream, protocol=pickle.HIGHEST_PROTOCOL)

    LOGGER.info("Zapisano enkodery do %s", target_path)


def load_encoders(path: Path) -> dict[str, object]:
    """Wczytuje i waliduje enkodery z pliku pickle."""
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Plik z enkoderami nie istnieje: {source_path}")

    with source_path.open("rb") as input_stream:
        loaded_encoders = pickle.load(input_stream)

    validated_encoders = _validate_encoders_structure(loaded_encoders)
    LOGGER.info("Wczytano enkodery z %s", source_path)
    return validated_encoders


def _validate_dataframe(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Oczekiwano obiektu pandas DataFrame.")


def _normalize_ignore_columns(ignore_columns: list[str] | None) -> set[str]:
    if ignore_columns is None:
        return set()

    if not isinstance(ignore_columns, list):
        raise ValueError("ignore_columns musi byc lista stringow albo None.")

    normalized_columns: set[str] = set()
    for column_name in ignore_columns:
        if not isinstance(column_name, str):
            raise ValueError("ignore_columns moze zawierac tylko stringi.")
        normalized_columns.add(column_name)

    return normalized_columns


def _build_encoder_mapping(series: pd.Series) -> dict[object, int]:
    """Buduje mapowanie kategoria -> kolejny numer dla jednej kolumny."""
    cleaned_series = series.dropna()

    try:
        unique_values = pd.unique(cleaned_series)
    except TypeError as exc:
        raise ValueError(
            f"Kolumna '{series.name}' zawiera wartosci, ktorych nie da sie zakodowac."
        ) from exc

    mapping: dict[object, int] = {}
    for index, category in enumerate(unique_values.tolist()):
        try:
            hash(category)
        except TypeError as exc:
            raise ValueError(
                f"Kolumna '{series.name}' zawiera niehashowalne wartosci."
            ) from exc
        mapping[category] = index

    return mapping


def _validate_encoders_structure(encoders: Any) -> dict[str, dict[object, int]]:
    """Sprawdza, czy struktura enkoderow jest poprawna."""
    if not isinstance(encoders, dict):
        raise ValueError("Enkodery musza byc slownikiem.")

    validated_encoders: dict[str, dict[object, int]] = {}
    for column_name, encoder in encoders.items():
        if not isinstance(column_name, str):
            raise ValueError("Klucze enkoderow musza byc nazwami kolumn (string).")
        if not isinstance(encoder, dict):
            raise ValueError(f"Enkoder dla '{column_name}' musi byc slownikiem.")

        validated_encoder: dict[object, int] = {}
        for category, encoded_value in encoder.items():
            try:
                hash(category)
            except TypeError as exc:
                raise ValueError(
                    f"Enkoder dla '{column_name}' ma niehashowalne klucze."
                ) from exc
            if not isinstance(encoded_value, int):
                raise ValueError(
                    f"Enkoder dla '{column_name}' musi mapowac na liczby calkowite."
                )
            validated_encoder[category] = encoded_value

        validated_encoders[column_name] = validated_encoder

    return validated_encoders


def _encode_single_value(value: Any, mapping: dict[object, int]) -> int:
    """Koduje pojedyncza wartosc; brak/nieznana -> -1."""
    if _is_missing_value(value):
        return UNKNOWN_CATEGORY_CODE

    try:
        return mapping.get(value, UNKNOWN_CATEGORY_CODE)
    except TypeError:
        return UNKNOWN_CATEGORY_CODE


def _is_missing_value(value: Any) -> bool:
    try:
        missing_value = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return isinstance(missing_value, bool) and missing_value

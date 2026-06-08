"""
Pipeline przygotowania danych - czyszczenie, inzynieria cech, enkodowanie.
"""

import logging

import pandas as pd

from crash_kedro.utils.encoders import fit_encoders, transform_with_encoders

logger = logging.getLogger(__name__)


def drop_unnecessary_columns(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    columns_to_drop = parameters["columns_to_drop"]
    existing = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # prosta imputacja: moda dla tekstu, mediana dla liczb
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN")

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Crash Date/Time" in df.columns:
        df["Crash Date/Time"] = pd.to_datetime(df["Crash Date/Time"], format="mixed")
        df["crash_hour"] = df["Crash Date/Time"].dt.hour
        df["crash_dayofweek"] = df["Crash Date/Time"].dt.dayofweek
        df["crash_month"] = df["Crash Date/Time"].dt.month
        df["crash_year"] = df["Crash Date/Time"].dt.year
        df = df.drop(columns=["Crash Date/Time"])

    if "Light" in df.columns:
        df["is_night"] = df["Light"].str.contains("DARK", case=False, na=False).astype(int)

    if "Weather" in df.columns:
        df["is_bad_weather"] = (~df["Weather"].isin(["CLEAR", "CLOUDY"])).astype(int)

    if "Surface Condition" in df.columns:
        df["is_wet_surface"] = (~df["Surface Condition"].isin(["DRY"])).astype(int)

    if "Vehicle Year" in df.columns:
        current_year = pd.Timestamp.now().year
        df["vehicle_age"] = current_year - df["Vehicle Year"]
        df["vehicle_age"] = df["vehicle_age"].clip(lower=0, upper=50)

    return df


def map_target(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    severity_mapping = parameters["severity_mapping"]
    target_col = parameters["target_column"]

    reverse_map = {}
    for group_name, values in severity_mapping.items():
        for v in values:
            reverse_map[v] = group_name

    df["Severity_Group"] = df[target_col].map(reverse_map).fillna("NO_INJURY")
    df = df.drop(columns=[target_col])

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Koduje kolumny kategoryczne. Target Severity_Group zostaje bez zmian.

    Zwraca zakodowany DataFrame oraz slownik enkoderow (do zapisania i ponownego
    uzycia przy inferencji). Nieznane kategorie sa kodowane jako -1.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Oczekiwano obiektu pandas DataFrame.")

    # enkodery na wszystkich kolumnach object oprocz targetu
    ignore_cols = ["Severity_Group"]
    encoders_dict = fit_encoders(df, ignore_columns=ignore_cols)
    df_encoded = transform_with_encoders(df, encoders_dict)

    logger.info(
        "Zakodowano %d kolumn kategorycznych: %s",
        len(encoders_dict),
        ", ".join(encoders_dict.keys()) if encoders_dict else "brak",
    )

    return df_encoded, encoders_dict

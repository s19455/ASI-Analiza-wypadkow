"""
Pipeline przygotowania danych - czyszczenie, inzynieria cech, enkodowanie.
"""

import logging

import pandas as pd

from ...utils.encoders import fit_encoders, transform_with_encoders

logger = logging.getLogger(__name__)


def _combo_feature(df: pd.DataFrame, columns: list[str], new_column: str) -> None:
    """Tworzy cechę złożoną z kilku kolumn tekstowych, jeśli wszystkie istnieją."""
    if all(column in df.columns for column in columns):
        df[new_column] = (
            df[columns]
            .fillna("UNKNOWN")
            .astype(str)
            .agg("__".join, axis=1)
        )


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
        df["time_of_day"] = pd.cut(
            df["crash_hour"],
            bins=[-1, 5, 11, 15, 19, 23],
            labels=["night", "morning", "midday", "afternoon", "evening"],
            include_lowest=True,
        ).astype(str)
        df["is_rush_hour"] = df["crash_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
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
        df["vehicle_age_band"] = pd.cut(
            df["vehicle_age"],
            bins=[-1, 3, 7, 15, 25, 50],
            labels=["new", "young", "mid_age", "old", "very_old"],
            include_lowest=True,
        ).astype(str)

    _combo_feature(df, ["Light", "Weather"], "light_weather_combo")
    _combo_feature(df, ["Weather", "Surface Condition"], "weather_surface_combo")
    _combo_feature(df, ["Collision Type", "Traffic Control"], "collision_control_combo")
    _combo_feature(df, ["Vehicle Type", "Vehicle Damage"], "vehicle_damage_combo")

    airbag_columns = [column for column in df.columns if "airbag" in column.lower()]
    if airbag_columns:
        airbag_text = df[airbag_columns].fillna("").astype(str).agg(" ".join, axis=1)
        df["airbag_deployed"] = airbag_text.str.contains(
            r"DEPLOY|DEPLOYED|ACTIV|YES|TRUE",
            case=False,
            regex=True,
            na=False,
        ).astype(int)

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

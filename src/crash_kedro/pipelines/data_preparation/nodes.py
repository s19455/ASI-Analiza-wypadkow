"""
Pipeline przygotowania danych - czyszczenie, inzynieria cech, enkodowanie.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def drop_unnecessary_columns(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    columns_to_drop = parameters["columns_to_drop"]
    existing = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: pomyslec o lepszej imputacji - moze KNN imputer
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


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # uzywamy label encoding zamiast one-hot bo niektore kolumny maja
    # bardzo duzo unikatowych wartosci (Vehicle Make, Road Name itd.)
    for col in df.select_dtypes(include=["object"]).columns:
        if col == "Severity_Group":
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # print(f"encoded {col}: {len(le.classes_)} unikalnych")

    return df

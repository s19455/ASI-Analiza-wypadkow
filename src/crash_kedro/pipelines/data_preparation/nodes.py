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


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features in the DataFrame.

    This function fits label encoders on all text columns (dtype='object')
    except for the target column 'Severity_Group' and other critical columns,
    then applies the encoders to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with categorical and numerical columns.
        The 'Severity_Group' column is preserved as-is and not encoded.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing:
        - df_encoded: DataFrame with categorical features encoded to integers.
        - encoders_dict: Dictionary mapping column names to encoder dictionaries
          (mapping of original category values to integer codes).

    Raises
    ------
    ValueError
        If df is not a pandas DataFrame or if encoding fails.

    Notes
    -----
    Unknown categories encountered during transformation are coded as -1.
    The encoders can be saved to disk and reused for inference.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Weather': ['CLEAR', 'CLOUDY', 'RAIN'],
    ...     'Severity_Group': ['NO_INJURY', 'INJURY', 'FATALITY']
    ... })
    >>> encoded_df, encoders = encode_features(df)
    >>> isinstance(encoded_df, pd.DataFrame)
    True
    >>> isinstance(encoders, dict)
    True
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame.")

    # Fit encoders on all object columns except Severity_Group
    ignore_cols = ["Severity_Group"]
    encoders_dict = fit_encoders(df, ignore_columns=ignore_cols)

    # Transform the DataFrame using fitted encoders
    df_encoded = transform_with_encoders(df, encoders_dict)

    # Log encoding details
    logger.info(
        "Encoded %d categorical columns. %s",
        len(encoders_dict),
        ", ".join(encoders_dict.keys()) if encoders_dict else "none",
    )

    return df_encoded, encoders_dict

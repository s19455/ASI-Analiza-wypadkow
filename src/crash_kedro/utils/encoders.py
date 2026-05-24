"""Utilities for fitting, applying, and persisting categorical encoders.

The helpers in this module implement a lightweight, LabelEncoder-like workflow
for text columns stored as ``object`` dtype in pandas DataFrames. Encoders are
stored as plain dictionaries so they can be safely serialized with pickle and
loaded back without additional dependencies.
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
    """Fit encoders for text columns in a DataFrame.

    Parameters
    ----------
    df:
        Source DataFrame containing categorical text columns.
    ignore_columns:
        Optional list of column names that should be skipped during fitting.

    Returns
    -------
    dict[str, object]
        A dictionary mapping each encoded column name to a serializable mapping
        of category values to integer codes.

    Raises
    ------
    ValueError
        If ``df`` is not a DataFrame or if ``ignore_columns`` contains invalid
        values.
    """

    _validate_dataframe(df)
    ignore_set = _normalize_ignore_columns(ignore_columns)

    encoders: dict[str, object] = {}
    text_columns = df.select_dtypes(include=["object"]).columns

    for column_name in text_columns:
        if column_name in ignore_set:
            LOGGER.debug("Skipping ignored column: %s", column_name)
            continue

        encoder = _build_encoder_mapping(df[column_name])
        encoders[column_name] = encoder
        LOGGER.debug(
            "Fitted encoder for column '%s' with %d categories.",
            column_name,
            len(encoder),
        )

    return encoders


def transform_with_encoders(
    df: pd.DataFrame,
    encoders: dict[str, object],
) -> pd.DataFrame:
    """Transform a DataFrame using fitted encoders.

    The original DataFrame is not modified. Columns not present in ``encoders``
    remain unchanged.

    Parameters
    ----------
    df:
        DataFrame to transform.
    encoders:
        Encoders produced by :func:`fit_encoders` or loaded from disk.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with encoded columns.

    Raises
    ------
    ValueError
        If the inputs are invalid, if the encoder structure is malformed, or if
        an encoder refers to a missing column in ``df``.
    """

    _validate_dataframe(df)
    validated_encoders = _validate_encoders_structure(encoders)

    transformed_df = df.copy(deep=True)
    for column_name, mapping in validated_encoders.items():
        if column_name not in transformed_df.columns:
            raise ValueError(
                f"Column '{column_name}' is missing from the DataFrame."
            )

        transformed_df[column_name] = transformed_df[column_name].map(
            lambda value, column_mapping=mapping: _encode_single_value(
                value,
                column_mapping,
            )
        )

    return transformed_df


def save_encoders(encoders: dict[str, object], path: Path) -> None:
    """Serialize encoders to disk using pickle.

    Parameters
    ----------
    encoders:
        Encoders produced by :func:`fit_encoders`.
    path:
        Target file path for the serialized pickle file.

    Raises
    ------
    ValueError
        If the encoder structure is invalid or if ``path`` is empty.
    """

    validated_encoders = _validate_encoders_structure(encoders)
    target_path = Path(path)

    if not str(target_path):
        raise ValueError("Path must not be empty.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as output_stream:
        pickle.dump(validated_encoders, output_stream, protocol=pickle.HIGHEST_PROTOCOL)

    LOGGER.info("Saved encoders to %s", target_path)


def load_encoders(path: Path) -> dict[str, object]:
    """Load and validate encoders from a pickle file.

    Parameters
    ----------
    path:
        Path to a pickle file created by :func:`save_encoders`.

    Returns
    -------
    dict[str, object]
        Validated encoders ready to be passed to
        :func:`transform_with_encoders`.

    Raises
    ------
    ValueError
        If the file does not exist, cannot be read, or does not contain a valid
        encoder structure.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Encoder file does not exist: {source_path}")

    with source_path.open("rb") as input_stream:
        loaded_encoders = pickle.load(input_stream)

    validated_encoders = _validate_encoders_structure(loaded_encoders)
    LOGGER.info("Loaded encoders from %s", source_path)
    return validated_encoders


def _validate_dataframe(df: Any) -> None:
    """Validate that the provided object is a pandas DataFrame.

    Parameters
    ----------
    df:
        Object to validate.

    Raises
    ------
    ValueError
        If ``df`` is not a pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame.")


def _normalize_ignore_columns(ignore_columns: list[str] | None) -> set[str]:
    """Normalize ignore columns into a validated set of strings.

    Parameters
    ----------
    ignore_columns:
        Optional list of column names to skip.

    Returns
    -------
    set[str]
        A set of validated column names.

    Raises
    ------
    ValueError
        If ``ignore_columns`` contains values that are not strings.
    """

    if ignore_columns is None:
        return set()

    if not isinstance(ignore_columns, list):
        raise ValueError("ignore_columns must be a list of strings or None.")

    normalized_columns: set[str] = set()
    for column_name in ignore_columns:
        if not isinstance(column_name, str):
            raise ValueError("ignore_columns must contain only strings.")
        normalized_columns.add(column_name)

    return normalized_columns


def _build_encoder_mapping(series: pd.Series) -> dict[object, int]:
    """Build a category-to-integer mapping for one series.

    Parameters
    ----------
    series:
        Pandas Series containing the categories to encode.

    Returns
    -------
    dict[object, int]
        Mapping of observed categories to integer codes.

    Raises
    ------
    ValueError
        If the series contains unhashable values.
    """

    cleaned_series = series.dropna()

    try:
        unique_values = pd.unique(cleaned_series)
    except TypeError as exc:
        raise ValueError(
            f"Column '{series.name}' contains values that cannot be encoded."
        ) from exc

    mapping: dict[object, int] = {}
    for index, category in enumerate(unique_values.tolist()):
        try:
            hash(category)
        except TypeError as exc:
            raise ValueError(
                f"Column '{series.name}' contains unhashable values."
            ) from exc
        mapping[category] = index

    return mapping


def _validate_encoders_structure(encoders: Any) -> dict[str, dict[object, int]]:
    """Validate the serialized encoder structure.

    Parameters
    ----------
    encoders:
        Object to validate.

    Returns
    -------
    dict[str, dict[object, int]]
        A normalized dictionary containing category-to-code mappings.

    Raises
    ------
    ValueError
        If the structure is malformed.
    """

    if not isinstance(encoders, dict):
        raise ValueError("Encoders must be provided as a dictionary.")

    validated_encoders: dict[str, dict[object, int]] = {}
    for column_name, encoder in encoders.items():
        if not isinstance(column_name, str):
            raise ValueError("Encoder keys must be column names as strings.")
        if not isinstance(encoder, dict):
            raise ValueError(
                f"Encoder for column '{column_name}' must be a dictionary."
            )

        validated_encoder: dict[object, int] = {}
        for category, encoded_value in encoder.items():
            try:
                hash(category)
            except TypeError as exc:
                raise ValueError(
                    f"Encoder for column '{column_name}' contains unhashable keys."
                ) from exc
            if not isinstance(encoded_value, int):
                raise ValueError(
                    f"Encoder for column '{column_name}' must map to integers."
                )
            validated_encoder[category] = encoded_value

        validated_encoders[column_name] = validated_encoder

    return validated_encoders


def _encode_single_value(value: Any, mapping: dict[object, int]) -> int:
    """Encode a single value using a validated mapping.

    Parameters
    ----------
    value:
        Value to encode.
    mapping:
        Category-to-code mapping.

    Returns
    -------
    int
        Integer code for the category, or ``UNKNOWN_CATEGORY_CODE`` for values
        that are missing or unseen during fitting.
    """

    if _is_missing_value(value):
        return UNKNOWN_CATEGORY_CODE

    try:
        return mapping.get(value, UNKNOWN_CATEGORY_CODE)
    except TypeError:
        return UNKNOWN_CATEGORY_CODE


def _is_missing_value(value: Any) -> bool:
    """Check whether a scalar value should be treated as missing."""

    try:
        missing_value = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return isinstance(missing_value, bool) and missing_value

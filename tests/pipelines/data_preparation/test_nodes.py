"""Unit tests for data preparation pipeline nodes.

Tests validate the data preparation transformations including:
- Feature encoding with encoder fitting and persistence.
- Handling of categorical and numerical columns.
- Preservation of target column (Severity_Group).
- Deterministic behavior without heavy artifacts.
"""

import pandas as pd
import pytest

from src.crash_kedro.pipelines.data_preparation.nodes import encode_features, engineer_features


class TestEncodeFeatures:
    """Test suite for the encode_features function."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Fixture providing a minimal sample DataFrame for testing.

        Returns
        -------
        pd.DataFrame
            A DataFrame with categorical, numerical, and target columns.
        """
        return pd.DataFrame({
            "Weather": ["CLEAR", "CLOUDY", "CLEAR", "RAIN"],
            "Light": ["DAY", "DARK", "DAY", "DARK"],
            "Surface Condition": ["DRY", "WET", "DRY", "WET"],
            "crash_hour": [8, 14, 22, 3],
            "vehicle_age": [5, 10, 2, 15],
            "Severity_Group": ["NO_INJURY", "INJURY", "FATALITY", "NO_INJURY"],
        })

    def test_encode_features_returns_tuple(self, sample_df: pd.DataFrame) -> None:
        """Test that encode_features returns a tuple of DataFrame and dict.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        result = encode_features(sample_df)
        assert isinstance(result, tuple), "encode_features should return a tuple."
        assert len(result) == 2, "Tuple should have exactly 2 elements."

    def test_encode_features_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        """Test that first return value is a pandas DataFrame.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, _ = encode_features(sample_df)
        assert isinstance(df_encoded, pd.DataFrame), "First return value should be a DataFrame."

    def test_encode_features_returns_encoder_dict(self, sample_df: pd.DataFrame) -> None:
        """Test that second return value is a dictionary.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        _, encoders = encode_features(sample_df)
        assert isinstance(encoders, dict), "Second return value should be a dictionary."

    def test_categorical_columns_are_encoded(self, sample_df: pd.DataFrame) -> None:
        """Test that categorical columns are converted to numeric.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, encoders = encode_features(sample_df)

        # Check that object columns (except Severity_Group) are now numeric
        for col in ["Weather", "Light", "Surface Condition"]:
            if col in df_encoded.columns:
                assert df_encoded[col].dtype in [int, float], \
                    f"Column '{col}' should be numeric after encoding."

    def test_severity_group_is_not_encoded(self, sample_df: pd.DataFrame) -> None:
        """Test that Severity_Group is preserved as-is.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, encoders = encode_features(sample_df)

        # Check that Severity_Group is not in encoders
        assert "Severity_Group" not in encoders, \
            "Severity_Group should not be encoded."

        # Check that Severity_Group column is still object type
        assert df_encoded["Severity_Group"].dtype == object, \
            "Severity_Group should remain as object type."

        # Check that values match original
        pd.testing.assert_series_equal(
            df_encoded["Severity_Group"],
            sample_df["Severity_Group"],
            check_names=True,
        )

    def test_numeric_columns_unchanged(self, sample_df: pd.DataFrame) -> None:
        """Test that numeric columns remain unchanged.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, _ = encode_features(sample_df)

        # Numeric columns should be unchanged (values, not dtype necessarily)
        pd.testing.assert_series_equal(
            df_encoded["crash_hour"],
            sample_df["crash_hour"],
            check_names=True,
        )
        pd.testing.assert_series_equal(
            df_encoded["vehicle_age"],
            sample_df["vehicle_age"],
            check_names=True,
        )

    def test_encoders_dict_maps_to_integers(self, sample_df: pd.DataFrame) -> None:
        """Test that encoder mappings are dictionaries with integer values.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        _, encoders = encode_features(sample_df)

        for col_name, encoder in encoders.items():
            assert isinstance(encoder, dict), \
                f"Encoder for '{col_name}' should be a dictionary."
            for key, value in encoder.items():
                assert isinstance(value, int), \
                    f"Encoder for '{col_name}' should map to integers."

    def test_encode_features_idempotency_on_unique_values(self, sample_df: pd.DataFrame) -> None:
        """Test that encoding is deterministic (consistent across runs).

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        # First encoding
        df_encoded_1, encoders_1 = encode_features(sample_df)

        # Second encoding
        df_encoded_2, encoders_2 = encode_features(sample_df)

        # Check that encoded values are identical
        pd.testing.assert_frame_equal(df_encoded_1, df_encoded_2)

        # Check that encoders are identical
        assert encoders_1 == encoders_2, "Encoders should be identical across runs."

    def test_encode_features_with_nan_values(self) -> None:
        """Test encoding handles NaN values gracefully.

        NaN values should be converted to -1 (unknown category code).
        """
        df_with_nan = pd.DataFrame({
            "Weather": ["CLEAR", None, "RAIN", "CLOUDY"],
            "crash_hour": [8, 14, 22, 3],
            "Severity_Group": ["NO_INJURY", "INJURY", "FATALITY", "NO_INJURY"],
        })

        df_encoded, encoders = encode_features(df_with_nan)

        # Check that Weather column is numeric
        assert df_encoded["Weather"].dtype in [int, float], \
            "Weather should be numeric after encoding."

        # Check that NaN was converted to -1
        assert df_encoded.loc[1, "Weather"] == -1, \
            "NaN should be encoded as -1."

    def test_encode_features_raises_on_non_dataframe(self) -> None:
        """Test that encode_features raises ValueError for non-DataFrame input."""
        with pytest.raises(ValueError, match="pandas DataFrame"):
            encode_features([1, 2, 3])

    def test_encode_features_with_unknown_categories(self, sample_df: pd.DataFrame) -> None:
        """Test that unknown categories are handled safely.

        When the encoder encounters an unknown category (not in the training encoder),
        it should use the unknown category code (-1).
        """
        # Fit encoders on the sample
        df_encoded_1, encoders = encode_features(sample_df)

        # Create a new DataFrame with an unknown category
        df_with_unknown = pd.DataFrame({
            "Weather": ["CLEAR", "CLOUDY", "SNOW"],  # SNOW is unknown
            "Light": ["DAY", "DARK", "DAY"],
            "Surface Condition": ["DRY", "WET", "ICE"],  # ICE is unknown
            "crash_hour": [8, 14, 22],
            "vehicle_age": [5, 10, 2],
            "Severity_Group": ["NO_INJURY", "INJURY", "FATALITY"],
        })

        # Transform using fitted encoders (simulating inference)
        from src.crash_kedro.utils.encoders import transform_with_encoders
        df_transformed = transform_with_encoders(df_with_unknown, encoders)

        # Unknown categories should be encoded as -1
        assert df_transformed.loc[2, "Weather"] == -1, \
            "Unknown category 'SNOW' should be encoded as -1."
        assert df_transformed.loc[2, "Surface Condition"] == -1, \
            "Unknown category 'ICE' should be encoded as -1."

    def test_output_shapes_match_input(self, sample_df: pd.DataFrame) -> None:
        """Test that output DataFrame has same shape as input.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, _ = encode_features(sample_df)

        assert df_encoded.shape == sample_df.shape, \
            "Output DataFrame should have the same shape as input."

    def test_encode_features_preserves_column_count(self, sample_df: pd.DataFrame) -> None:
        """Test that encoding does not add or remove columns.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample input DataFrame.
        """
        df_encoded, _ = encode_features(sample_df)

        assert len(df_encoded.columns) == len(sample_df.columns), \
            "Number of columns should be preserved."

    def test_engineer_features_adds_interactions_and_buckets(self) -> None:
        """Test that engineer_features creates richer categorical/time features."""
        df = pd.DataFrame({
            "Crash Date/Time": ["2024-01-01 08:30:00", "2024-01-01 20:15:00"],
            "Light": ["DARK - LIGHTS ON", "DAYLIGHT"],
            "Weather": ["RAIN", "CLEAR"],
            "Surface Condition": ["WET", "DRY"],
            "Vehicle Year": [2018, 2023],
            "Collision Type": ["ANGLE", "REAR END"],
            "Traffic Control": ["STOP SIGN", "NONE"],
            "Vehicle Type": ["SEDAN", "SUV"],
            "Vehicle Damage": ["DISABLED", "FUNCTIONAL"],
        })

        engineered = engineer_features(df)

        expected_columns = {
            "crash_hour",
            "crash_dayofweek",
            "crash_month",
            "crash_year",
            "time_of_day",
            "is_rush_hour",
            "vehicle_age",
            "vehicle_age_band",
            "light_weather_combo",
            "weather_surface_combo",
            "collision_control_combo",
            "vehicle_damage_combo",
        }

        for column in expected_columns:
            assert column in engineered.columns, f"Brak oczekiwanej cechy: {column}"

        assert engineered.loc[0, "time_of_day"] == "morning"
        assert engineered.loc[1, "time_of_day"] == "evening"
        assert engineered.loc[0, "is_rush_hour"] == 1
        assert engineered.loc[0, "light_weather_combo"] == "DARK - LIGHTS ON__RAIN"
        assert engineered.loc[0, "weather_surface_combo"] == "RAIN__WET"


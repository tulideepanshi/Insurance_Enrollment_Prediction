"""
Tests for src.data.data_loader module.

Covers: schema validation, null detection, duplicate detection,
range checks, stratified splitting, and feature/target separation.
"""

import pandas as pd
import pytest

from src.data.data_loader import (
    get_features_and_target,
    split_data,
    validate_dataframe,
)


class TestValidateDataframe:
    """Tests for the validate_dataframe function."""

    def test_valid_data_passes(self, sample_raw_df):
        """Happy path — valid data returns the same DataFrame."""
        result = validate_dataframe(sample_raw_df)
        assert result is sample_raw_df

    def test_missing_column_raises(self, sample_raw_df):
        """Should raise ValueError when a required column is absent."""
        df = sample_raw_df.drop(columns=["salary"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df)

    def test_duplicate_ids_raise(self, sample_raw_df):
        """Should raise ValueError when employee_id has duplicates."""
        df = sample_raw_df.copy()
        df.loc[1, "employee_id"] = df.loc[0, "employee_id"]
        with pytest.raises(ValueError, match="duplicate employee IDs"):
            validate_dataframe(df)

    def test_invalid_target_values_raise(self, sample_raw_df):
        """Should raise ValueError when target is not binary {0,1}."""
        df = sample_raw_df.copy()
        df.loc[0, "enrolled"] = 2
        with pytest.raises(ValueError, match="invalid values"):
            validate_dataframe(df)

    def test_null_values_raise(self, sample_raw_df):
        """Should raise ValueError when nulls are present in features."""
        df = sample_raw_df.copy()
        df.loc[0, "salary"] = None
        with pytest.raises(ValueError, match="Null values"):
            validate_dataframe(df)

    def test_negative_salary_raises(self, sample_raw_df):
        """Should raise ValueError for negative salary."""
        df = sample_raw_df.copy()
        df.loc[0, "salary"] = -1000
        with pytest.raises(ValueError, match="Negative salary"):
            validate_dataframe(df)

    def test_negative_tenure_raises(self, sample_raw_df):
        """Should raise ValueError for negative tenure."""
        df = sample_raw_df.copy()
        df.loc[0, "tenure_years"] = -0.5
        with pytest.raises(ValueError, match="Negative tenure"):
            validate_dataframe(df)

    def test_age_out_of_range_raises(self, sample_raw_df):
        """Should raise ValueError for implausible age."""
        df = sample_raw_df.copy()
        df.loc[0, "age"] = 150
        with pytest.raises(ValueError, match="Age values out of plausible range"):
            validate_dataframe(df)


class TestSplitData:
    """Tests for the split_data function."""

    def test_split_sizes(self, sample_raw_df):
        """Train/test sizes should match the configured ratio."""
        train, test = split_data(sample_raw_df, test_size=0.2)
        assert len(train) + len(test) == len(sample_raw_df)
        assert len(test) == pytest.approx(len(sample_raw_df) * 0.2, abs=1)

    def test_stratification_preserved(self, sample_raw_df):
        """Target distribution should be preserved in both splits."""
        train, test = split_data(sample_raw_df, test_size=0.2)
        original_rate = sample_raw_df["enrolled"].mean()
        train_rate = train["enrolled"].mean()
        test_rate = test["enrolled"].mean()
        # Allow tolerance due to small sample
        assert abs(train_rate - original_rate) < 0.15
        assert abs(test_rate - original_rate) < 0.3  # Wider tolerance for 2-sample test set

    def test_no_data_leakage(self, sample_raw_df):
        """Train and test should have no overlapping employee_ids."""
        train, test = split_data(sample_raw_df)
        overlap = set(train["employee_id"]) & set(test["employee_id"])
        assert len(overlap) == 0


class TestGetFeaturesAndTarget:
    """Tests for get_features_and_target."""

    def test_returns_correct_shapes(self, sample_raw_df):
        """X should exclude employee_id and target; y should be 1D."""
        X, y = get_features_and_target(sample_raw_df)
        assert "employee_id" not in X.columns
        assert "enrolled" not in X.columns
        assert len(y) == len(sample_raw_df)

    def test_target_values_binary(self, sample_raw_df):
        """Target should only contain 0 and 1."""
        _, y = get_features_and_target(sample_raw_df)
        assert set(y.unique()) <= {0, 1}

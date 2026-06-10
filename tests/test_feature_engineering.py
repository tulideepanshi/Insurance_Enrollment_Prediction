"""
Tests for src.features.feature_engineering module.

Covers: SafeEncoder drift detection, OutlierCapper, CategoricalDtypeTransformer,
and all three model-specific pipelines (LR, RF, boosting).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    CategoricalDtypeTransformer,
    OutlierCapper,
    SafeEncoder,
    build_feature_pipeline,
)


class TestSafeEncoder:
    """Tests for the SafeEncoder transformer."""

    def test_known_values_unchanged(self, sample_features_df):
        """Known categorical values should pass through untouched."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)
        result = encoder.transform(sample_features_df)
        pd.testing.assert_frame_equal(result, sample_features_df)

    def test_unknown_mapped_to_token(self, sample_features_df):
        """Unseen values should be replaced with '__unknown__'."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)

        new_data = sample_features_df.iloc[[0]].copy()
        new_data["region"] = "Mars"
        result = encoder.transform(new_data)
        assert result["region"].iloc[0] == "__unknown__"

    def test_drift_warning_logged(self, sample_features_df, caplog):
        """Unseen values should trigger a DRIFT ALERT log warning."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)

        new_data = sample_features_df.iloc[[0]].copy()
        new_data["gender"] = "Alien"

        with caplog.at_level(logging.WARNING):
            encoder.transform(new_data)

        assert any("DRIFT ALERT" in msg for msg in caplog.messages)
        assert any("Alien" in msg for msg in caplog.messages)

    def test_multiple_unknowns_all_mapped(self, sample_features_df):
        """Multiple unknown values across different columns should all be caught."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)

        new_data = sample_features_df.iloc[[0]].copy()
        new_data["region"] = "Mars"
        new_data["gender"] = "Robot"
        result = encoder.transform(new_data)
        assert result["region"].iloc[0] == "__unknown__"
        assert result["gender"].iloc[0] == "__unknown__"

    def test_unknown_token_in_vocabulary(self, sample_features_df):
        """__unknown__ should be part of the learned vocabulary."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)
        for col, vocab in encoder.vocabularies_.items():
            assert "__unknown__" in vocab

    def test_get_known_categories(self, sample_features_df):
        """get_known_categories should return sorted lists per column."""
        encoder = SafeEncoder()
        encoder.fit(sample_features_df)
        cats = encoder.get_known_categories()
        assert isinstance(cats, dict)
        for col, values in cats.items():
            assert isinstance(values, list)
            assert values == sorted(values)


class TestOutlierCapper:
    """Tests for the OutlierCapper transformer."""

    def test_caps_upper_outliers(self):
        """Values above Q3 + 1.5*IQR should be clipped."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        capper = OutlierCapper(columns=["val"])
        capper.fit(df)
        result = capper.transform(df)
        assert result["val"].max() <= capper.bounds_["val"][1]

    def test_caps_lower_outliers(self):
        """Values below Q1 - 1.5*IQR should be clipped."""
        df = pd.DataFrame({"val": [-100, 1, 2, 3, 4, 5]})
        capper = OutlierCapper(columns=["val"])
        capper.fit(df)
        result = capper.transform(df)
        assert result["val"].min() >= capper.bounds_["val"][0]

    def test_no_modification_within_bounds(self):
        """Values within IQR bounds should not change."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        capper = OutlierCapper(columns=["val"])
        capper.fit(df)
        result = capper.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fitted_bounds_persisted(self):
        """Bounds learned during fit should be applied to new data."""
        train = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
        test = pd.DataFrame({"val": [0, 200]})
        capper = OutlierCapper(columns=["val"])
        capper.fit(train)
        result = capper.transform(test)
        assert result["val"].min() >= capper.bounds_["val"][0]
        assert result["val"].max() <= capper.bounds_["val"][1]


class TestCategoricalDtypeTransformer:
    """Tests for the CategoricalDtypeTransformer."""

    def test_converts_to_categorical(self, sample_features_df):
        """String columns should become pandas Categorical dtype."""
        transformer = CategoricalDtypeTransformer(columns=["gender", "region"])
        transformer.fit(sample_features_df)
        result = transformer.transform(sample_features_df)
        assert result["gender"].dtype.name == "category"
        assert result["region"].dtype.name == "category"

    def test_numerics_unchanged(self, sample_features_df):
        """Numeric columns should not be affected."""
        transformer = CategoricalDtypeTransformer(columns=["gender"])
        transformer.fit(sample_features_df)
        result = transformer.transform(sample_features_df)
        np.testing.assert_array_equal(result["age"].values, sample_features_df["age"].values)


class TestLRPipeline:
    """Tests for the Logistic Regression feature pipeline."""

    def test_fits_and_transforms(self, sample_features_df, sample_target):
        """LR pipeline should produce a numeric array with no NaNs."""
        pipeline = build_feature_pipeline("logistic_regression")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        assert result.shape[0] == len(sample_features_df)
        assert not np.isnan(result).any()

    def test_output_is_scaled(self, sample_features_df, sample_target):
        """LR pipeline output should be approximately standardized."""
        pipeline = build_feature_pipeline("logistic_regression")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        # Mean should be near 0 for numeric columns (first few cols are spline basis)
        col_means = np.abs(result.mean(axis=0))
        assert col_means.mean() < 1.0  # Rough check that scaling happened

    def test_handles_unknowns_via_safe_encoder(self, sample_features_df, sample_target):
        """LR pipeline should handle unseen categories through SafeEncoder."""
        pipeline = build_feature_pipeline("logistic_regression")
        pipeline.fit(sample_features_df, sample_target)

        new_data = sample_features_df.iloc[[0]].copy()
        new_data["region"] = "Unknown_Region"
        result = pipeline.transform(new_data)
        assert result.shape[0] == 1
        assert not np.isnan(result).any()


class TestRFPipeline:
    """Tests for the Random Forest feature pipeline."""

    def test_fits_and_transforms(self, sample_features_df, sample_target):
        """RF pipeline should produce a numeric array."""
        pipeline = build_feature_pipeline("random_forest")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        assert result.shape[0] == len(sample_features_df)

    def test_no_scaling_applied(self, sample_features_df, sample_target):
        """RF pipeline should not scale numeric features."""
        pipeline = build_feature_pipeline("random_forest")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        # Age values should be preserved (first numeric column)
        # The ColumnTransformer puts numerics first via "passthrough"
        original_ages = sample_features_df["age"].values
        # Result is a numpy array; first 3 columns are numerics
        np.testing.assert_array_almost_equal(result[:, 0], original_ages)

    def test_categoricals_are_ordinal_encoded(self, sample_features_df, sample_target):
        """Categorical columns should be integers (ordinal encoded)."""
        pipeline = build_feature_pipeline("random_forest")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        # Columns after the 3 numerics are ordinal-encoded categoricals
        cat_cols = result[:, len(["age", "salary", "tenure_years"]):]
        # All values should be integers (or -1 for unknown)
        assert np.all(cat_cols == cat_cols.astype(int))


class TestBoostingPipeline:
    """Tests for the XGBoost/LightGBM feature pipeline."""

    def test_fits_and_transforms(self, sample_features_df, sample_target):
        """Boosting pipeline should return a DataFrame with Categorical dtypes."""
        pipeline = build_feature_pipeline("xgboost")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_features_df)

    def test_categoricals_are_categorical_dtype(self, sample_features_df, sample_target):
        """Categorical columns should have pandas Categorical dtype."""
        pipeline = build_feature_pipeline("xgboost")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        for col in ["gender", "marital_status", "employment_type", "region", "has_dependents"]:
            assert result[col].dtype.name == "category", f"{col} is not Categorical"

    def test_numerics_are_raw(self, sample_features_df, sample_target):
        """Numeric columns should be unmodified."""
        pipeline = build_feature_pipeline("xgboost")
        result = pipeline.fit_transform(sample_features_df, sample_target)
        np.testing.assert_array_almost_equal(
            result["age"].values, sample_features_df["age"].values
        )

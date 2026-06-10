"""
Tests for src.models.trainer module.

Covers: model instantiation, model-specific pipeline integration,
fit/predict, evaluation metrics, and serialization round-trip.
"""

import tempfile

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import build_feature_pipeline
from src.models.trainer import evaluate_model, get_candidate_models


class TestGetCandidateModels:
    """Tests for model instantiation."""

    def test_returns_expected_models(self):
        """Should return all 4 candidate model families."""
        models = get_candidate_models()
        expected = {"logistic_regression", "random_forest", "xgboost", "lightgbm"}
        assert set(models.keys()) == expected

    def test_all_models_have_fit_predict(self):
        """Every model should implement fit() and predict()."""
        models = get_candidate_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba()"

    def test_xgboost_has_native_categorical(self):
        """XGBoost should have enable_categorical=True."""
        models = get_candidate_models()
        xgb = models["xgboost"]
        assert xgb.get_params()["enable_categorical"] is True


class TestModelPipeline:
    """Tests for the full pipeline (model-specific features + classifier)."""

    @pytest.fixture(params=["logistic_regression", "random_forest", "xgboost", "lightgbm"])
    def model_name(self, request):
        return request.param

    def test_pipeline_fit_predict(self, sample_features_df, sample_target, model_name):
        """Pipeline should fit and produce predictions of correct length."""
        model = get_candidate_models(sample_target)[model_name]
        pipeline = Pipeline([
            ("features", build_feature_pipeline(model_name)),
            ("classifier", model),
        ])
        pipeline.fit(sample_features_df, sample_target)
        preds = pipeline.predict(sample_features_df)
        assert len(preds) == len(sample_target)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_pipeline_predict_proba(self, sample_features_df, sample_target, model_name):
        """predict_proba should return probabilities in [0, 1]."""
        model = get_candidate_models(sample_target)[model_name]
        pipeline = Pipeline([
            ("features", build_feature_pipeline(model_name)),
            ("classifier", model),
        ])
        pipeline.fit(sample_features_df, sample_target)
        proba = pipeline.predict_proba(sample_features_df)
        assert proba.shape == (len(sample_target), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_expected_metrics(self, sample_features_df, sample_target):
        """Should return dict with accuracy, precision, recall, f1, roc_auc."""
        model = get_candidate_models(sample_target)["logistic_regression"]
        pipeline = Pipeline([
            ("features", build_feature_pipeline("logistic_regression")),
            ("classifier", model),
        ])
        pipeline.fit(sample_features_df, sample_target)
        metrics = evaluate_model(pipeline, sample_features_df, sample_target)
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(metrics.keys()) == expected_keys

    def test_metric_values_in_range(self, sample_features_df, sample_target):
        """All metrics should be in [0, 1]."""
        model = get_candidate_models(sample_target)["logistic_regression"]
        pipeline = Pipeline([
            ("features", build_feature_pipeline("logistic_regression")),
            ("classifier", model),
        ])
        pipeline.fit(sample_features_df, sample_target)
        metrics = evaluate_model(pipeline, sample_features_df, sample_target)
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name}={value} is out of [0,1]"


class TestModelSerialization:
    """Tests for model save/load round-trip."""

    @pytest.fixture(params=["logistic_regression", "random_forest", "xgboost", "lightgbm"])
    def model_name(self, request):
        return request.param

    def test_joblib_round_trip(self, sample_features_df, sample_target, model_name):
        """Model should produce identical predictions after save/load."""
        model = get_candidate_models(sample_target)[model_name]
        pipeline = Pipeline([
            ("features", build_feature_pipeline(model_name)),
            ("classifier", model),
        ])
        pipeline.fit(sample_features_df, sample_target)
        original_preds = pipeline.predict_proba(sample_features_df)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(pipeline, f.name)
            loaded = joblib.load(f.name)

        loaded_preds = loaded.predict_proba(sample_features_df)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

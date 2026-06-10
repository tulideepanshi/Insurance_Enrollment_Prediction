"""
Feature engineering — model-family-specific pipelines.

Three distinct pipelines, each respecting its model's assumptions:

1. LR pipeline: OutlierCapper → SplineTransformer (quantile knots) →
   TargetEncoder (cross-fitted, smoothed) → StandardScaler + SafeEncoder
   for drift alerting. TargetEncoder replaces OneHotEncoder: one numeric
   column per categorical (no multicollinearity), handles unknowns
   natively (global mean fallback), prevents leakage via internal CV.

2. RF pipeline: OrdinalEncoder + SafeEncoder. No scaling, no outlier
   capping. Raw numerics passed through — trees are scale-invariant and
   handle outliers via threshold splits.

3. XGBoost/LightGBM pipeline: CategoricalDtypeTransformer (converts
   string columns to pandas Categorical for native handling) + SafeEncoder.
   No scaling, no capping. Models use their built-in categorical split
   algorithms which evaluate optimal category groupings directly.

All three share the SafeEncoder, which maps unseen categories to an
explicit "__unknown__" token and emits a drift warning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "__unknown__"


# ──────────────────────────────────────────────
# Shared transformers
# ──────────────────────────────────────────────


class SafeEncoder(BaseEstimator, TransformerMixin):
    """Maps unseen categorical values to an explicit '__unknown__' token.

    During fit(): learns the vocabulary (set of known values) for each
    categorical column.

    During transform(): replaces any value not in the vocabulary with
    '__unknown__' and emits a structured drift warning.

    This replaces handle_unknown="ignore" — instead of silently zeroing
    out unknown categories, we give the model an explicit signal and
    alert the operator.
    """

    def __init__(self, columns: List[str] | None = None):
        self.columns = columns
        self.vocabularies_: Dict[str, Set[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn known categories from training data."""
        cols = self.columns or X.select_dtypes(include=["object"]).columns.tolist()
        for col in cols:
            # Add __unknown__ to vocabulary so downstream encoders see it
            self.vocabularies_[col] = set(X[col].unique()) | {UNKNOWN_TOKEN}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace unseen values with '__unknown__' and log drift warnings."""
        X = X.copy()
        for col, vocab in self.vocabularies_.items():
            if col not in X.columns:
                continue
            known = vocab - {UNKNOWN_TOKEN}
            mask = ~X[col].isin(known)
            if mask.any():
                unseen_values = X.loc[mask, col].unique().tolist()
                logger.warning(
                    "DRIFT ALERT — column '%s' has unseen values: %s. "
                    "Mapped to '%s'. Consider retraining.",
                    col, unseen_values, UNKNOWN_TOKEN,
                )
                X.loc[mask, col] = UNKNOWN_TOKEN
        return X

    def get_known_categories(self) -> Dict[str, List[str]]:
        """Export vocabularies for API-side drift detection."""
        return {col: sorted(vocab) for col, vocab in self.vocabularies_.items()}

    def save_known_categories(self, path: Path) -> None:
        """Save known categories to JSON for use by the serving layer."""
        with open(path, "w") as f:
            json.dump(self.get_known_categories(), f, indent=2)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numeric outliers at IQR-based bounds.

    Used ONLY in the LR pipeline — tree models benefit from seeing
    the full range of values.

    Fitted on training data to learn bounds; applied consistently
    on train and test to prevent data leakage.
    """

    def __init__(self, factor: float = 1.5, columns: List[str] | None = None):
        self.factor = factor
        self.columns = columns
        self.bounds_: Dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn IQR-based bounds from training data."""
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip values to learned bounds."""
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X


class CategoricalDtypeTransformer(BaseEstimator, TransformerMixin):
    """Converts string columns to pandas Categorical dtype.

    Used in the XGBoost/LightGBM pipeline so these models can use
    their native categorical split algorithms instead of one-hot or
    ordinal encoding. Native handling evaluates optimal category
    groupings directly, which is more efficient than encoding.
    """

    def __init__(self, columns: List[str] | None = None):
        self.columns = columns
        self.categories_: Dict[str, list] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn category sets from training data."""
        cols = self.columns or X.select_dtypes(include=["object"]).columns.tolist()
        for col in cols:
            self.categories_[col] = list(X[col].unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to pandas Categorical dtype."""
        X = X.copy()
        for col, cats in self.categories_.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats)
        return X


# ──────────────────────────────────────────────
# Pipeline builders
# ──────────────────────────────────────────────


def build_lr_pipeline() -> Pipeline:
    """Feature pipeline for Logistic Regression.

    Design rationale:
    - OutlierCapper: LR is sensitive to high-leverage points. Extreme
      salary/tenure values can disproportionately pull the decision boundary.
    - SplineTransformer with quantile knots: Captures non-linear effects
      (e.g., the age step function at 30) without manual binning. Quantile
      knots place more basis functions where data is dense.
    - TargetEncoder with cross-fitting: Replaces OneHotEncoder. Each
      categorical becomes a single numeric column (smoothed mean target).
      Eliminates multicollinearity (no drop="first" needed), handles
      unknowns natively (fallback to global mean), and doesn't expand
      dimensionality. Internal 5-fold cross-fitting prevents target leakage.
    - StandardScaler (post-ColumnTransformer): Scales all features uniformly
      for fair L2 regularization — spline basis outputs and target-encoded
      values are on different scales.
    - No derived features (bins, interactions): Splines handle non-linearity;
      adding bins/interactions would reintroduce the collinearity problem.
    """
    numeric_pipeline = Pipeline([
        ("splines", SplineTransformer(
            n_knots=5,
            degree=3,
            knots="quantile",  # Place knots at data percentiles
            include_bias=False,  # Avoid redundancy with intercept
        )),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", TargetEncoder(
            smooth="auto",          # Empirical Bayes smoothing — low-count
            # categories shrink toward the global mean
            cv=5,                   # Internal cross-fitting to prevent target leakage
            random_state=RANDOM_STATE,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([
        ("outlier_capper", OutlierCapper(columns=["salary", "tenure_years"])),
        ("safe_encoder", SafeEncoder(columns=CATEGORICAL_FEATURES)),
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),  # Uniform scaling across all features
    ])


def build_rf_pipeline() -> Pipeline:
    """Feature pipeline for Random Forest.

    Design rationale:
    - No OutlierCapper: Trees split on thresholds — an extreme value
      just becomes "salary > 105,000?" which is a perfectly valid split.
      Capping destroys real signal at the tails.
    - No StandardScaler: Trees are scale-invariant. The split
      "salary > 65000?" is identical whether salary is raw or z-scored.
    - OrdinalEncoder: Maps categories to integers. RF splits on thresholds
      ("region <= 1?") and can recover any partition through multiple splits.
      Avoids the feature-importance dilution of one-hot across k-1 columns.
    - No drop="first": Unlike LR, RF has no multicollinearity assumption.
    """
    categorical_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,  # SafeEncoder maps unknowns to __unknown__ first
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([
        ("safe_encoder", SafeEncoder(columns=CATEGORICAL_FEATURES)),
        ("preprocessor", preprocessor),
    ])


def build_boosting_pipeline() -> Pipeline:
    """Feature pipeline for XGBoost and LightGBM.

    Design rationale:
    - No OutlierCapper or StandardScaler: Same reasoning as RF.
    - CategoricalDtypeTransformer: Converts strings to pandas Categorical
      dtype so XGBoost (enable_categorical=True) and LightGBM
      (categorical_feature="auto") can use their native split algorithms.
      Native handling evaluates all possible category groupings at each
      split, which is strictly better than ordinal encoding (which imposes
      an arbitrary order that may require extra splits to undo).
    - Numerics pass through raw.
    """
    return Pipeline([
        ("safe_encoder", SafeEncoder(columns=CATEGORICAL_FEATURES)),
        ("categorical_dtype", CategoricalDtypeTransformer(columns=CATEGORICAL_FEATURES)),
    ])


# ──────────────────────────────────────────────
# Pipeline registry
# ──────────────────────────────────────────────

# Maps model family names to their pipeline builder
PIPELINE_REGISTRY = {
    "logistic_regression": build_lr_pipeline,
    "random_forest": build_rf_pipeline,
    "xgboost": build_boosting_pipeline,
    "lightgbm": build_boosting_pipeline,
}


def build_feature_pipeline(model_name: str) -> Pipeline:
    """Build the correct feature pipeline for a given model family.

    Args:
        model_name: Key matching get_candidate_models() in trainer.py.

    Returns:
        sklearn Pipeline tailored to that model's assumptions.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    builder = PIPELINE_REGISTRY.get(model_name)
    if builder is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(PIPELINE_REGISTRY.keys())}"
        )
    return builder()

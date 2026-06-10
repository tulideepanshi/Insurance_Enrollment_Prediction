"""
Post-training analysis: SHAP explanations + calibration diagnostics.

Loads the trained model and test data, then generates:
1. SHAP summary plot (global feature importance)
2. SHAP dependence plots (top 5 features)
3. Bivariate calibration: predicted vs actual enrollment rate per feature
4. Enrollment rate curves by numeric features (age, salary, tenure)

Auto-selects the SHAP explainer based on model type:
- Tree models → TreeExplainer (exact, fast)
- Linear models → LinearExplainer (exact, coefficient-based)

Usage:
    python analyze.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.data.data_loader import get_features_and_target, load_raw_data, split_data, validate_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SHAP_DIR = FIGURES_DIR / "shap"
CALIBRATION_DIR = FIGURES_DIR / "calibration"


def _load_model_and_data():
    """Load trained pipeline and reproduce the exact train/test split."""
    pipeline = joblib.load(MODEL_DIR / "best_model.joblib")

    # Load model info to know which type we're working with
    model_info_path = MODEL_DIR / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            model_info = json.load(f)
        model_name = model_info["model_name"]
    else:
        # Fallback: infer from classifier type
        classifier = pipeline.named_steps["classifier"]
        type_name = type(classifier).__name__.lower()
        if "logistic" in type_name:
            model_name = "logistic_regression"
        elif "xgb" in type_name:
            model_name = "xgboost"
        elif "lgbm" in type_name or "lightgbm" in type_name:
            model_name = "lightgbm"
        else:
            model_name = "random_forest"

    # Reproduce the same split used during training
    raw_df = load_raw_data()
    validate_dataframe(raw_df)
    _, test_df = split_data(raw_df)
    X_test, y_test = get_features_and_target(test_df)

    return pipeline, X_test, y_test, model_name


def _get_shap_explainer(pipeline, X_test, model_name):
    """Create the appropriate SHAP explainer for the model type.

    - Tree models: TreeExplainer (exact, O(TLD) per sample)
    - Linear models: LinearExplainer (exact, coefficient-based)
    """
    classifier = pipeline.named_steps["classifier"]
    feature_pipeline = pipeline.named_steps["features"]

    # Transform features so SHAP sees what the model sees
    X_transformed = feature_pipeline.transform(X_test)

    if model_name == "logistic_regression":
        # LinearExplainer needs the transformed data as background
        explainer = shap.LinearExplainer(classifier, X_transformed)
    elif model_name in ("xgboost", "lightgbm"):
        # TreeExplainer works directly on the tree model
        explainer = shap.TreeExplainer(classifier)
    elif model_name == "random_forest":
        explainer = shap.TreeExplainer(classifier)
    else:
        logger.warning("Unknown model type '%s', falling back to KernelExplainer", model_name)
        explainer = shap.KernelExplainer(classifier.predict_proba, X_transformed[:100])

    return explainer, X_transformed


def _get_feature_names(pipeline, X_test, model_name):
    """Extract feature names after transformation.

    Different pipelines produce different feature sets:
    - LR: spline basis columns + one-hot encoded categoricals
    - RF: raw numerics + ordinal-encoded categoricals
    - Boosting: raw numerics + categorical dtype columns
    """
    feature_pipeline = pipeline.named_steps["features"]

    if model_name == "logistic_regression":
        # ColumnTransformer provides feature names
        try:
            preprocessor = feature_pipeline.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            return [str(n) for n in names]
        except Exception:
            pass

    # For tree models, features are the original column names
    if model_name in ("xgboost", "lightgbm"):
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES
    elif model_name == "random_forest":
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

    return None


def _extract_shap_values_positive_class(raw_shap_values):
    """Extract SHAP values for the positive class from various return formats.

    TreeExplainer returns different shapes depending on the SHAP version:
    - List of 2 arrays, each (n_samples, n_features): old API, one per class
    - 3D ndarray (n_samples, n_features, 2): newer API, last dim is classes
    - 2D ndarray (n_samples, n_features): already single-class (LinearExplainer)
    """
    if isinstance(raw_shap_values, list):
        return raw_shap_values[1]
    if isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        return raw_shap_values[:, :, 1]
    return raw_shap_values


def plot_shap_summary(explainer, X_transformed, feature_names, model_name):
    """Generate SHAP summary (beeswarm) plot — global feature importance."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    raw_shap = explainer.shap_values(X_transformed)
    shap_values = _extract_shap_values_positive_class(raw_shap)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved")

    return shap_values


def plot_shap_dependence(explainer, X_transformed, shap_values, feature_names, model_name):
    """Generate SHAP dependence plots for top 5 features."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Find top 5 features by mean absolute SHAP value
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-5:][::-1]

    for idx in top_indices:
        idx = int(idx)  # numpy int → Python int (lists can't index with np scalars)
        feat_name = feature_names[idx] if feature_names else f"feature_{idx}"

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            idx,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False,
            ax=ax,
        )
        plt.title(f"SHAP Dependence — {feat_name}")
        plt.tight_layout()
        safe_name = str(feat_name).replace("/", "_").replace(" ", "_")
        plt.savefig(SHAP_DIR / f"shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("SHAP dependence plots saved for top 5 features")


def plot_calibration_by_feature(pipeline, X_test, y_test):
    """Bivariate calibration: predicted vs actual enrollment rate per feature.

    For each feature, bins the values, computes mean predicted probability
    and mean actual enrollment rate per bin, and overlays them. Shows where
    the model is well-calibrated vs where predictions diverge from reality.
    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Numeric features: use quantile bins
    for col in NUMERIC_FEATURES:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create 10 quantile bins
        bins = pd.qcut(X_test[col], q=10, duplicates="drop")
        grouped = pd.DataFrame({
            "bin": bins,
            "actual": y_test.values,
            "predicted": y_proba,
        }).groupby("bin", observed=True).agg(
            actual_rate=("actual", "mean"),
            predicted_rate=("predicted", "mean"),
            count=("actual", "count"),
        )

        x = range(len(grouped))
        width = 0.35
        ax.bar([i - width / 2 for i in x], grouped["actual_rate"], width,
               label="Actual Rate", color="#3b82f6", alpha=0.8)
        ax.bar([i + width / 2 for i in x], grouped["predicted_rate"], width,
               label="Predicted Rate", color="#f97316", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(b) for b in grouped.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Enrollment Rate")
        ax.set_xlabel(col)
        ax.set_title(f"Calibration: Predicted vs Actual — {col}")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(CALIBRATION_DIR / f"calibration_{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Categorical features: one bar group per category value
    for col in CATEGORICAL_FEATURES:
        fig, ax = plt.subplots(figsize=(8, 5))

        grouped = pd.DataFrame({
            "category": X_test[col].values,
            "actual": y_test.values,
            "predicted": y_proba,
        }).groupby("category").agg(
            actual_rate=("actual", "mean"),
            predicted_rate=("predicted", "mean"),
            count=("actual", "count"),
        ).sort_values("actual_rate", ascending=False)

        x = range(len(grouped))
        width = 0.35
        ax.bar([i - width / 2 for i in x], grouped["actual_rate"], width,
               label="Actual Rate", color="#3b82f6", alpha=0.8)
        ax.bar([i + width / 2 for i in x], grouped["predicted_rate"], width,
               label="Predicted Rate", color="#f97316", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right")
        ax.set_ylabel("Enrollment Rate")
        ax.set_xlabel(col)
        ax.set_title(f"Calibration: Predicted vs Actual — {col}")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(CALIBRATION_DIR / f"calibration_{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("Calibration plots saved for all features")


def plot_top4_calibration(pipeline, X_test, y_test, shap_values, feature_names):
    """Combined 2x2 calibration plot for the top 4 features by SHAP importance.

    Identifies the 4 most important features from SHAP values, then renders
    predicted vs actual enrollment rate for each in a single figure.
    Automatically handles both numeric (quantile bins) and categorical features.
    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Identify top 4 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top4_indices = np.argsort(mean_abs)[-4:][::-1]

    # Map indices back to original feature names
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if feature_names and len(feature_names) == len(mean_abs):
        # Transformed feature names — map back to originals
        top_original = []
        for idx in top4_indices:
            transformed_name = feature_names[int(idx)]
            matched = None
            for orig in all_features:
                if orig in str(transformed_name):
                    matched = orig
                    break
            if matched and matched not in top_original:
                top_original.append(matched)
        # Fill up to 4 if some transformed features mapped to the same original
        for orig in all_features:
            if len(top_original) >= 4:
                break
            if orig not in top_original:
                top_original.append(orig)
        top4_features = top_original[:4]
    else:
        top4_features = all_features[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col in zip(axes.flat, top4_features):
        if col in NUMERIC_FEATURES:
            bins = pd.qcut(X_test[col], q=10, duplicates="drop")
            grouped = pd.DataFrame({
                "bin": bins, "actual": y_test.values, "predicted": y_proba,
            }).groupby("bin", observed=True).agg(
                actual_rate=("actual", "mean"),
                predicted_rate=("predicted", "mean"),
            )
            x = range(len(grouped))
            labels = [str(b) for b in grouped.index]
        else:
            grouped = pd.DataFrame({
                "category": X_test[col].values, "actual": y_test.values, "predicted": y_proba,
            }).groupby("category").agg(
                actual_rate=("actual", "mean"),
                predicted_rate=("predicted", "mean"),
            ).sort_values("actual_rate", ascending=False)
            x = range(len(grouped))
            labels = list(grouped.index)

        width = 0.35
        ax.bar([i - width / 2 for i in x], grouped["actual_rate"], width,
               label="Actual", color="#3b82f6", alpha=0.85)
        ax.bar([i + width / 2 for i in x], grouped["predicted_rate"], width,
               label="Predicted", color="#f97316", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Enrollment Rate")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Calibration — Top 4 Features by SHAP Importance", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(CALIBRATION_DIR / "calibration_top4_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Combined top-4 calibration plot saved")


def main():
    """Run full post-training analysis."""
    logger.info("=" * 60)
    logger.info("POST-TRAINING ANALYSIS")
    logger.info("=" * 60)

    pipeline, X_test, y_test, model_name = _load_model_and_data()
    logger.info("Model type: %s, test set: %d samples", model_name, len(X_test))

    # 1. SHAP analysis
    logger.info("Computing SHAP values...")
    explainer, X_transformed = _get_shap_explainer(pipeline, X_test, model_name)
    feature_names = _get_feature_names(pipeline, X_test, model_name)

    shap_values = plot_shap_summary(explainer, X_transformed, feature_names, model_name)
    plot_shap_dependence(explainer, X_transformed, shap_values, feature_names, model_name)

    # 2. Calibration plots — predicted vs actual per feature
    logger.info("Generating calibration plots...")
    plot_calibration_by_feature(pipeline, X_test, y_test)

    # 3. Combined top-4 calibration plot
    logger.info("Generating combined top-4 calibration plot...")
    plot_top4_calibration(pipeline, X_test, y_test, shap_values, feature_names)

    logger.info("=" * 60)
    logger.info("Analysis complete. Figures saved to:")
    logger.info("  SHAP:        %s", SHAP_DIR)
    logger.info("  Calibration: %s", CALIBRATION_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

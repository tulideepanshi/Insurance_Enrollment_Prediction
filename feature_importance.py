"""
Feature importance analysis across all 4 model families.

Trains each model with default parameters, extracts the native
importance metric, maps transformed feature names back to original
columns, normalizes to [0, 1], and generates horizontal bar plots.

Importance metrics:
- Logistic Regression: aggregated |coefficient| across spline/one-hot columns
- Random Forest: Gini impurity (mean decrease in impurity)
- XGBoost: gain (total loss reduction from splits on this feature)
- LightGBM: split count (number of times a feature is used)

Usage:
    python feature_importance.py
    make importance
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
)
from src.data.data_loader import load_and_prepare
from src.features.feature_engineering import build_feature_pipeline
from src.models.trainer import get_candidate_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

IMPORTANCE_DIR = FIGURES_DIR / "importance"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Display labels and colors per model
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression (|coef|)",
    "random_forest": "Random Forest (Gini impurity)",
    "xgboost": "XGBoost (gain)",
    "lightgbm": "LightGBM (gain)",
}
MODEL_SHORT = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGBM",
}
MODEL_COLORS = {
    "logistic_regression": "#3b82f6",
    "random_forest": "#10b981",
    "xgboost": "#f97316",
    "lightgbm": "#8b5cf6",
}


# ──────────────────────────────────────────────
# Importance extraction
# ──────────────────────────────────────────────


def _map_to_original_features(transformed_names, values) -> dict:
    """Sum importance values from transformed columns back to original features.

    Matches by substring: "num__scaler__age_sp_0" → "age",
    "cat__encoder__gender_Female" → "gender", etc.
    """
    importance = {f: 0.0 for f in ALL_FEATURES}
    for name, val in zip(transformed_names, values):
        for orig in ALL_FEATURES:
            if orig in str(name):
                importance[orig] += val
                break
    return importance


def _extract_lr_importance(pipeline) -> dict:
    """Aggregate |coefficient| from spline basis + one-hot columns back to original features."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["features"].named_steps["preprocessor"]
    coefs = np.abs(classifier.coef_[0])
    names = preprocessor.get_feature_names_out()
    return _map_to_original_features(names, coefs)


def _extract_rf_importance(pipeline) -> dict:
    """Map Gini impurity importance through ColumnTransformer back to original features."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["features"].named_steps["preprocessor"]
    importances = classifier.feature_importances_
    names = preprocessor.get_feature_names_out()
    return _map_to_original_features(names, importances)


def _extract_boosting_importance(pipeline, model_name: str) -> dict:
    """XGBoost/LightGBM with native categoricals — features map directly to original names.

    Both use gain-based importance for consistency:
    - XGBoost: importance_type="gain" (default with tree_method="hist")
    - LightGBM: booster_.feature_importance(importance_type="gain")
    """
    classifier = pipeline.named_steps["classifier"]
    if model_name == "lightgbm":
        # LightGBM defaults to split count; explicitly request gain
        importances = classifier.booster_.feature_importance(importance_type="gain")
    else:
        importances = classifier.feature_importances_
    return dict(zip(ALL_FEATURES, importances))


def extract_importance(pipeline, model_name: str) -> dict:
    """Dispatch to the correct extractor based on model family."""
    if model_name == "logistic_regression":
        return _extract_lr_importance(pipeline)
    elif model_name == "random_forest":
        return _extract_rf_importance(pipeline)
    else:
        return _extract_boosting_importance(pipeline, model_name)


def normalize_importance(importance: dict) -> dict:
    """Normalize importance values to sum to 1."""
    total = sum(importance.values())
    if total == 0:
        return importance
    return {k: v / total for k, v in importance.items()}


# ──────────────────────────────────────────────
# Training + extraction
# ──────────────────────────────────────────────


def train_all_and_extract(X_train, y_train) -> dict:
    """Train all 4 models with default params and extract importance from each.

    Returns:
        Dict of model_name -> {feature_name: normalized_importance}.
    """
    models = get_candidate_models(y_train)
    results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        pipeline = Pipeline([
            ("features", build_feature_pipeline(name)),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)

        raw = extract_importance(pipeline, name)
        results[name] = normalize_importance(raw)

        top3 = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info("  Top 3: %s", [(f, f"{v:.4f}") for f, v in top3])

    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────


def plot_per_model(results: dict) -> None:
    """One horizontal bar plot per model, features ranked by importance."""
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, importance in results.items():
        sorted_items = sorted(importance.items(), key=lambda x: x[1])
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, values, color=MODEL_COLORS[model_name], alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel("Normalized Importance", fontsize=10)
        ax.set_title(MODEL_LABELS[model_name], fontsize=12, fontweight="bold")

        # Value labels on bars
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)

        ax.set_xlim(0, max(values) * 1.15)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMPORTANCE_DIR / f"importance_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("Per-model importance plots saved")


def plot_comparison(results: dict) -> None:
    """Grouped horizontal bar chart comparing all 4 models side by side."""
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Sort features by mean importance across models (highest at top)
    df["_mean"] = df.mean(axis=1)
    df = df.sort_values("_mean", ascending=True)
    df = df.drop(columns=["_mean"])

    fig, ax = plt.subplots(figsize=(11, 6))
    n_models = len(df.columns)
    bar_h = 0.18
    y_pos = np.arange(len(df))

    for i, model_name in enumerate(df.columns):
        offset = (i - n_models / 2 + 0.5) * bar_h
        ax.barh(
            y_pos + offset, df[model_name],
            height=bar_h,
            label=MODEL_SHORT[model_name],
            color=MODEL_COLORS[model_name],
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlabel("Normalized Importance", fontsize=10)
    ax.set_title("Feature Importance — All Models Compared", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / "importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Comparison plot saved")


def plot_rank_table(results: dict) -> None:
    """Visual rank table: feature × model with rank numbers + color intensity."""
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    # Rank within each model (1 = most important)
    ranks = df.rank(ascending=False).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    # Build table data
    col_labels = [MODEL_SHORT[m] for m in df.columns]
    row_labels = list(df.index)
    cell_text = ranks.values.tolist()

    # Color cells by rank (lower rank = darker green)
    n_features = len(row_labels)
    cell_colors = []
    for row in ranks.values:
        row_colors = []
        for rank in row:
            intensity = 1.0 - (rank - 1) / (n_features - 1) * 0.7
            row_colors.append((0.06, intensity * 0.7 + 0.3, 0.06 + intensity * 0.3, 0.3))
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    ax.set_title("Feature Importance Rank (1 = most important)", fontsize=12,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / "importance_rank_table.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Rank table saved")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    _, X_train, y_train, _, _ = load_and_prepare()
    results = train_all_and_extract(X_train, y_train)

    plot_per_model(results)
    plot_comparison(results)
    plot_rank_table(results)

    # Save raw values as CSV
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.index.name = "feature"
    df["mean"] = df.mean(axis=1)
    df = df.sort_values("mean", ascending=False)
    df.to_csv(IMPORTANCE_DIR / "importance_scores.csv", float_format="%.4f")

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (normalized, sorted by mean)")
    print("=" * 60)
    print(df.rename(columns=MODEL_SHORT).to_string(float_format="{:.4f}".format))

    print(f"\nFigures saved to: {IMPORTANCE_DIR}/")
    print("  - importance_<model>.png    (per-model bar plots)")
    print("  - importance_comparison.png (all models side by side)")
    print("  - importance_rank_table.png (rank heatmap)")
    print("  - importance_scores.csv     (raw values)")


if __name__ == "__main__":
    main()

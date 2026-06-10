"""
Main training orchestrator.

Runs the full ML pipeline end-to-end:
1. Load and validate data
2. Stratified train/test split
3. Cross-validation comparison of 4 model families
4. Optuna hyperparameter tuning on the best model
5. Final training on full train set, evaluation on held-out test set
6. Save model artifact + MLflow logs

Usage:
    python train.py
    python train.py --skip-tuning  # Skip Optuna, use default hyperparameters
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import MODEL_DIR, PRIMARY_METRIC
from src.data.data_loader import load_and_prepare
from src.models.trainer import (
    evaluate_model,
    get_candidate_models,
    run_cv_comparison,
    train_final_model,
    tune_best_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(skip_tuning: bool = False) -> None:
    """Execute the full training pipeline."""

    # ── Step 1: Load & validate ──
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and validating data")
    logger.info("=" * 60)
    raw_df, X_train, y_train, X_test, y_test = load_and_prepare()

    logger.info("Features shape: train=%s, test=%s", X_train.shape, X_test.shape)

    # ── Step 2: Cross-validation comparison ──
    logger.info("=" * 60)
    logger.info("STEP 2: Cross-validation model comparison")
    logger.info("=" * 60)
    cv_results = run_cv_comparison(X_train, y_train)

    print("\n" + "=" * 60)
    print("CV RESULTS (sorted by %s):" % PRIMARY_METRIC)
    print("=" * 60)
    display_cols = ["model"] + [c for c in cv_results.columns if c != "model"]
    print(cv_results[display_cols].to_string(index=False, float_format="%.4f"))

    best_model_name = cv_results.iloc[0]["model"]
    logger.info("Best model family: %s", best_model_name)

    # ── Step 3: Hyperparameter tuning ──
    if skip_tuning:
        logger.info("Skipping Optuna tuning (--skip-tuning flag)")
        # Use known-safe default params per model family
        default_params = {
            "logistic_regression": {"C": 1.0, "solver": "lbfgs", "penalty": "l2"},
            "random_forest": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2,
                              "min_samples_leaf": 1, "max_features": "sqrt"},
            "xgboost": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                        "reg_alpha": 1e-5, "reg_lambda": 1.0, "gamma": 1e-5},
            "lightgbm": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1,
                         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
                         "reg_alpha": 1e-5, "reg_lambda": 1.0, "num_leaves": 31},
        }
        best_params = default_params[best_model_name]
    else:
        logger.info("=" * 60)
        logger.info("STEP 3: Optuna hyperparameter tuning for %s", best_model_name)
        logger.info("=" * 60)
        best_params, best_cv_score = tune_best_model(X_train, y_train, best_model_name)
        logger.info("Best CV %s: %.4f", PRIMARY_METRIC, best_cv_score)
        logger.info("Best params: %s", json.dumps(best_params, indent=2, default=str))

    # ── Step 4: Train final model ──
    logger.info("=" * 60)
    logger.info("STEP 4: Training final model")
    logger.info("=" * 60)
    pipeline, test_metrics = train_final_model(
        X_train, y_train, X_test, y_test,
        model_name=best_model_name,
        params=best_params,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"  {metric:>12s}: {value:.4f}")

    # Save CV results for report generation
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    with open(MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    logger.info("Training complete. Model saved to %s", MODEL_DIR / "best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance enrollment model")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna hyperparameter tuning, use defaults",
    )
    args = parser.parse_args()
    main(skip_tuning=args.skip_tuning)

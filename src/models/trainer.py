"""
Model training, cross-validation, hyperparameter tuning, and evaluation.

Orchestrates:
- Stratified 5-fold CV comparison across 4 model families, each with
  its own feature pipeline (LR, RF, XGBoost/LightGBM)
- Optuna hyperparameter search on the best family
- MLflow experiment tracking for all runs
- Final model training, serialization, and known-category export
"""

import logging
import warnings
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    KNOWN_CATEGORIES_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    PRIMARY_METRIC,
    RANDOM_STATE,
)
from src.features.feature_engineering import build_feature_pipeline

logger = logging.getLogger(__name__)

# Suppress convergence warnings during Optuna trials
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────


def _compute_scale_pos_weight(y: pd.Series | None) -> float:
    """Compute XGBoost scale_pos_weight = n_negative / n_positive.

    Compensates for class imbalance by upweighting the minority class.
    Returns 1.0 if y is None (no reweighting).
    """
    if y is None:
        return 1.0
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def get_candidate_models(y_train: pd.Series | None = None) -> Dict[str, Any]:
    """Return a dict of model_name -> instantiated estimator.

    These are default configurations used for the initial CV comparison.
    The best family then gets Optuna tuning.

    Args:
        y_train: Training labels, used to compute class imbalance weight
                 for XGBoost. If None, scale_pos_weight defaults to 1.0.

    XGBoost and LightGBM are configured for native categorical support
    to work with the boosting pipeline (CategoricalDtypeTransformer).
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            scale_pos_weight=_compute_scale_pos_weight(y_train),
            eval_metric="logloss",
            enable_categorical=True,  # Native categorical support
            tree_method="hist",  # Required for enable_categorical
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            # categorical_feature is auto-detected from pandas Categorical dtype
            # (set by CategoricalDtypeTransformer). Passing it here triggers a warning.
            n_jobs=-1,
            verbose=-1,
        ),
    }


def _build_full_pipeline(model_name: str, model: Any) -> Pipeline:
    """Assemble feature pipeline + classifier into a single Pipeline.

    Selects the correct feature pipeline for the model family.
    """
    return Pipeline([
        ("features", build_feature_pipeline(model_name)),
        ("classifier", model),
    ])


# ──────────────────────────────────────────────
# Cross-validation comparison
# ──────────────────────────────────────────────

def run_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Run stratified k-fold CV for all candidate models.

    Each model is wrapped in its model-specific feature pipeline:
    - logistic_regression → LR pipeline (splines + one-hot + scaler)
    - random_forest → RF pipeline (ordinal encoding)
    - xgboost, lightgbm → Boosting pipeline (native categoricals)

    Results are logged to MLflow.

    Args:
        X_train: Training features (raw, pre-transformation).
        y_train: Training labels.

    Returns:
        DataFrame with mean/std CV scores for each model and metric.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    models = get_candidate_models(y_train)
    results = []

    for name, model in models.items():
        logger.info("Running %d-fold CV for %s...", CV_FOLDS, name)

        pipeline = _build_full_pipeline(name, model)

        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False,
        )

        row = {"model": name}
        with mlflow.start_run(run_name=f"cv_{name}"):
            for metric in scoring:
                key = f"test_{metric}"
                mean_val = cv_results[key].mean()
                std_val = cv_results[key].std()
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                mlflow.log_metric(f"cv_{metric}_mean", mean_val)
                mlflow.log_metric(f"cv_{metric}_std", std_val)

            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("pipeline_type", _pipeline_type_label(name))

        logger.info(
            "%s — ROC-AUC: %.4f (+/- %.4f), F1: %.4f (+/- %.4f)",
            name,
            row["roc_auc_mean"], row["roc_auc_std"],
            row["f1_mean"], row["f1_std"],
        )
        results.append(row)

    return pd.DataFrame(results).sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)


def _pipeline_type_label(model_name: str) -> str:
    """Human-readable label for the pipeline type (for MLflow logging)."""
    labels = {
        "logistic_regression": "lr_spline_targetenc_scaled",
        "random_forest": "rf_ordinal_raw",
        "xgboost": "xgb_native_categorical",
        "lightgbm": "lgbm_native_categorical",
    }
    return labels.get(model_name, "unknown")


# ──────────────────────────────────────────────
# Optuna hyperparameter tuning
# ──────────────────────────────────────────────

def _get_optuna_search_space(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Define per-model hyperparameter search space."""
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        }
    elif model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "penalty": "l2",
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _create_model_from_params(model_name: str, params: Dict[str, Any], y_train: pd.Series | None = None) -> Any:
    """Instantiate a model with given hyperparameters."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    common = {"random_state": RANDOM_STATE}

    if model_name == "xgboost":
        return XGBClassifier(
            **params, **common,
            scale_pos_weight=_compute_scale_pos_weight(y_train),
            eval_metric="logloss",
            enable_categorical=True,
            tree_method="hist",
            n_jobs=-1, verbosity=0,
        )
    elif model_name == "lightgbm":
        return LGBMClassifier(
            **params, **common,
            class_weight="balanced",
            n_jobs=-1, verbose=-1,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            **params, **common,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_name == "logistic_regression":
        return LogisticRegression(
            **params, **common,
            max_iter=1000,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> Tuple[Dict[str, Any], float]:
    """Run Optuna hyperparameter search for the given model family.

    Uses stratified k-fold CV as the objective to avoid overfitting
    to a single train/val split.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_name: Key from get_candidate_models().
        n_trials: Maximum Optuna trials.
        timeout: Maximum seconds for the search.

    Returns:
        Tuple of (best_params dict, best CV ROC-AUC score).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = _get_optuna_search_space(trial, model_name)
        model = _create_model_from_params(model_name, params, y_train)
        pipeline = _build_full_pipeline(model_name, model)

        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=PRIMARY_METRIC, n_jobs=-1,
        )
        return cv_results["test_score"].mean()

    study = optuna.create_study(
        direction="maximize",
        study_name=f"tune_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    # Log best trial to MLflow
    with mlflow.start_run(run_name=f"optuna_best_{model_name}"):
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_cv_{PRIMARY_METRIC}", best_score)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("optuna_n_trials", len(study.trials))

    logger.info(
        "Optuna best for %s — %s: %.4f (in %d trials)",
        model_name, PRIMARY_METRIC, best_score, len(study.trials),
    )
    return best_params, best_score


# ──────────────────────────────────────────────
# Final model training and evaluation
# ──────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test",
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        pipeline: Trained pipeline (features + classifier).
        X: Feature matrix.
        y: True labels.
        dataset_name: Label for logging (e.g., "test", "train").

    Returns:
        Dict of metric_name -> value.
    """
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    logger.info(
        "%s set — Accuracy: %.4f, F1: %.4f, ROC-AUC: %.4f",
        dataset_name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
    )
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report)

    return metrics


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    params: Dict[str, Any],
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train the final model on full training data and evaluate on test set.

    The trained pipeline is:
    1. Saved to disk as a joblib artifact
    2. Logged to MLflow with all metrics
    3. Known categories exported for API-side drift detection

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Held-out test data.
        model_name: Model family name.
        params: Best hyperparameters from Optuna.

    Returns:
        Tuple of (trained pipeline, test metrics dict).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = _create_model_from_params(model_name, params, y_train)
    pipeline = _build_full_pipeline(model_name, model)

    logger.info("Training final %s model on %d samples...", model_name, len(X_train))
    pipeline.fit(X_train, y_train)

    # Evaluate on both splits
    train_metrics = evaluate_model(pipeline, X_train, y_train, "train")
    test_metrics = evaluate_model(pipeline, X_test, y_test, "test")

    # Save model artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Model saved to %s", model_path)

    # Export known categories for API drift detection
    safe_encoder = pipeline.named_steps["features"].named_steps["safe_encoder"]
    safe_encoder.save_known_categories(KNOWN_CATEGORIES_PATH)
    logger.info("Known categories saved to %s", KNOWN_CATEGORIES_PATH)

    # Save model name for downstream tools (SHAP, API)
    import json
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump({"model_name": model_name, "pipeline_type": _pipeline_type_label(model_name)}, f)

    # Log to MLflow
    with mlflow.start_run(run_name=f"final_{model_name}"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("pipeline_type", _pipeline_type_label(model_name))

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Provide input example so MLflow auto-infers the model signature
        mlflow.sklearn.log_model(
            pipeline, "model",
            input_example=X_train.head(1),
        )

    return pipeline, test_metrics

"""
Central configuration for the insurance enrollment prediction pipeline.

All hyperparameters, paths, and constants are defined here to avoid
magic numbers scattered throughout the codebase.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "employee_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = "file:///Users/deepanshi/ML_insurance_uniblox/mlruns"

# ──────────────────────────────────────────────
# Data splitting
# ──────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# ──────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────
TARGET_COL = "enrolled"
ID_COL = "employee_id"

NUMERIC_FEATURES = ["age", "salary", "tenure_years"]
CATEGORICAL_FEATURES = ["gender", "marital_status", "employment_type", "region", "has_dependents"]

# Known categories path (saved during training, used by API for drift detection)
KNOWN_CATEGORIES_PATH = MODEL_DIR / "known_categories.json"

# EDA / analysis figure output
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 300  # seconds

# Scoring metric for model comparison and Optuna objective
PRIMARY_METRIC = "roc_auc"

# MLflow experiment name
EXPERIMENT_NAME = "insurance_enrollment_prediction"

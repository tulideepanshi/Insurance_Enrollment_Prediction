"""
Data loading, validation, and train/test splitting.

Responsible for:
- Reading raw CSV and performing schema validation
- Detecting and reporting data quality issues
- Producing a clean, stratified train/test split
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# Expected schema for validation
EXPECTED_COLUMNS = {ID_COL, TARGET_COL} | set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
EXPECTED_DTYPES = {
    "age": "int64",
    "salary": "float64",
    "tenure_years": "float64",
    "enrolled": "int64",
}


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw employee data from CSV.

    Args:
        path: Override path for the CSV file. Defaults to config.RAW_DATA_PATH.

    Returns:
        Raw DataFrame as read from disk.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run schema and quality checks on the raw DataFrame.

    Checks performed:
    1. All expected columns are present
    2. No duplicate employee IDs
    3. Target column contains only {0, 1}
    4. Numeric columns have no nulls and are within plausible ranges
    5. Categorical columns have no nulls

    Args:
        df: Raw DataFrame to validate.

    Returns:
        The same DataFrame (pass-through for chaining).

    Raises:
        ValueError: If any validation check fails.
    """
    # 1. Column presence
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Duplicate IDs
    dup_count = df[ID_COL].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate employee IDs")

    # 3. Target values
    invalid_targets = set(df[TARGET_COL].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Target column contains invalid values: {invalid_targets}")

    # 4. Null checks
    null_counts = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        logger.warning("Columns with nulls:\n%s", cols_with_nulls)
        # For production: could impute here; for now, raise
        raise ValueError(f"Null values found in: {cols_with_nulls.index.tolist()}")

    # 5. Range checks
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        raise ValueError("Age values out of plausible range [0, 120]")
    if (df["salary"] < 0).any():
        raise ValueError("Negative salary values detected")
    if (df["tenure_years"] < 0).any():
        raise ValueError("Negative tenure values detected")

    logger.info("All validation checks passed")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving target distribution.

    Args:
        df: Validated DataFrame with target column.
        test_size: Fraction of data for the test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )

    logger.info(
        "Split: train=%d (%.1f%% enrolled), test=%d (%.1f%% enrolled)",
        len(train_df),
        train_df[TARGET_COL].mean() * 100,
        len(test_df),
        test_df[TARGET_COL].mean() * 100,
    )
    return train_df, test_df


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix from target vector.

    Drops employee_id (not a feature) and the target column from X.

    Args:
        df: DataFrame containing features and target.

    Returns:
        Tuple of (X, y).
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def load_and_prepare(
    path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Full pipeline: load -> validate -> split -> separate features/target.

    Convenience function that chains all steps.

    Returns:
        Tuple of (raw_df, X_train, y_train, X_test, y_test).
    """
    raw_df = load_raw_data(path)
    validate_dataframe(raw_df)
    train_df, test_df = split_data(raw_df)
    X_train, y_train = get_features_and_target(train_df)
    X_test, y_test = get_features_and_target(test_df)
    return raw_df, X_train, y_train, X_test, y_test

"""
Shared pytest fixtures for the test suite.

Provides small, deterministic sample data that mirrors the production
schema without requiring the full CSV to be present.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal valid DataFrame matching the production schema."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 35, 45, 55, 30, 40, 50, 60, 28, 38],
        "gender": [
            "Male", "Female", "Other", "Male", "Female",
            "Male", "Female", "Other", "Male", "Female",
        ],
        "marital_status": [
            "Single", "Married", "Divorced", "Widowed", "Single",
            "Married", "Divorced", "Widowed", "Single", "Married",
        ],
        "salary": [
            45000, 72000, 55000, 90000, 60000,
            85000, 48000, 110000, 52000, 68000,
        ],
        "employment_type": [
            "Full-time", "Full-time", "Part-time", "Full-time", "Contract",
            "Full-time", "Part-time", "Full-time", "Contract", "Full-time",
        ],
        "region": [
            "Northeast", "South", "Midwest", "West", "Northeast",
            "South", "Midwest", "West", "Northeast", "South",
        ],
        "has_dependents": [
            "No", "Yes", "No", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes",
        ],
        "tenure_years": [1.0, 5.5, 2.3, 15.0, 0.5, 8.2, 3.1, 20.0, 1.8, 6.7],
        "enrolled": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def sample_features_df(sample_raw_df) -> pd.DataFrame:
    """Feature matrix (no employee_id or target)."""
    return sample_raw_df.drop(columns=["employee_id", "enrolled"])


@pytest.fixture
def sample_target(sample_raw_df) -> pd.Series:
    """Target vector."""
    return sample_raw_df["enrolled"]

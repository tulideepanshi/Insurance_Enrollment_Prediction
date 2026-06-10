"""
FastAPI prediction service for insurance enrollment.

Provides:
- POST /predict — single prediction with probability + drift warnings
- POST /predict/batch — batch predictions
- GET /health — liveness/readiness check
- GET /model/info — model metadata

The model pipeline (feature engineering + classifier) is loaded once
at startup. Known categories are loaded from a JSON sidecar file
for API-side drift detection — unseen categorical values trigger
a warning in the response and a structured log entry.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi.responses import RedirectResponse

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import KNOWN_CATEGORIES_PATH, MODEL_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────


class EmployeeInput(BaseModel):
    """Single employee record for prediction.

    Field constraints mirror the data validation rules from the training
    pipeline to catch invalid inputs before they reach the model.
    """

    age: int = Field(..., ge=18, le=120, description="Employee age in years")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    marital_status: str = Field(
        ..., description="Marital status: Single, Married, Divorced, Widowed"
    )
    salary: float = Field(..., gt=0, description="Annual salary in USD")
    employment_type: str = Field(
        ..., description="Employment type: Full-time, Part-time, or Contract"
    )
    region: str = Field(
        ..., description="Region: Northeast, South, Midwest, or West"
    )
    has_dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure_years: float = Field(
        ..., ge=0, description="Years of tenure at the company"
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female", "Other"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("marital_status")
    @classmethod
    def validate_marital_status(cls, v: str) -> str:
        allowed = {"Single", "Married", "Divorced", "Widowed"}
        if v not in allowed:
            raise ValueError(f"marital_status must be one of {allowed}")
        return v

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        allowed = {"Full-time", "Part-time", "Contract"}
        if v not in allowed:
            raise ValueError(f"employment_type must be one of {allowed}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        allowed = {"Northeast", "South", "Midwest", "West"}
        if v not in allowed:
            raise ValueError(f"region must be one of {allowed}")
        return v

    @field_validator("has_dependents")
    @classmethod
    def validate_has_dependents(cls, v: str) -> str:
        allowed = {"Yes", "No"}
        if v not in allowed:
            raise ValueError(f"has_dependents must be one of {allowed}")
        return v

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 35,
            "gender": "Female",
            "marital_status": "Married",
            "salary": 72000.0,
            "employment_type": "Full-time",
            "region": "Northeast",
            "has_dependents": "Yes",
            "tenure_years": 5.2,
        }]
    }}


class PredictionResponse(BaseModel):
    """Single prediction result with optional drift warnings."""

    enrolled: int = Field(..., description="Predicted class: 1=enrolled, 0=not enrolled")
    probability: float = Field(
        ..., description="Probability of enrollment (0.0 to 1.0)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Drift warnings for unseen categorical values",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results with aggregated warnings."""

    predictions: List[PredictionResponse]
    count: int
    warnings: List[str] = Field(
        default_factory=list,
        description="Drift warnings across the entire batch",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata."""

    model_type: str
    pipeline_steps: List[str]
    model_path: str


# ──────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────

_model_pipeline = None
_known_categories: dict = {}


def _load_model() -> None:
    """Load the trained model pipeline and known categories from disk."""
    global _model_pipeline, _known_categories

    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run training first."
        )
    _model_pipeline = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)

    # Load known categories for drift detection
    if KNOWN_CATEGORIES_PATH.exists():
        with open(KNOWN_CATEGORIES_PATH) as f:
            _known_categories = json.load(f)
        logger.info("Known categories loaded from %s", KNOWN_CATEGORIES_PATH)
    else:
        logger.warning(
            "Known categories file not found at %s — drift detection disabled",
            KNOWN_CATEGORIES_PATH,
        )


def _check_drift(inputs: List[EmployeeInput]) -> List[str]:
    """Scan inputs for unseen categorical values.

    Compares each categorical field against the training-time vocabulary
    stored in known_categories.json. Returns a list of human-readable
    warning strings and logs structured warnings.

    The SafeEncoder in the pipeline will also handle the unknown mapping,
    but this API-side check provides the caller with explicit warnings
    before they need to inspect logs.
    """
    if not _known_categories:
        return []

    warnings_list = []
    categorical_fields = [
        "gender", "marital_status", "employment_type", "region", "has_dependents"
    ]

    for field in categorical_fields:
        known = set(_known_categories.get(field, []))
        # Remove the __unknown__ token from comparison — it's our sentinel
        known.discard("__unknown__")
        for i, inp in enumerate(inputs):
            value = getattr(inp, field)
            if value not in known:
                msg = (
                    f"Unknown value '{value}' for field '{field}' "
                    f"(record {i}) — model may be unreliable. "
                    f"Consider retraining."
                )
                warnings_list.append(msg)
                logger.warning(
                    "DRIFT ALERT — field='%s' value='%s' record=%d. "
                    "Not seen during training.",
                    field, value, i,
                )

    return warnings_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    _load_model()
    yield


app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description=(
        "Predicts whether an employee will enroll in a voluntary insurance product "
        "based on demographic and employment data. Includes drift detection for "
        "unseen categorical values."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


def _input_to_dataframe(inputs: List[EmployeeInput]) -> pd.DataFrame:
    """Convert Pydantic models to the DataFrame format expected by the pipeline."""
    records = [inp.model_dump() for inp in inputs]
    return pd.DataFrame(records)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_pipeline is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: EmployeeInput):
    """Predict enrollment for a single employee.

    Returns the predicted class (0/1), enrollment probability, and
    any drift warnings if unseen categorical values are detected.
    """
    if _model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    drift_warnings = _check_drift([employee])

    df = _input_to_dataframe([employee])
    prediction = int(_model_pipeline.predict(df)[0])
    probability = float(_model_pipeline.predict_proba(df)[0, 1])

    return PredictionResponse(
        enrolled=prediction,
        probability=round(probability, 4),
        warnings=drift_warnings,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(employees: List[EmployeeInput]):
    """Predict enrollment for multiple employees.

    Drift warnings are aggregated across the entire batch and also
    included per-record in individual prediction responses.
    """
    if _model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(employees) == 0:
        raise HTTPException(status_code=400, detail="Empty input list")

    if len(employees) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 records per batch")

    drift_warnings = _check_drift(employees)

    df = _input_to_dataframe(employees)
    predictions = _model_pipeline.predict(df)
    probabilities = _model_pipeline.predict_proba(df)[:, 1]

    results = [
        PredictionResponse(
            enrolled=int(pred),
            probability=round(float(prob), 4),
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
        warnings=drift_warnings,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return metadata about the loaded model."""
    if _model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    classifier = _model_pipeline.named_steps["classifier"]
    model_type = type(classifier).__name__
    steps = [name for name, _ in _model_pipeline.steps]

    return ModelInfoResponse(
        model_type=model_type,
        pipeline_steps=steps,
        model_path=str(MODEL_DIR / "best_model.joblib"),
    )

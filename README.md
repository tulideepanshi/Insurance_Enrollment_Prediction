# Insurance Enrollment Prediction

Production-grade ML pipeline that predicts whether an employee will enroll in a voluntary insurance product based on demographic and employment data. Quick Overview - https://tulideepanshi.github.io/Insurance_Enrollment_Prediction/

## Project Structure

```
├── data/
│   └── employee_data.csv          # Raw dataset (10K rows)
├── notebooks/
│   ├── eda.py                     # Exploratory data analysis script
│   └── figures/                   # EDA + analysis visualizations
│       ├── shap/                  # SHAP summary + dependence plots
│       ├── calibration/           # Predicted vs actual calibration plots
│       └── importance/            # Feature importance bar plots + rank table
├── src/
│   ├── config.py                  # Central configuration
│   ├── data/
│   │   └── data_loader.py         # Loading, validation, splitting
│   ├── features/
│   │   └── feature_engineering.py # 3 model-specific pipelines + SafeEncoder
│   ├── models/
│   │   └── trainer.py             # CV comparison, Optuna tuning, MLflow tracking
│   └── api/
│       └── app.py                 # FastAPI service with drift detection
├── models/                        # Serialized artifacts (generated)
├── tests/                         # pytest test suite
├── train.py                       # Training orchestrator
├── analyze.py                     # SHAP + calibration analysis (post-training)
├── feature_importance.py          # Feature importance across all 4 models
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── report.md                      # Analysis report
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run EDA

```bash
make eda
# Figures saved to notebooks/figures/
```

### 3. Train the model

```bash
# Full pipeline: CV comparison + Optuna tuning + final training
make train

# Quick run without hyperparameter tuning
make train-fast
```

Training outputs:
- `models/best_model.joblib` — serialized pipeline (features + classifier)
- `models/known_categories.json` — training-time category vocabulary for drift detection
- `models/model_info.json` — model type metadata
- `models/cv_results.csv` — cross-validation comparison
- `models/test_metrics.json` — held-out test set metrics
- `models/best_params.json` — best hyperparameters
- `mlruns/` — MLflow experiment logs

### 4. View Experiment Tracking

```bash
make mlflow
# Open http://localhost:5000 in your browser
```

### 5. Run post-training analysis

```bash
make analyze
# SHAP plots → notebooks/figures/shap/
# Calibration plots → notebooks/figures/calibration/
```

### 5b. Feature importance comparison

```bash
make importance
# Bar plots + rank table → notebooks/figures/importance/
# Raw scores → notebooks/figures/importance/importance_scores.csv
```

Trains all 4 model families with default parameters, extracts each model's native importance metric, maps transformed features back to original column names, and generates per-model bar plots, a grouped comparison chart, and a rank table.

### 6. Serve the API

```bash
make serve
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 7. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Female",
    "marital_status": "Married",
    "salary": 72000,
    "employment_type": "Full-time",
    "region": "Northeast",
    "has_dependents": "Yes",
    "tenure_years": 5.2
  }'
```

Response (includes drift warnings if unseen categories are detected):
```json
{"enrolled": 1, "probability": 0.8923, "warnings": []}
```

### 8. Run tests

```bash
make test
```

## Architecture: Model-Specific Feature Pipelines

Each model family gets a feature pipeline tailored to its mathematical assumptions:

| Model | Pipeline | Rationale |
|-------|----------|-----------|
| **Logistic Regression** | OutlierCapper → SplineTransformer (quantile knots) → TargetEncoder (cv=5, smoothed) → StandardScaler | Splines for non-linear effects (age step function), TargetEncoder replaces OHE (no multicollinearity, handles unknowns via global mean, prevents leakage via cross-fitting), uniform scaling for fair L2 regularization |
| **Random Forest** | OrdinalEncoder, raw numerics | Trees are scale-invariant, handle outliers naturally. Ordinal encoding avoids feature-importance dilution from one-hot |
| **XGBoost / LightGBM** | CategoricalDtypeTransformer, raw numerics | Native categorical support evaluates optimal category groupings directly — strictly better than any encoding scheme |

All pipelines share a **SafeEncoder** that maps unseen categories to `__unknown__` with drift logging.

## Drift Detection

When the API receives a categorical value not seen during training:
1. A structured `WARNING` is logged with the field name, unseen value, and record index
2. The prediction response includes a `warnings` array describing the drift
3. The SafeEncoder maps the value to `__unknown__` so the model still produces a prediction

This supports both real-time monitoring (via log aggregation) and caller-side handling.

## Docker

```bash
docker-compose up --build    # Full stack: train → API → MLflow
docker-compose run --rm train  # Train only
docker-compose up api          # Serve only (needs trained model)
```

| Service | Port | Description |
|---------|------|-------------|
| api     | 8000 | Prediction API (Swagger at /docs) |
| mlflow  | 5000 | Experiment tracking UI |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single prediction + drift warnings |
| POST | `/predict/batch` | Batch predictions (up to 1000) + drift warnings |
| GET | `/model/info` | Model metadata |

## Configuration

All hyperparameters are centralized in `src/config.py`:

- `TEST_SIZE = 0.20` — train/test split ratio
- `CV_FOLDS = 5` — cross-validation folds
- `OPTUNA_N_TRIALS = 50` — max tuning trials
- `PRIMARY_METRIC = "roc_auc"` — metric for model comparison

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

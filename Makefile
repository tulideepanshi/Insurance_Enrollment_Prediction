.PHONY: install train serve test lint clean docker-train docker-serve docker-up docker-down mlflow eda analyze importance

# ── Local development ──────────────────────────

install:
	pip install -r requirements.txt

train:
	python train.py

train-fast:
	python train.py --skip-tuning

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

eda:
	python -m notebooks.eda

analyze:
	python analyze.py

importance:
	python feature_importance.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Docker ─────────────────────────────────────

docker-up:
	docker-compose up --build

docker-train:
	docker-compose run --rm train

docker-serve:
	docker-compose up --build api

docker-down:
	docker-compose down

# ── MLflow ─────────────────────────────────────

mlflow:
	mlflow ui --backend-store-uri file://$(shell pwd)/mlruns --port 5000

# ── Cleanup ────────────────────────────────────

clean:
	rm -rf mlruns/ models/*.joblib models/*.json models/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

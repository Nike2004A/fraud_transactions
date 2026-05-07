.PHONY: install ingest features train evaluate test test-api test-all all clean help \
        api-local api-docker-build api-docker-run api-docker-stop \
        ui batch-score

PYTHON ?= python3

# Si existe un venv local, preferimos sus binarios — así `make ui`,
# `make api-local`, `make test` funcionan sin necesidad de activar el venv
# en la shell que invoca make. (`make` lanza /bin/sh, que no hereda la
# activación del venv automáticamente.)
VENV_BIN := $(abspath .venv/bin)
ifneq ($(wildcard $(VENV_BIN)/streamlit),)
STREAMLIT ?= $(VENV_BIN)/streamlit
else
STREAMLIT ?= streamlit
endif
ifneq ($(wildcard $(VENV_BIN)/uvicorn),)
UVICORN ?= $(VENV_BIN)/uvicorn
else
UVICORN ?= uvicorn
endif
ifneq ($(wildcard $(VENV_BIN)/python),)
PYTHON := $(VENV_BIN)/python
endif

help:
	@echo "Pipeline (Fases 0-4):"
	@echo "  install            - pip install -e .[dev]"
	@echo "  ingest             - CSV crudo -> parquet en data/staging/"
	@echo "  features           - staging -> curated con ~45 features (Fase 2)"
	@echo "  train              - XGBoost + Optuna + MLflow (Fase 3)"
	@echo "  evaluate           - métricas, SHAP, reportes (Fase 4)"
	@echo "  all                - ingest -> features -> train -> evaluate"
	@echo ""
	@echo "Fase 5 - Packaging:"
	@echo "  api-local          - uvicorn (sin Docker) con reload"
	@echo "  api-docker-build   - docker build -t fraud-api"
	@echo "  api-docker-run     - docker run -p 8000:8000 fraud-api"
	@echo "  api-docker-stop    - matar el container"
	@echo "  ui                 - streamlit run fraud_ui/app.py"
	@echo "  batch-score        - score full test set via API"
	@echo ""
	@echo "Tests:"
	@echo "  test               - tests del proyecto (Fases 0-4 + batch_score)"
	@echo "  test-api           - tests del API (FastAPI TestClient)"
	@echo "  test-all           - test + test-api"
	@echo "  clean              - limpiar artefactos generados"

install:
	pip install -e ".[dev]"

ingest:
	$(PYTHON) -m src.ingest

features:
	$(PYTHON) -m src.features

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTHON) -m pytest tests/ -v

test-api:
	cd fraud_api && $(PYTHON) -m pytest tests/ -v

test-all: test test-api

api-local:
	cd fraud_api && $(UVICORN) src.main:app --reload --port 8000

api-docker-build:
	docker build -t fraud-api -f fraud_api/Dockerfile .

api-docker-run:
	docker rm -f fraud-api 2>/dev/null || true
	docker run -d --name fraud-api -p 8000:8000 fraud-api

api-docker-stop:
	docker rm -f fraud-api

ui:
	cd fraud_ui && $(STREAMLIT) run app.py

batch-score:
	$(PYTHON) -m scripts.batch_score \
		--input data/curated/transactions_features.parquet \
		--output data/scored/test_scored.parquet \
		--filter-split test \
		--batch-size 1000

all: ingest features train evaluate

clean:
	rm -rf data/staging/*.parquet data/curated/*.parquet data/scored/*.parquet \
		models/* mlruns/ reports/figures/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +

.PHONY: install ingest features train evaluate test all clean help

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  install   - pip install -e .[dev]"
	@echo "  ingest    - CSV crudo -> parquet en data/staging/"
	@echo "  features  - staging -> curated con ~45 features (Fase 2)"
	@echo "  train     - XGBoost + Optuna + MLflow (Fase 3)"
	@echo "  evaluate  - métricas, SHAP, reportes (Fase 4)"
	@echo "  test      - pytest"
	@echo "  all       - ingest -> features -> train -> evaluate"
	@echo "  clean     - limpiar artefactos generados"

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
	$(PYTHON) -m pytest -v

all: ingest features train evaluate

clean:
	rm -rf data/staging/*.parquet data/curated/*.parquet models/* mlruns/ reports/figures/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +

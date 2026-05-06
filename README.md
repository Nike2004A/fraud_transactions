# Fraud Transactions — Capstone

Pipeline end-to-end de detección de fraude en transacciones (caso ShopNow,
~120K tx/mes simulado) usando XGBoost + Optuna + MLflow.

## Estado

- [x] **Fase 0** — Bootstrap (estructura, deps, config, ingesta a parquet)
- [x] **Fase 1** — EDA (`notebooks/01_eda.ipynb`, `reports/eda_report.md`)
- [x] **Fase 2** — Feature engineering (27 features, `src/features.py`)
- [x] **Fase 3** — Entrenamiento (XGBoost + Optuna + MLflow) — `models/xgb_best.json`
- [x] **Fase 4** — Evaluación (SHAP + calibración + costo-beneficio) — `reports/evaluation_report.md`
- [ ] **Fase 5** — Artefactos finales y `predict()` para FastAPI

## Resultados (Fase 4)

| métrica (test) | valor |
|---|---:|
| PR-AUC | **0.8771** |
| ROC-AUC | **0.9923** |
| recall@1 % | **0.939** |
| F1 @ thr* | 0.828 |

Ver [`docs/`](docs/) para análisis completo, decisiones de arquitectura y guía de presentación.

## Stack

- Python **3.11** (target). Hoy el host tiene 3.10 — instalar 3.11 antes de
  `make install` (`pyenv install 3.11`, `uv python install 3.11`, o
  `apt install python3.11` con deadsnakes).
- XGBoost 2.x con **GPU/CUDA** (`device='cuda'`, `tree_method='hist'`).
- DuckDB para ingesta, Pandas/Polars para features, scikit-learn pipelines,
  Optuna (TPE) para tuning, MLflow para tracking, SHAP para interpretabilidad.

## Setup

```bash
# 1) crear venv con Python 3.11 (ej. con uv)
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2) instalar dependencias del proyecto + dev
make install

# 3) colocar el CSV de Kaggle en data/raw/
#    fraudTrain.csv y fraudTest.csv
#    https://www.kaggle.com/datasets/kartik2112/fraud-detection

# 4) correr la ingesta
make ingest    # -> data/staging/transactions.parquet
```

## Estructura

```
.
├── data/
│   ├── raw/         # CSV originales (gitignored)
│   ├── staging/     # parquet post-ingesta
│   └── curated/     # parquet con features
├── notebooks/       # 01_eda.ipynb, 02_model_analysis.ipynb
├── src/
│   ├── config.py    # paths, seeds, hiperparámetros
│   ├── ingest.py    # CSV -> DuckDB -> parquet
│   ├── features.py  # (Fase 2)
│   ├── train.py     # (Fase 3)
│   ├── evaluate.py  # (Fase 4)
│   └── predict.py   # (Fase 5)
├── tests/           # pytest
├── models/          # joblib serializado
├── reports/         # eda_report.md, model_card.md, figures/
├── mlruns/          # MLflow tracking (gitignored)
├── Makefile
├── pyproject.toml
└── README.md
```

## Targets de `make`

| target | descripción |
|---|---|
| `install` | `pip install -e ".[dev]"` |
| `ingest` | CSV crudo → parquet staging |
| `features` | staging → curated con ~45 features (Fase 2) |
| `train` | XGBoost + Optuna + MLflow (Fase 3) |
| `evaluate` | métricas, SHAP, reportes (Fase 4) |
| `test` | `pytest` |
| `all` | ingest → features → train → evaluate |
| `clean` | limpia artefactos generados |

## Reproducibilidad

`SEED=42` fijo en `src/config.py`. Splits temporales 70/15/15 ordenados por
`unix_time` (no aleatorios). Target encoding con KFold sin shuffle para
respetar la temporalidad.

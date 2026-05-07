# Fraud Transactions — Capstone

Pipeline end-to-end de detección de fraude en transacciones (caso ShopNow,
~120K tx/mes simulado) usando XGBoost + Optuna + MLflow.

## Estado

- [x] **Fase 0** — Bootstrap (estructura, deps, config, ingesta a parquet)
- [x] **Fase 1** — EDA (`notebooks/01_eda.ipynb`, `reports/eda_report.md`)
- [x] **Fase 2** — Feature engineering (27 features, `src/features.py`)
- [x] **Fase 3** — Entrenamiento (XGBoost + Optuna + MLflow) — `models/xgb_best.json`
- [x] **Fase 4** — Evaluación (SHAP + calibración + costo-beneficio) — `reports/evaluation_report.md`
- [x] **Fase 5** — Packaging (FastAPI containerizada + Streamlit UI + batch scoring)

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
| `test` | tests del proyecto (16, incluye `test_batch_score`) |
| `test-api` | tests del API con FastAPI TestClient (11) |
| `test-all` | `test` + `test-api` |
| `api-local` | uvicorn local con reload, puerto 8000 |
| `api-docker-build` | `docker build -t fraud-api -f fraud_api/Dockerfile .` |
| `api-docker-run` | levantar el container en `localhost:8000` |
| `api-docker-stop` | parar el container |
| `ui` | `streamlit run fraud_ui/app.py` |
| `batch-score` | scorear el test set entero vía API → `data/scored/` |
| `all` | ingest → features → train → evaluate |
| `clean` | limpia artefactos generados |

## Fase 5 — Quickstart

```bash
# 1) Build de la imagen del API (modelo + calibrador embebidos, ~1 GB CPU-only)
make api-docker-build

# 2) Levantar el API
make api-docker-run
curl -s http://localhost:8000/healthz

# 3) UI en otra terminal (corre LOCAL, no en Docker)
pip install -r fraud_ui/requirements.txt
make ui
# -> http://localhost:8501

# 4) Batch scoring del test set entero (sanity check: PR-AUC ≈ 0.8771)
make batch-score
```

Tres componentes:

- [`fraud_api/`](fraud_api/) — FastAPI containerizada. 3 endpoints:
  `GET /healthz`, `POST /score` (con SHAP top-5),
  `POST /score_batch` (≤ 1000 filas). Ver
  [`docs/api_reference.md`](docs/api_reference.md).
- [`fraud_ui/`](fraud_ui/) — Streamlit local. Dropdown de tx + score + decisión a
  ambos thresholds + SHAP. Lee parquet directo (no hay endpoint `/samples`,
  ver [ADR-14](docs/architecture_decisions.md#adr-14--api-containerizada--ui-streamlit-local--ui-lee-parquet-directo)).
- [`scripts/batch_score.py`](scripts/batch_score.py) — CLI que lee parquet,
  llama `/score_batch` en chunks y escribe parquet con scores. Reproduce
  PR-AUC = 0.8771 sobre los 194,502 rows del test set como sanity end-to-end.

## Reproducibilidad

`SEED=42` fijo en `src/config.py`. Splits temporales 70/15/15 ordenados por
`unix_time` (no aleatorios). Target encoding con KFold sin shuffle para
respetar la temporalidad.

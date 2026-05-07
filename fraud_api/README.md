# fraud_api — Inferencia FastAPI containerizada

API CPU-only que expone el modelo XGBoost de Fase 3 + calibrador isotónico
de Fase 4. Tres endpoints: `/healthz`, `/score`, `/score_batch`. Modelo +
calibrador + TreeExplainer se cargan **una sola vez al startup** (lifespan).

Ver [`docs/api_reference.md`](../docs/api_reference.md) para schemas
completos y ejemplos curl.

## Quickstart local (sin Docker)

Asume que `models/xgb_best.json` y `models/calibrator.pkl` existen en la
raíz del repo (correr `make train evaluate` si no).

```bash
# Desde la raíz del repo
make api-local
# o:
cd fraud_api && uvicorn src.main:app --reload --port 8000
```

```bash
curl -s http://localhost:8000/healthz
```

## Quickstart Docker

```bash
make api-docker-build      # build (~3-5 min en cold cache, ~1 GB)
make api-docker-run        # run -p 8000:8000

# en otra terminal
curl -s http://localhost:8000/healthz | python -m json.tool
```

## Tests

```bash
cd fraud_api && python -m pytest tests/ -v
```

11 tests con FastAPI TestClient — corren contra el modelo real, sin
levantar el container.

## Ejemplos

### Score de una transacción

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @- <<'JSON' | python -m json.tool
{
  "features": {
    "log1p_amt": 6.12, "amt_gt_p95_legit": 1, "hour": 23, "dow": 5,
    "is_night": 1, "hour_sin": -0.259, "hour_cos": 0.966, "age": 35.0,
    "time_since_last_tx": 3600.0, "dist_consecutive_km": 12.5,
    "velocity_kmh": 12.5, "rolling_count_1h": 1.0, "rolling_amt_sum_1h": 100.0,
    "rolling_amt_mean_1h": 100.0, "rolling_amt_std_1h": null,
    "rolling_count_24h": 5.0, "rolling_amt_sum_24h": 750.0,
    "rolling_amt_mean_24h": 318.7, "rolling_amt_std_24h": 359.7,
    "rolling_count_7d": 30.0, "rolling_amt_sum_7d": 4500.0,
    "rolling_amt_mean_7d": 150.0, "rolling_amt_std_7d": 200.0,
    "te_merchant": 0.005, "te_category": 0.0024,
    "te_state": 0.006, "te_job": 0.005
  }
}
JSON
```

### Batch desde el script

```bash
python -m scripts.batch_score \
    --input data/curated/transactions_features.parquet \
    --output data/scored/test_scored.parquet \
    --filter-split test \
    --batch-size 1000
```

Sanity end-to-end: el script chequea que el PR-AUC reproducido sobre el
test set sea ≈ 0.8771.

## Imagen Docker

- Base: `python:3.10-slim`
- xgboost-cpu (no CUDA wheels) → ~1 GB final.
- Lifespan: el modelo se carga al arrancar, NO por request.
- `HEALTHCHECK` por curl a `/healthz` cada 30 s.
- Variable `FRAUD_API_MODELS_DIR` redirige el path de los modelos
  (default: `/app/models` en el container).

## Troubleshooting

| síntoma | causa probable | fix |
|---|---|---|
| `/healthz` devuelve 503 con `model_loaded: false` | `models/xgb_best.json` o `models/calibrator.pkl` ausentes en la imagen | Rebuild con `make api-docker-build` (necesita los modelos en repo root) |
| `port already in use` al hacer `docker run` | otro proceso usa el 8000 | `docker run -p 8001:8000 fraud-api` |
| `score_raw` siempre da 0.5 o un valor uniforme | El cliente cargó `XGBClassifier` con `Booster.load_model` (best_iteration=1, solo usa 1 árbol) | Usar `XGBClassifier().load_model(...)` — el wrapper sklearn usa los 32 árboles |
| 422 con `extra_forbidden` | el cliente está enviando un campo que no es del schema (typo de feature) | Revisar la lista de 27 features en `src/settings.py:FEATURE_NAMES` |
| 422 con `int_type` y `loc=["body","features","hour",null]` | Cliente envió `null` en una feature non-nullable | `hour`, `dow`, `is_night`, `amt_gt_p95_legit` no admiten null |
| El batch_score reproduce un PR-AUC distinto a 0.8771 | xgboost del container con versión incompatible al artefacto serializado | `xgboost-cpu==2.1.4` debe matchear la mayor.minor del entrenamiento |

## Estructura

```
fraud_api/
├── Dockerfile
├── requirements.txt
├── README.md
├── src/
│   ├── settings.py     # constantes (paths, thresholds, feature names)
│   ├── schema.py       # Pydantic v2 (FeaturesIn, ScoreResponse, ...)
│   ├── inference.py    # load_bundle, predict_one, predict_batch, shap_top5
│   └── main.py         # FastAPI app + lifespan
└── tests/
    ├── conftest.py
    ├── _fixtures.json  # 1 fraude + 1 legit con scores conocidos
    └── test_api.py     # 11 tests con TestClient
```

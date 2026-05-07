# API Reference — Fraud Detection (Fase 5)

FastAPI / uvicorn / Pydantic v2. Modelo y calibrador embebidos en la imagen
Docker. CPU-only para portabilidad. Cargados una sola vez al startup
(lifespan).

| endpoint | método | descripción |
|---|---|---|
| `/healthz` | GET | Estado del servicio + metadatos del modelo |
| `/score` | POST | Inferencia para una sola transacción + SHAP top-5 |
| `/score_batch` | POST | Inferencia batch (≤ 1000 filas), sin SHAP |

## Convenciones

- **Score crudo (`score_raw`)** vs **calibrado (`score_calibrated`)**: el modelo
  se entrenó con `scale_pos_weight ≈ 176`, así que el score crudo está
  comprimido pero preserva el ranking (PR-AUC test = 0.8771). El calibrado
  (isotónica, fit en val) baja el Brier de 0.115 → 0.0015 y se interpreta
  como probabilidad.
- **Decisiones** (`decision_operating` y `decision_cost`) se aplican siempre
  al **score crudo** — coincide con la decisión que hace
  [`src/evaluate.py`](../src/evaluate.py).
- **NaN handling**: las features `rolling_*` y `te_*` aceptan `null` JSON
  (sentinel para NaN). XGBoost las trata como missing-value. Las
  no-nullable (`hour`, `dow`, `is_night`, `amt_gt_p95_legit`) son requeridas
  con valor numérico.
- **Validación estricta**: extra fields → 422. Fields faltantes → 422.

## `GET /healthz`

```bash
curl -s http://localhost:8000/healthz | python -m json.tool
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "calibrator_kind": "isotonic",
  "n_features": 27
}
```

- **200**: modelo + calibrador cargados correctamente.
- **503**: falló el startup. El body trae `status: "error"` y
  `model_loaded: false`. Endpoints de inferencia también devolverán 503.

## `POST /score`

Request:

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "log1p_amt": 6.12,
      "amt_gt_p95_legit": 1,
      "hour": 23,
      "dow": 5,
      "is_night": 1,
      "hour_sin": -0.259,
      "hour_cos": 0.966,
      "age": 35.0,
      "time_since_last_tx": 3600.0,
      "dist_consecutive_km": 12.5,
      "velocity_kmh": 12.5,
      "rolling_count_1h": 1.0,
      "rolling_amt_sum_1h": 100.0,
      "rolling_amt_mean_1h": 100.0,
      "rolling_amt_std_1h": null,
      "rolling_count_24h": 5.0,
      "rolling_amt_sum_24h": 750.0,
      "rolling_amt_mean_24h": 318.7,
      "rolling_amt_std_24h": 359.7,
      "rolling_count_7d": 30.0,
      "rolling_amt_sum_7d": 4500.0,
      "rolling_amt_mean_7d": 150.0,
      "rolling_amt_std_7d": 200.0,
      "te_merchant": 0.005,
      "te_category": 0.0024,
      "te_state": 0.006,
      "te_job": 0.005
    }
  }'
```

Response:

```json
{
  "score_raw": 0.6701,
  "score_calibrated": 0.3537,
  "decision_operating": {"threshold": 0.6642, "is_fraud": true},
  "decision_cost":      {"threshold": 0.52,   "is_fraud": true},
  "shap_top5": [
    {"feature": "rolling_amt_mean_24h", "value": 318.7, "contribution":  0.324, "direction": "increases_fraud"},
    {"feature": "log1p_amt",            "value":   6.12, "contribution":  0.236, "direction": "increases_fraud"},
    {"feature": "rolling_amt_std_24h",  "value": 359.7, "contribution":  0.100, "direction": "increases_fraud"},
    {"feature": "te_category",          "value": 0.0024, "contribution":  0.076, "direction": "increases_fraud"},
    {"feature": "rolling_amt_std_7d",   "value": 139.2, "contribution": -0.072, "direction": "decreases_fraud"}
  ],
  "base_value": -0.0014
}
```

**Latencia esperada (CPU local)**: 15-25 ms por request (la mayor parte
es SHAP TreeExplainer; sin SHAP serían < 5 ms).

## `POST /score_batch`

Hasta 1000 filas por request. Sin SHAP (sería costoso).

```bash
curl -s -X POST http://localhost:8000/score_batch \
  -H "Content-Type: application/json" \
  -d '{"features": [{...}, {...}, ...]}'
```

Response:

```json
{
  "results": [
    {"score_raw": 0.41, "score_calibrated": 0.05,
     "decision_operating": {"threshold": 0.6642, "is_fraud": false},
     "decision_cost":      {"threshold": 0.52,   "is_fraud": false}},
    ...
  ]
}
```

**Throughput**: ~10 batches de 1000 por segundo en CPU local (~10K rows/s).
Validado: el test set entero (194,502 filas, batch=1000) se procesa en
~20 s y reproduce PR-AUC = 0.8771 / ROC-AUC = 0.9923 contra
[`src/evaluate.py`](../src/evaluate.py).

## Errores 422 (validación de schema)

Field faltante:

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": {"hour": 23}}'
```

```json
{
  "detail": [
    {"type":"missing","loc":["body","features","log1p_amt"],"msg":"Field required",...},
    ...
  ]
}
```

Extra field:

```json
{"detail":[{"type":"extra_forbidden","loc":["body","features","extra_field"],...}]}
```

Null en non-nullable:

```json
{"detail":[{"type":"int_type","loc":["body","features","hour"],...}]}
```

## Lista canónica de features (orden fijo)

`log1p_amt`, `amt_gt_p95_legit`, `hour`, `dow`, `is_night`, `hour_sin`,
`hour_cos`, `age`, `time_since_last_tx`, `dist_consecutive_km`,
`velocity_kmh`, `rolling_count_1h`, `rolling_amt_sum_1h`,
`rolling_amt_mean_1h`, `rolling_amt_std_1h`, `rolling_count_24h`,
`rolling_amt_sum_24h`, `rolling_amt_mean_24h`, `rolling_amt_std_24h`,
`rolling_count_7d`, `rolling_amt_sum_7d`, `rolling_amt_mean_7d`,
`rolling_amt_std_7d`, `te_merchant`, `te_category`, `te_state`, `te_job`.

Non-nullable: `log1p_amt`, `amt_gt_p95_legit`, `hour`, `dow`, `is_night`,
`hour_sin`, `hour_cos`, `age`. El resto admite `null`.

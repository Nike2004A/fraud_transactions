# Training — Fase 3 (XGBoost + Optuna + MLflow)

Pipeline: `src/train.py`. Entrada: `data/curated/transactions_features.parquet`
(salida de Fase 2). Salida: `models/xgb_best.json` + `mlruns/` + figuras en
`reports/figures/`.

## Setup

- Device: `cuda` / tree_method: `hist`
- `scale_pos_weight = 176` (≈ (1 − tasa_train) / tasa_train).
- Early stopping: 30 rondas sobre `aucpr` en val.
- Métrica primaria: **PR-AUC** (val). Secundaria: **recall@1 %**.
- Seed XGBoost + Optuna + KFold = `42`.
- Optuna: `50` trials, sampler `TPESampler(multivariate=True)`,
  pruner `MedianPruner(n_startup_trials=5, n_warmup_steps=20)`,
  `XGBoostPruningCallback` sobre `validation_0-aucpr`.
- NO se usa SMOTE: el desbalance se compensa con `scale_pos_weight`.

## Smoke baseline

| métrica | valor | piso trivial |
|---|---|---|
| PR-AUC val | 0.6653 | 0.0322 (= 5 × tasa_val) |
| ROC-AUC val | 0.9779 | 0.5 |
| recall@1 % val | 0.6302 | 1.00% (random) |
| F1 (thr*) val | 0.6937 | — |
| best_iter | 123 | — |

El baseline supera el piso trivial holgadamente, así que la búsqueda de
hiperparámetros es justificable.

## Mejores hiperparámetros (Optuna)

```json
{
  "max_depth": 10,
  "learning_rate": 0.19266830004180385,
  "min_child_weight": 89.81390281008092,
  "subsample": 0.9680165504587879,
  "colsample_bytree": 0.707379982509407,
  "gamma": 4.782912676538393,
  "reg_alpha": 0.5720224303935324,
  "reg_lambda": 0.043935728341872426
}
```

Búsqueda completada en 89.7s sobre 50 trials.

## Resultados finales

| split | PR-AUC | ROC-AUC | recall@1 % | F1@thr* |
|---|---|---|---|---|
| val  | 0.8893 | 0.9917 | 0.9137 | 0.8355 |
| test | 0.8771 | 0.9923 | 0.9391 | 0.8282 |

- Threshold óptimo en val: **0.6642** (aplicado fijo a test).
- `best_iteration` final: 1.

## Curvas (val)

![PR](figures/pr_curve_val.png)

![ROC](figures/roc_curve_val.png)

## Top-15 features por importancia (gain)

| feature | gain |
|---|---|
| `rolling_amt_mean_24h` | 10,637.5 |
| `amt_gt_p95_legit` | 7,700.3 |
| `log1p_amt` | 4,736.0 |
| `rolling_amt_std_24h` | 3,102.7 |
| `rolling_count_1h` | 970.9 |
| `is_night` | 890.7 |
| `hour_cos` | 675.4 |
| `te_category` | 647.2 |
| `hour` | 554.8 |
| `te_job` | 510.1 |
| `rolling_amt_mean_7d` | 490.2 |
| `rolling_amt_std_7d` | 483.7 |
| `rolling_amt_std_1h` | 478.8 |
| `te_merchant` | 419.8 |
| `hour_sin` | 330.0 |

## Próximos pasos (Fase 4 — out of scope)

- SHAP values (global + local) sobre el split de test.
- Calibración de probabilidades (Platt / isotónica) si se va a usar el score
  como riesgo y no como ranking.
- Análisis de costo-beneficio: matriz de costos por FP/FN.

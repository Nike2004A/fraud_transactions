# Resumen ejecutivo

## El problema

Detectar fraude en transacciones con tarjeta de crédito sobre un dataset de **1.3 millones de transacciones** (Kaggle: *Credit Card Transactions Fraud Detection*). Tasa base de fraude: **0.58%** — fuertemente desbalanceado.

## La solución

Pipeline end-to-end en 4 fases:

1. **Ingesta** → CSV crudo a parquet staging (1.3M filas).
2. **Feature engineering** → 27 features anti-leakage (rolling por tarjeta con `closed='left'`, target encoding KFold solo en train, flag `amt > p95(legítimas_train)`).
3. **Entrenamiento** → XGBoost en GPU (CUDA) + Optuna 50 trials con MedianPruner.
4. **Evaluación** → métricas, SHAP, calibración (Platt + isotónica), análisis por segmento, costo-beneficio.

## Los resultados

| métrica | valor (test) | interpretación |
|---|---:|---|
| **PR-AUC** | **0.8771** | tasa base 0.58% → un modelo random daría 0.006 |
| **ROC-AUC** | **0.9923** | en línea con el techo del dataset |
| **recall@1%** | **0.939** | revisando el 1% más sospechoso, capturamos el 94% de los fraudes |
| **lift@1%** | **93.9×** | el top-1% concentra 94× más fraude que el promedio |
| F1 @ thr* | 0.828 | precision 0.83 / recall 0.83 — equilibrado |
| gap val→test | 1.2 puntos PR-AUC | sin overfit en el split temporal |

## El número que vende

> **Recall del 94% revisando solo el 1% del volumen.**

Con un equipo de fraude que pueda revisar manualmente 1 de cada 100 transacciones, capturamos prácticamente la totalidad del fraude.

## Impacto económico

Análisis costo-beneficio (FN = monto perdido, FP = $5 USD de revisión manual):

- **Threshold operativo F1\*** (0.6642): costo en val = **$65,730**
- **Threshold óptimo de costo** (0.52): costo en val = **$18,032**
- **Ahorro al operar en el threshold óptimo: $47,698 (val)**

Equivale a una reducción de costo del **73%** vs operar al threshold académico de F1 máximo.

## Lo que está bien defendido

- **Splits temporales** (no random) — evita leakage del futuro.
- **`scale_pos_weight≈176`** en lugar de SMOTE — corrige desbalance sin sintetizar datos.
- **Anti-leakage en features** — rolling con `closed='left'`, target encoding fitteado solo en train, p95 calculado solo sobre legítimas de train.
- **Optuna con MedianPruner + early stopping** — búsqueda eficiente, no fuerza bruta.
- **Calibración isotónica** — Brier 0.115 → 0.0015 (mejora 76×). Calibrador persistido en `models/calibrator.pkl`.
- **14 tests passing** — incluye round-trip de modelo, mejora de Brier, propiedad de aditividad SHAP.

## Las salvedades honestas

1. El dataset Kaggle es semi-sintético. En producción, esperar PR-AUC más cercano a **0.55–0.70** en el primer mes.
2. Las top-3 features giran alrededor del **monto** (`log1p_amt`, `rolling_amt_mean_24h`, `amt_gt_p95_legit`) → vulnerable a smurfing.
3. Gap **day vs night** de 20 puntos PR-AUC: el modelo es notablemente más fuerte en horario nocturno. Hay margen de mejora con features diurnas más granulares.

## Para producción se recomienda

1. **Operar al threshold de costo (0.52)**, no al F1\* (0.6642).
2. **Monitor de drift activo** sobre las top-3 features (PSI semanal, alerta si >0.2).
3. **Re-entrenamiento mensual** con chargebacks confirmados (delay típico 30-60 días).

---

*Para el detalle técnico ver [`model_evaluation.md`](model_evaluation.md). Para las decisiones de diseño ver [`architecture_decisions.md`](architecture_decisions.md).*

# Evaluación final — Fase 4

Modelo: `models/xgb_best.json` (Fase 3, no re-tuneado).
Threshold operativo: **0.6642** (F1 max en val, fijo).

## 1. Resumen ejecutivo (test, n=194,502)

| métrica | valor |
|---|---:|
| PR-AUC | 0.8771 |
| ROC-AUC | 0.9923 |
| recall@1 % | 0.9391 |
| precision @ thr* | 0.8267 |
| recall @ thr* | 0.8297 |
| F1 @ thr* | 0.8282 |
| TP / FP / FN / TN | 940 / 197 / 193 / 193,172 |
| lift @ 1 % / 5 % / 10 % | 93.9 / 19.7 / 9.9 |

![PR test](figures/pr_curve_test.png)
![ROC test](figures/roc_curve_test.png)
![Confusión](figures/confusion_matrix.png)

## 2. Calibración

| score | Brier (test) |
|---|---:|
| crudo | 0.11525 |
| Platt | 0.00214 |
| isotónica | 0.00151 |

Calibrador elegido (mejor Brier en val): **isotonic**. Brier en test al
calibrador elegido: **0.00151**.

![Calibración](figures/calibration_curve.png)

**Recomendación**: el modelo se entrenó con `scale_pos_weight≈176`, por lo que
los scores crudos están sesgados hacia arriba en términos absolutos pero
preservan ranking. Usar **score crudo** cuando el caso de uso es ranking
(top-k para revisión manual, alerting). Usar **score calibrado** cuando se
necesita interpretarlo como probabilidad de fraude (umbral de auto-bloqueo,
modelo de riesgo, scoring para ensembles posteriores). El calibrador se
persiste en `models/calibrator.pkl` cuando mejora el Brier en val.

## 3. SHAP (sample n=5000 de test)

Top-10 features por mean(|SHAP|):

| feature | mean(|SHAP|) |
|---|---:|
| `log1p_amt` | 0.2789 |
| `rolling_amt_mean_24h` | 0.1828 |
| `rolling_amt_std_24h` | 0.0590 |
| `hour` | 0.0550 |
| `te_category` | 0.0391 |
| `rolling_amt_std_7d` | 0.0361 |
| `hour_cos` | 0.0284 |
| `rolling_amt_mean_7d` | 0.0271 |
| `te_merchant` | 0.0186 |
| `hour_sin` | 0.0150 |

![SHAP bar](figures/shap_summary_bar.png)
![SHAP beeswarm](figures/shap_summary_beeswarm.png)

### Casos de falso positivo (alto score, label 0)

Tres tx que el modelo marcaría como fraude pero no lo eran:

![FP 1](figures/shap_force_fp_1.png)
![FP 2](figures/shap_force_fp_2.png)
![FP 3](figures/shap_force_fp_3.png)

Patrón típico: combinación de monto alto (`log1p_amt`, `amt_gt_p95_legit`)
con desviación contra el rolling de 24h del titular (`rolling_amt_mean_24h`)
en horario nocturno. El modelo aprende que esa combinación es predominantemente
fraude, pero existe una minoría de tx legítimas atípicas (compras grandes
puntuales, viajes, regalos) que entran en el mismo régimen y no son
distinguibles con las features actuales.

### Casos de verdadero positivo (alto score, label 1)

![TP 1](figures/shap_force_tp_1.png)
![TP 2](figures/shap_force_tp_2.png)
![TP 3](figures/shap_force_tp_3.png)

## 4. Análisis por segmento

| segmento | n | n_pos | PR-AUC | recall@1% |
|---|---:|---:|---:|---:|
| category / **gas_transport** | 19,825 | 93 | 0.9783 | 0.9785 |
| category / **grocery_pos** | 18,645 | 261 | 0.9971 | 0.7165 |
| category / **home** | 18,306 | 33 | 0.8642 | 1.0000 |
| category / **shopping_pos** | 17,698 | 136 | 0.9026 | 0.9265 |
| category / **kids_pets** | 16,949 | 33 | 0.6658 | 0.9091 |
| time_of_day / **night** | 39,440 | 872 | 0.9146 | 0.4415 |
| time_of_day / **day** | 155,062 | 261 | 0.7216 | 0.9349 |
| amount_quartile / **Q1** | 48,582 | 69 | 0.5841 | 0.9565 |
| amount_quartile / **Q2** | 48,659 | 162 | 0.7452 | 0.8889 |
| amount_quartile / **Q3** | 48,633 | 11 | 0.1812 | 0.6364 |
| amount_quartile / **Q4** | 48,628 | 891 | 0.9207 | 0.5320 |
| age_group / **<30** | 69,385 | 335 | 0.8629 | 0.9463 |
| age_group / **30-50** | 75,134 | 385 | 0.8497 | 0.9299 |
| age_group / **50+** | 49,983 | 413 | 0.9148 | 0.9298 |

![Segmentos](figures/segment_metrics.png)

**Segmentos donde el modelo subperformea** (PR-AUC < mediana − 0.10):

- category / **kids_pets** (PR-AUC=0.6658, n_pos=33)
- time_of_day / **day** (PR-AUC=0.7216, n_pos=261)
- amount_quartile / **Q1** (PR-AUC=0.5841, n_pos=69)
- amount_quartile / **Q2** (PR-AUC=0.7452, n_pos=162)
- amount_quartile / **Q3** (PR-AUC=0.1812, n_pos=11)

Hipótesis: en segmentos con bajo `n_pos` el PR-AUC es ruidoso. Cuando el
descenso es sistemático (no por muestra chica) sugiere que las features
agregadas (`rolling_*`, `te_*`) capturan peor el comportamiento de ese
subconjunto — p.ej. tarjetas con poca historia o categorías con
distribución de monto bimodal.

## 5. Costo-beneficio

Asunciones: `costo_FN = monto_tx` (fraude no detectado = pérdida total),
`costo_FP = $5` (revisión manual). Sweep en val.

| threshold | costo total val (USD) | n_FP | n_FN |
|---|---:|---:|---:|
| F1* = 0.6642 | $65,730 | — | — |
| **mín costo = 0.52** | **$18,032** | — | — |

Ahorro en val al pasar de F1* a threshold de mínimo costo: **$47,698**.

![Costo](figures/cost_curve.png)

Estimación de ahorro mensual extrapolando volumen de val al test set:
- Volumen test ≈ 54,028 tx/mes (asumiendo split test cubre ~3 meses).
- Ahorro proyectado: **$13,250 / mes** si se opera al
  threshold de mínimo costo en lugar del F1*.

> El threshold óptimo de costo es más bajo que el F1* (más recall a costa
> de más FPs): el dolor por dejar pasar fraude es proporcional al monto,
> mientras que el dolor por revisar de más es lineal y barato.

## 6. Limitaciones

- **Distribución de monto**: las pérdidas por FN se calculan con el monto
  observado; en producción habría que descontar la fracción recuperable
  por chargeback.
- **Drift**: el split temporal cubre un período acotado del dataset
  Kaggle. En producción se requiere monitoreo de drift (PSI sobre features
  top-3: `rolling_amt_mean_24h`, `amt_gt_p95_legit`, `log1p_amt`) y
  re-entrenamiento periódico.
- **Features ausentes**: device fingerprint, IP, MCC granular, historial
  comercial del merchant — todas darían señal adicional pero no están
  en el dataset.
- **Threshold inestable**: el F1* está cerca de un plateau de la curva
  PR; pequeñas variaciones en val mueven el threshold significativamente.

## 7. Próximos pasos

1. **Online learning**: entrenar incrementalmente con los chargebacks
   confirmados (etiquetas tardías), con un pipeline que normalice el
   delay (~30-60 días) entre tx y label final.
2. **Drift monitoring**: dashboard con PSI/KS sobre las features top-3
   y alertas si el PR-AUC en validación rolling baja >5 % vs baseline.
3. **Features adicionales**: ratio del monto vs media histórica del
   merchant; conteo de tx en países distintos en 1 h; flag de primera
   tx en ese MCC para esa cc.
4. **Modelo dual**: separar el problema en "alta confianza" (auto-bloqueo)
   vs "media confianza" (revisión manual) usando dos thresholds. Permite
   calibrar la cobertura del equipo de fraude.

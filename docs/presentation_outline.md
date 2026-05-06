# Estructura de presentación sugerida

Deck de **15 slides** (~20 min + Q&A). Cada slide tiene: título, contenido sugerido, speaker notes y referencia al doc/figura que lo soporta.

---

## Slide 1 — Portada

**Título**: Detección de fraude en transacciones con tarjeta de crédito
**Subtítulo**: Capstone Data Science — pipeline end-to-end con XGBoost + GPU

**Contenido**:
- Tu nombre + fecha
- Una imagen impactante (la curva PR-AUC del modelo final)

**Speaker note**: 30 segundos para contexto. "Voy a presentar un sistema que captura el 94% del fraude revisando solo el 1% del volumen."

---

## Slide 2 — El problema en una línea

**Título**: ¿Por qué importa esto?

**Contenido**:
- Dataset: 1.3M transacciones, tasa de fraude **0.58%**.
- Desafío: desbalance extremo + costo asimétrico (un FN cuesta el monto, un FP cuesta $5).
- Objetivo: maximizar **recall@k** (capturar fraude con presupuesto fijo de revisión manual).

**Speaker note**: enfatizar que la tasa base es 0.58% — un modelo que diga "siempre legítimo" tiene 99.42% accuracy y es inútil. Por eso PR-AUC y recall@1%, no accuracy.

**Referencia**: [`executive_summary.md`](executive_summary.md) § El problema.

---

## Slide 3 — Pipeline en una imagen

**Título**: Cómo se construyó

**Contenido**: el diagrama mermaid de 4 fases (Ingesta → Features → Train → Evaluate). Imprimir como imagen o copiar el flujo.

**Speaker note**: cada fase es idempotente y reproducible con `make`. Mencionar que está versionado en GitHub.

**Referencia**: [`methodology.md`](methodology.md) § Vista de alto nivel.

---

## Slide 4 — Fase 2: feature engineering (lo crítico)

**Título**: 27 features anti-leakage

**Contenido**:
- 6 categorías: monto, temporal, demografía, geografía, histórico (rolling), categórico (target encoding).
- 3 garantías clave:
  - `closed='left'` en rolling → la tx actual no está en su propio agregado.
  - p95 calculado solo sobre legítimas-train → flag de monto sin contaminación.
  - Target encoding KFold + fitteado solo en train.

**Speaker note**: este es el slide más técnico. El leakage es el error más común en proyectos de fraude — destacar que está cubierto por tests.

**Referencia**: [`methodology.md`](methodology.md) § Fase 2 + [ADR-04](architecture_decisions.md#adr-04--feature-engineering-rolling-con-closedleft) / [ADR-05](architecture_decisions.md#adr-05--target-encoding-kfold-ajustado-solo-en-train) / [ADR-06](architecture_decisions.md#adr-06--anti-leakage-del-p95-de-monto).

---

## Slide 5 — Fase 3: XGBoost + Optuna en GPU

**Título**: Modelo y tuning

**Contenido**:
- XGBoost 2.1 con `device='cuda'` (RTX 4060).
- Optuna: 50 trials, TPESampler multivariado, MedianPruner → ahorra ~40% del cómputo.
- `scale_pos_weight≈176` en lugar de SMOTE.
- Early stopping en `aucpr` val (30 rondas).
- Tiempo total: ~10 min en GPU.

**Speaker note**: defender SMOTE-no — sintetizar muestras minoría rompe el orden temporal. `scale_pos_weight` re-pondera el gradiente sin alterar datos.

**Referencia**: [`methodology.md`](methodology.md) § Fase 3 + [ADR-03](architecture_decisions.md#adr-03--scale_pos_weight176-en-lugar-de-smote).

---

## Slide 6 — Resultados: el número que vende

**Título**: Recall del 94% revisando solo el 1% del volumen

**Contenido**: tabla grande, una sola línea importante:

| | val | test |
|---|---:|---:|
| PR-AUC | 0.8893 | **0.8771** |
| ROC-AUC | — | **0.9923** |
| recall@1% | — | **0.939** |
| lift@1% | — | **93.9×** |
| F1 @ thr* | — | 0.828 |

Sub-mensaje: gap val→test = **1.2 puntos**, sin overfit en split temporal.

**Speaker note**: este es **el** slide. Pausa antes de avanzar. Si solo recuerdan un número, que sea "94% de fraude con 1% de revisión".

**Referencia**: [`executive_summary.md`](executive_summary.md) § Los resultados + [`figures/pr_curve_test.png`](figures/pr_curve_test.png).

---

## Slide 7 — Confusion matrix

**Título**: Cómo se distribuyen las predicciones (test)

**Contenido**: la matriz de confusión + interpretación.

| | pred = 0 | pred = 1 |
|---|---:|---:|
| **y = 0** | 193,172 (TN) | 197 (FP) |
| **y = 1** | 193 (FN) | 940 (TP) |

- Threshold operativo: 0.6642 (F1 max en val).
- Precision = 0.83, Recall = 0.83.

**Speaker note**: por cada 100 alertas, 83 son fraude real. La mayoría de modelos en producción operan en 60-70%.

**Referencia**: [`figures/confusion_matrix.png`](figures/confusion_matrix.png).

---

## Slide 8 — Calibración

**Título**: Score crudo vs calibrado

**Contenido**:
- El score crudo está **mal calibrado** como probabilidad (Brier 0.115) — efecto del `scale_pos_weight≈176`.
- Isotónica corrige a Brier 0.0015 (mejora 76×).
- Recomendación: **score crudo para ranking**, **score calibrado para probabilidad**.

**Speaker note**: aclarar que la calibración no afecta el ranking ni las métricas operativas. Es solo si se quiere usar el score como "probabilidad" en otro sistema.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 2 + [`figures/calibration_curve.png`](figures/calibration_curve.png).

---

## Slide 9 — Análisis costo-beneficio

**Título**: ¿Cuánto se ahorra con el modelo?

**Contenido**:
- Asunciones: FN = monto perdido, FP = $5 USD.
- Threshold F1\* (0.6642): costo en val = **$65,730**.
- Threshold óptimo (0.52): costo en val = **$18,032**.
- **Ahorro: $47,698 en val (–73%)**.

**Speaker note**: el threshold académico no es el threshold óptimo de negocio. El óptimo es **más bajo** (más recall) porque dejar pasar fraude es proporcional al monto.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 1 (cost-benefit) + [`figures/cost_curve.png`](figures/cost_curve.png).

---

## Slide 10 — Interpretabilidad: SHAP

**Título**: Qué features manejan al modelo

**Contenido**: bar plot de SHAP top-10. Subrayar:
- Top-3: `log1p_amt`, `rolling_amt_mean_24h`, `amt_gt_p95_legit`.
- Las 3 giran alrededor del **monto** → vulnerable a smurfing (próximo paso).

**Speaker note**: SHAP da exactitud (no aproximación) sobre árboles. El test [`tests/test_evaluate.py::test_shap_sums_to_score`](../tests/test_evaluate.py) verifica la propiedad de aditividad.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 3.B + [`figures/shap_summary_bar.png`](figures/shap_summary_bar.png).

---

## Slide 11 — Análisis por segmento

**Título**: Dónde el modelo es fuerte y dónde no

**Contenido**: tabla resumida de segmentos.

| segmento | PR-AUC | nota |
|---|---:|---|
| night | 0.915 | 🟢 fuerte |
| day | 0.722 | 🟡 20pt menos |
| Q4 monto | 0.921 | 🟢 |
| Q3 monto | 0.181 | ⚠️ ruido (n_pos=11) |
| <30, 30-50, 50+ | 0.85–0.91 | 🟢 homogéneo |

**Speaker note**: el gap day vs night es real (no muestra chica) y honesto. Hipótesis: el día tiene más actividad legítima ambigua. Hay margen de mejora con features diurnas más granulares.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 2 (segmentos) + [`figures/segment_metrics.png`](figures/segment_metrics.png).

---

## Slide 12 — Las 3 salvedades honestas

**Título**: Lo que no es perfecto

**Contenido**:
1. **Dataset semi-sintético** → en producción esperar PR-AUC 0.55–0.70.
2. **Top features dominadas por monto** → vulnerable a smurfing.
3. **Threshold sobre plateau** → drift puede degradarlo silenciosamente.

**Speaker note**: hablar de esto **antes** de que el panel pregunte. Demuestra honestidad técnica y madurez. Mucho más fuerte que esquivarlo.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 3.

---

## Slide 13 — Para producción

**Título**: Tres barandas obligatorias

**Contenido**:
1. **Operar al threshold de costo (0.52)**, no al F1\* (0.6642).
2. **Monitor de drift** (PSI semanal sobre top-3 features, alerta si >0.2).
3. **Re-entrenamiento mensual** con chargebacks confirmados (delay 30-60 días).

**Speaker note**: sin estas tres cosas, el modelo va a andar bárbaro 1-2 meses y después degradarse. Mostrar que entendés el lifecycle, no solo el modelo.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 4.

---

## Slide 14 — Lo que sumaría con más tiempo

**Título**: Próximos pasos

**Contenido**:
- **Comparación vs LightGBM / CatBoost** — el benchmark que el panel va a pedir.
- **Features adicionales**: ratio del monto vs media histórica del merchant, conteo de tx en países distintos en 1h, primer MCC para esa cc.
- **Modelo dual**: thresholds separados para auto-bloqueo (alta confianza) vs revisión manual (media).
- **Online learning** con chargebacks confirmados + drift monitoring activo.

**Speaker note**: si te queda tiempo, mencionar graph features (transacción ↔ merchant ↔ device) — eso es estado del arte en industria.

**Referencia**: [`model_evaluation.md`](model_evaluation.md) § 7.

---

## Slide 15 — Cierre

**Título**: TL;DR

**Contenido** (1 frase grande, centrada):

> **Recall del 94% revisando solo el 1% del volumen, gap train–test de 1.2 puntos PR-AUC, ahorro proyectado de $47k vs threshold académico.**

Sub-línea: pipeline reproducible end-to-end, 14 tests passing, código en GitHub.

**Speaker note**: agradecer + abrir Q&A.

---

## Apéndice — slides de respaldo (Q&A)

Tener listas pero no mostrar por defecto:

- **A1 — `scale_pos_weight` vs SMOTE**: justificación detallada (ver [ADR-03](architecture_decisions.md#adr-03--scale_pos_weight176-en-lugar-de-smote)).
- **A2 — Hiperparámetros Optuna**: tabla completa con rangos y óptimos.
- **A3 — MLflow runs**: screenshot del UI con las 3 runs (baseline, best, evaluation).
- **A4 — Tests**: lista de los 14 tests + qué garantiza cada uno.
- **A5 — Force plots SHAP**: 3 ejemplos de TP + 3 de FP con interpretación.
- **A6 — Estructura del repo**: `tree -L 2` + breve descripción de cada folder.

---

## Tips para la presentación

- **Tiempo**: 90s por slide → ~22 min para los 15 slides. Dejar 8-10 min para Q&A.
- **Imagen sobre texto**: cada slide debe tener ≤ 5 bullets o ≤ 1 tabla. Las figuras vienen de [`figures/`](figures/).
- **Números en negrita**: el ojo del panel los va a buscar.
- **Honestidad técnica > ventas**: el slide 12 (salvedades) suele ser el que más respeto da.
- **Cerrar con la frase de slide 15**: que se la lleven literal a casa.

# EDA — Fase 1 (Credit Card Transactions Fraud Detection)

Dataset: `data/staging/transactions.parquet` (generado por `make ingest`).
Notebook fuente: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb).
Figuras: [reports/figures/](figures/).

## Quality report (post-ingest)

| métrica              | valor                                  |
|----------------------|----------------------------------------|
| filas                | 1,296,675                              |
| fraudes              | 7,506 (**0.5789 %**)                   |
| ratio desbalance     | **172 : 1**                            |
| rango temporal       | 2019-01-01 00:00:18 → 2020-06-21 12:13:37 (537 días) |
| tarjetas únicas      | 983                                    |
| nulos                | sólo `merch_zipcode` (195,973 ≈ 15.1 %) |

## Hallazgos por análisis

### 1. Balance de clases ([fig](figures/01_class_balance.png))
- 0.58 % de fraude → **PR-AUC** como métrica primaria; ROC-AUC engaña a este nivel de desbalance.
- 172:1 → usar `scale_pos_weight` o `class_weight` en XGBoost (no SMOTE: agrega ruido en datos tabulares con muchas categóricas).

### 2. `amt` por clase ([fig](figures/02_amt_by_class.png))
| clase  | mediana | p95     | máximo    |
|--------|---------|---------|-----------|
| legit  | 47.28   | 189.90  | 28,948.90 |
| fraude | 396.51  | 1,083.99| **1,376.04** |

- Las medianas difieren **8×** y los fraudes están acotados arriba en ~1,376 USD: el fraude vive en una banda alta-pero-no-extrema.
- **Acción Fase 2**: `log1p(amt)`, flag `amt_p95_legit` (>189.9), bins por percentil sobre el split de train.

### 3. Distribución horaria/diaria ([fig](figures/03_temporal_fraud_rate.png))
- **Hora**: pico nocturno marcadísimo — 22 h: 2.88 %, 23 h: 2.84 %, 00–02 h: ~1.5 %. **5× la tasa global**.
- Horario diurno (08–18 h) está bajo la línea global. Señal extremadamente fuerte y barata.
- **DOW**: viernes 0.71 %, jue 0.68 %, mié 0.66 %, fin de semana ~0.49 %. Variación moderada (~1.5×).
- **Acción Fase 2**: `hour`, `is_night` (22-02), `dow` y posiblemente sin/cos cíclicos para hora.

### 4. Top categorías por tasa ([fig](figures/04_category_fraud_rate.png))
| category        | rate   |
|-----------------|--------|
| shopping_net    | 1.756 %|
| misc_net        | 1.446 %|
| grocery_pos     | 1.410 %|
| shopping_pos    | 0.723 %|
| gas_transport   | 0.469 %|
| (resto < 0.32 %)|        |

- Las dos `_net` (online) y `grocery_pos` triplican la tasa global.
- **Acción Fase 2**: target encoding **con KFold sobre el split de train** (smoothing m=20, ya configurado en `config.TARGET_ENCODING_SMOOTHING`). One-hot directo es viable (14 categorías) y evita riesgo de leakage si el TE no se hace bien.

### 5. Distancia haversine ([fig](figures/05_distance_by_class.png))
| clase  | mediana | media | p95   |
|--------|---------|-------|-------|
| legit  | 78.2 km | 76.1  | 120.5 |
| fraude | 77.9 km | 76.3  | 120.2 |

- **No hay separación**: las distribuciones se superponen. El generador sintético de Kaggle muestrea coordenadas de comercio uniformemente alrededor del titular, por lo que la distancia cruda **no es feature**.
- **Acción Fase 2**: descartar `distance_km` como feature directa. La geografía aporta sólo si se mide **relativa a la operativa habitual de cada tarjeta** (p.ej. desviación respecto al `merch_lat/long` mediano histórico de la tarjeta, en una ventana rolling).

### 6. Heatmap por estado ([fig](figures/06_state_fraud_heatmap.png))
- Top: AK 1.70 %, NV 0.84 %, CO 0.81 %, OR 0.80 %, TN 0.80 %.
- Variación moderada (≈3×); 51 estados con `count ≥ 1000` en su mayoría.
- **Acción Fase 2**: target encoding KFold sobre `state` (cardinalidad alta para one-hot razonable, pero TE con smoothing es más compacto).

### 7. `cc_num` ([fig](figures/07_card_analysis.png))
- 983 tarjetas únicas, **762 (77.5 %) tienen ≥1 fraude**.
- Tx por tarjeta: mediana 1,054, p95 2,922, máx 3,123 → tráfico denso y sostenido por tarjeta.
- Promedio 9.85 fraudes en las tarjetas comprometidas → señal repetitiva, no eventos aislados.
- **Acción Fase 2** (la más importante): **rolling features por `cc_num`** ordenadas por `unix_time` —
  - Conteo y monto en ventanas (1 h, 1 d, 7 d).
  - Desviación de `amt` respecto al mediano/std histórico de la tarjeta.
  - Tiempo desde la última transacción.
  - Distancia entre comercios consecutivos / velocidad implícita.
- `cc_num` se usa **sólo como llave de agregación**, nunca como feature (ya está listada en `DROP_COLUMNS` conceptual).

### 8. `merchant` — addendum ⭐ ([fig](figures/08_merchant_top_bottom.png))

| métrica | valor |
|---|---|
| cardinalidad | 693 comercios |
| count median | 1.863 tx por comercio (señal densa) |
| top fraud_rate | `fraud_Kozey-Boehm` 2.572 % |
| bottom (>0) | ~0.04 % |
| **comercios con 0 fraudes** | 14 |
| spread top / bottom-no-cero | ~64× |

- Señal **enorme y densa**: cada comercio tiene cientos de transacciones, así que las tasas son confiables (no son artefactos de muestra chica).
- Hay 14 comercios "limpios" (0 fraudes) y un grupo claramente abusado (top 10 por encima del 1.9 %).
- **Acción Fase 2**: target encoding KFold (smoothing m=20) sobre `merchant`. Probable predictor más fuerte después de las features rolling de `cc_num`.

### 9. Edad del titular — addendum ([fig](figures/09_age_by_class.png))

Edad derivada como `(trans_date_trans_time - dob)` en años.

| bucket | tasa de fraude |
|---|---|
| <25     | 0.628 % |
| 25-35   | 0.483 % |
| 35-50   | 0.456 % |
| 50-65   | 0.741 % |
| 65+     | 0.743 % |

- Patrón **bimodal**: jóvenes (<25) y mayores (50+) están por encima de la tasa global; el segmento 25-50 está por debajo.
- Spread ~1.6× entre el bucket más alto y el más bajo — moderado pero real.
- Mediana de edad: legit 44.0 años, fraude 47.8 años (titulares más mayores ligeramente sobre-representados en fraude).
- **Acción Fase 2**: incluir `age` como feature numérica directa (XGBoost encuentra los splits no-lineales solo); descartar `dob` raw.

### 10. `city_pop` — addendum (resultado negativo)

| bucket de población | tasa de fraude |
|---|---|
| <1k        | 0.578 % |
| 1k-10k     | 0.562 % |
| 10k-100k   | 0.555 % |
| 100k-1M    | 0.672 % |
| 1M+        | 0.591 % |

- Spread máx/min: **1.21×** — esencialmente plano.
- Medianas casi idénticas (legit 2.456 / fraude 2.623).
- **Acción Fase 2**: **descartar** `city_pop` como feature directa. La población de la ciudad del titular no separa.

## Implicancias para Fase 2 (feature engineering)

1. **Mantener orden temporal estricto** (`unix_time` ascending) para todos los rolling — el split temporal 70/15/15 ya está fijado en `config.py`.
2. **Target encoding** (`category`, `state`, `merchant`, `job`) **siempre con KFold dentro del fold de train** y aplicado luego a val/test. `TARGET_ENCODING_SMOOTHING=20`, `TARGET_ENCODING_FOLDS=5`.
3. **No incluir** como features: `cc_num`, `trans_num`, `first`, `last`, `street`, `Unnamed: 0`, `dob` (raw — usar `age` derivada), `city_pop` (sin señal), ni `merch_zipcode` si > 15 % nulo.
4. **Features prioritarias** (orden de impacto esperado, refinado tras addendum):
   1. rolling por tarjeta (counts, amt stats, time-deltas, distancia entre tx consecutivas)
   2. **target encoding KFold de `merchant`** ⭐ (señal nueva más fuerte que vimos)
   3. `hour` + `is_night`
   4. `log1p(amt)` + interacción con `category`
   5. target encoding de `category`, `state`
   6. `age` derivada de `dob` (numérica directa)
5. **Métrica primaria**: PR-AUC. Secundarias: recall@k (top 1 % de scores), F1 al threshold óptimo del val set.
6. **Modelo base Fase 3**: XGBoost `device='cuda'`, `tree_method='hist'`, `scale_pos_weight ≈ 172`.

## Riesgo de leakage detectado / vigilado

- Las tasas reportadas aquí (categoría, estado, hora) **son descriptivas sobre todo el dataset**. No usarlas como features computadas globalmente: en Fase 2 se recalculan **sólo dentro del fold de train**.
- Cualquier estadístico agregado por `cc_num` debe construirse con ventana **estrictamente pasada** (excluir la transacción actual) para no filtrar futuro.
- `unix_time` define el orden; no usarlo como feature numérica directa (acumula deriva temporal).

# Features — Fase 2 (Credit Card Transactions Fraud Detection)

Pipeline: `src/features.py`. Entrada: `data/staging/transactions.parquet`
(salida de Fase 1). Salida: `data/curated/transactions_features.parquet`
(1.296.675 × 29 columnas, ZSTD).

| split | filas    | tasa de fraude |
|-------|----------|----------------|
| train | 907.672  | 0.564 %        |
| val   | 194.501  | 0.644 %        |
| test  | 194.502  | 0.583 %        |

## Inventario de features (27)

| # | feature | tipo | grupo | motivación (sección del [EDA](eda_report.md)) |
|---|---|---|---|---|
| 1 | `log1p_amt`           | float32 | monto      | §2: spread legit/fraude 8× en mediana; log estabiliza la escala |
| 2 | `amt_gt_p95_legit`    | int8    | monto      | §2: p95 legit ≈ 189.9 USD; flag separa "monto inusualmente alto" |
| 3 | `hour`                | int16   | temporal   | §3: pico nocturno 22-02h (~5× tasa global) |
| 4 | `dow`                 | int16   | temporal   | §3: viernes/jueves picos moderados (~1.5×) |
| 5 | `is_night`            | int8    | temporal   | §3: bandera explícita 22-02h, predictor barato y fuerte |
| 6 | `hour_sin`            | float32 | temporal   | §3: codificación cíclica para que XGBoost capte vecindad 23h↔00h |
| 7 | `hour_cos`            | float32 | temporal   | §3: idem complementaria al seno |
| 8 | `age`                 | float32 | demografía | §9: bimodal <25 y 50+ (~1.6× spread); usar al momento de la tx |
| 9 | `time_since_last_tx`  | float32 | rolling    | §7: bursts de tx/tarjeta — el delta corto suele ser señal |
| 10 | `dist_consecutive_km` | float32 | rolling    | §5: la haversine cruda no separa, pero la geografía relativa por tarjeta sí |
| 11 | `velocity_kmh`        | float32 | rolling    | §5+§7: distancia/Δt → "imposible-travel" típico de fraude |
| 12 | `rolling_count_1h`    | float32 | rolling    | §7: densidad reciente — picos en ventana corta = compromiso |
| 13 | `rolling_amt_sum_1h`  | float32 | rolling    | §7: monto acumulado reciente |
| 14 | `rolling_amt_mean_1h` | float32 | rolling    | §7: comparable contra `amt` actual sin construirlo aquí (XGBoost lo cruza) |
| 15 | `rolling_amt_std_1h`  | float32 | rolling    | §7: dispersión reciente — fraude rompe la regularidad de la tarjeta |
| 16 | `rolling_count_24h`   | float32 | rolling    | §7: ídem 1h en escala diaria |
| 17 | `rolling_amt_sum_24h` | float32 | rolling    | §7 |
| 18 | `rolling_amt_mean_24h`| float32 | rolling    | §7 |
| 19 | `rolling_amt_std_24h` | float32 | rolling    | §7 |
| 20 | `rolling_count_7d`    | float32 | rolling    | §7: línea de base semanal por tarjeta |
| 21 | `rolling_amt_sum_7d`  | float32 | rolling    | §7 |
| 22 | `rolling_amt_mean_7d` | float32 | rolling    | §7 |
| 23 | `rolling_amt_std_7d`  | float32 | rolling    | §7 |
| 24 | `te_merchant` ⭐       | float32 | TE-KFold   | §8: 693 comercios, spread top/bottom 64×, 14 con 0 fraudes |
| 25 | `te_category`         | float32 | TE-KFold   | §4: shopping_net 1.76 % vs media 0.58 % |
| 26 | `te_state`            | float32 | TE-KFold   | §6: AK 1.70 %, spread ~3× |
| 27 | `te_job`              | float32 | TE-KFold   | cardinalidad alta → smoothing m=20 lo aplana cuando hay poca data |

Más: `is_fraud` (target) y `split` (`train`/`val`/`test`).

## Columnas descartadas (y por qué)

| columna | razón |
|---|---|
| `Unnamed: 0`, `trans_num`, `first`, `last`, `street` | IDs / texto libre (`config.DROP_COLUMNS`) |
| `cc_num` | sólo llave de agregación rolling — nunca feature |
| `dob` | sustituida por `age` derivada (al momento de la tx) |
| `city_pop` | EDA §10: spread 1.21× — sin señal |
| `merch_zipcode` | 15.1 % nulos, redundante con `merch_lat/long` |
| `trans_date_trans_time` | capturada en `hour`/`dow`/`is_night`/`age` |
| `gender` | no validado por el EDA |
| `lat`, `long`, `merch_lat`, `merch_long` | crudas no separan (§5); usadas sólo para `dist_consecutive_km` |
| `city`, `zip` | cardinalidad alta sin señal validada |
| `merchant`, `category`, `state`, `job` (raw) | reemplazadas por su `te_*` |
| `amt` (raw) | reemplazada por `log1p_amt` + flag p95 |
| `unix_time`, `ts` | sólo se usan durante el pipeline (split + rolling) |

## Anti-leakage — cómo se construye cada estadístico

1. **Split temporal estricto.** El df se ordena globalmente por `unix_time` y
   se asigna 70 % / 15 % / 15 % posicionalmente — sin `shuffle`. Esto garantiza
   que toda fila de val/test es **estrictamente posterior** a las de train.
2. **Rolling por `cc_num` con `closed='left'`.** La ventana para la fila `i`
   es `[t_i − win, t_i)`: excluye la propia transacción. Con
   `min_periods=0`, las filas iniciales de cada tarjeta tienen `count=0` y
   `sum=0`, mientras que `mean`/`std` quedan NaN (XGBoost los maneja).
3. **`time_since_last_tx`, `dist_consecutive_km`, `velocity_kmh`.** Usan
   `groupby(cc_num).shift(1)` y/o `.diff()`: por construcción nunca incluyen
   datos de la fila actual. La primera tx de cada tarjeta queda NaN (983
   tarjetas en el dataset → 983 NaNs en estas tres columnas, alineado con
   §7 del EDA). `velocity_kmh` también es NaN cuando `time_delta == 0`
   (timestamps duplicados, ~20 casos).
4. **`amt_gt_p95_legit`.** El umbral p95 se calcula **sólo sobre transacciones
   legítimas del split de train**. Aplicado uniforme a val/test: el flag no
   contamina con el techo de fraude ni con futuro.
5. **Target encoding (`merchant`, `category`, `state`, `job`).**
   - **En train**: KFold con 5 folds (seed=42, shuffle=True). El encoding de
     cada fila proviene del *out-of-fold* — no usa su propio `is_fraud`.
   - **En val/test**: mapping aprendido sobre el train completo (también con
     smoothing). Categorías nuevas se imputan al `prior_global` de train.
   - **Smoothing**: `(n·mean_cat + m·mean_global) / (n + m)` con `m=20`. Ver
     `config.TARGET_ENCODING_*`.
6. **Edad al momento de la transacción.** `age = (trans_date_trans_time − dob)
   / 365.25 días`. No depende de la fecha de hoy → la feature es estable
   entre re-entrenamientos.

Las tasas globales reportadas en el EDA se **describen** allí pero **no se
reutilizan como feature**: cualquier estadístico target-related se recalcula
sólo con datos del fold de train.

## Verificación (`tests/test_features.py`)

| test | qué garantiza |
|---|---|
| `test_no_leakage_rolling`        | Sobre 50 tx sintéticas: `count_1h[i]` y `sum_1h[i]` NO incluyen `amt[i]`. |
| `test_target_encoding_only_uses_train` | El TE de val/test toma exactamente los valores del mapping aprendido en train (con smoothing m=20). Construye un dataset donde `cat_a` tiene 30 % de fraude en train y 0 % en val/test: el TE en val/test refleja la tasa de **train**, no la global. |
| `test_split_temporal_orden`      | 70/15/15 sin solapamiento; `max(unix_time[train]) ≤ min(unix_time[val]) ≤ min(unix_time[test])`. |
| `test_age_derivation`            | Edad calculada al timestamp de la tx (no `today()`): tx separadas en años producen edades distintas y monotónicas. |

```
$ make test
8 passed in 0.92s
```

## Output curado — calidad

- **Shape**: 1.296.675 × 29 (27 features + `is_fraud` + `split`).
- **NaNs por diseño** (todos esperados):
  - `time_since_last_tx` / `dist_consecutive_km`: 983 — primera tx de cada
    tarjeta (= 983 tarjetas únicas).
  - `velocity_kmh`: 1.003 — 983 + 20 con `time_delta == 0`.
  - `rolling_amt_mean_*` / `rolling_amt_std_*`: NaN cuando la ventana
    estrictamente pasada está vacía (1h: ~84 % de las filas; 7d: 0.1 %).
- **TE en train** (smoothing m=20): mediana ~0.003-0.005 para todas las
  columnas; máximos 0.029 (merchant), 0.018 (category), 0.29 (state — AK
  con baja densidad), 0.42 (job).

## Próximos pasos (Fase 3 — out of scope acá)

- XGBoost `device='cuda'`, `tree_method='hist'`, `scale_pos_weight ≈ 172`.
- Búsqueda de hiperparámetros con Optuna sobre **PR-AUC** en val.
- Threshold tuning sobre val (top-k recall + F1 al threshold óptimo).

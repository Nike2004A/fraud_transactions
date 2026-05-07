# Architecture Decision Records (ADRs)

Decisiones de diseño del proyecto, en formato corto. Cada ADR explica **qué se decidió**, **qué alternativas se descartaron** y **por qué**. Útil para defender el proyecto frente a un panel y para retomarlo en 6 meses.

---

## ADR-01 — XGBoost como modelo principal

**Decisión**: usar `xgboost.XGBClassifier` con `tree_method='hist'` y `device='cuda'`.

**Alternativas consideradas**:
- LightGBM — performance similar, GPU support menos maduro al momento del proyecto.
- CatBoost — manejo nativo de categóricas, pero target encoding manual es más transparente.
- Redes neuronales (TabNet, FT-Transformer) — overkill para 1.3M filas tabulares.
- Logistic regression / RandomForest — baselines, descartados por límite superior bajo.

**Razón**: XGBoost es el estándar de facto para fraud tabular, soporta CUDA con `tree_method='hist'`, integra con SHAP TreeExplainer (cálculo exacto, no aproximado), y tiene Optuna integration con `XGBoostPruningCallback`.

**Costo de la decisión**: no se hizo benchmark formal vs LightGBM. Para un capstone es defendible; en industria sería raro no compararlos.

---

## ADR-02 — Splits temporales 70/15/15 (no random)

**Decisión**: ordenar por `unix_time` y dividir 70% train / 15% val / 15% test estrictamente cronológico.

**Alternativas consideradas**:
- Random shuffle — descartado, induce leakage del futuro hacia el pasado.
- K-fold cross-validation — descartado por la misma razón.
- Stratified split — descartado, mantener desbalance natural en val/test es más realista.
- Walk-forward validation — más riguroso pero exagera el costo computacional para un capstone.

**Razón**: en fraude el patrón cambia con el tiempo (drift de fraudsters, cambios de comportamiento legítimo). Validar con futuro real evita que el modelo memorice patrones que no generalizan a producción.

**Validación**: el gap PR-AUC val(0.8893) → test(0.8771) = 1.2 puntos. Si fuera mucho mayor, indicaría drift severo dentro del propio dataset.

---

## ADR-03 — `scale_pos_weight≈176` en lugar de SMOTE

**Decisión**: usar `scale_pos_weight = (1 - tasa_fraude_train) / tasa_fraude_train ≈ 176` para compensar el desbalance.

**Alternativas consideradas**:
- SMOTE / ADASYN — sintetizar minoría. Descartado: introduce muestras artificiales que no respetan el orden temporal ni la estructura de `cc_num`.
- Undersampling de la mayoría — descartado, descarta señal genuina.
- `class_weight='balanced'` (sklearn) — equivalente conceptual, pero `scale_pos_weight` es la API nativa de XGBoost.
- Focal loss — interesante, pero requiere custom objective y SHAP ya no es directo.

**Razón**: `scale_pos_weight` es matemáticamente exacto (re-pondera el gradiente sin alterar los datos), barato, y compatible con todo el ecosistema (SHAP, MLflow, Optuna).

**Costo de la decisión**: el score crudo queda mal calibrado como probabilidad — se compensa con calibración isotónica posterior (ADR-08).

---

## ADR-04 — Feature engineering: rolling con `closed='left'`

**Decisión**: las features `rolling_*` (1h, 24h, 7d) por `cc_num` usan `closed='left'` para excluir la transacción actual del agregado.

**Alternativas consideradas**:
- `closed='right'` (default) — incluye la fila actual → leakage del propio target.
- Calcular sobre todo el dataset sin separar por cc_num — pierde la señal individual de cada tarjeta.

**Razón**: la fila *i* solo debe "ver" transacciones estrictamente anteriores. Si la rolling mean incluye la transacción actual, el modelo aprende a usar `mean ≈ amt_actual` como atajo, lo que se rompe en producción donde la tx actual aún no está en el agregado.

**Costo**: la primera transacción de cada `cc_num` queda con NaN/0 en las rolling — XGBoost maneja NaN nativamente, no es un problema.

---

## ADR-05 — Target encoding KFold ajustado solo en train

**Decisión**: para `merchant`, `category`, `state`, `job` (alta cardinalidad) usar target encoding con smoothing m=20 y K=5 folds. **Fit solo sobre el split de train**; val/test usan el mapping del train completo.

**Alternativas consideradas**:
- One-hot encoding — explosión de dimensionalidad (`merchant` tiene >600 valores únicos).
- Frequency encoding — pierde información del target.
- Ordinal encoding directo — no respeta semántica.
- Target encoding sin KFold — leakage dentro de train (cada fila usa su propio target en el mean).

**Razón**: KFold out-of-fold dentro de train evita que cada fila "vea" su propio target. El smoothing con prior global (`smoothing=20`) protege contra categorías de baja frecuencia. Categorías nuevas en val/test caen al `mean_global` (graceful degradation).

**Riesgo conocido**: si en producción aparece un merchant nuevo, cae al prior y el modelo pierde señal en esa fila. Documentado en `model_evaluation.md` § 3.B.

---

## ADR-06 — Anti-leakage del p95 de monto

**Decisión**: la feature `amt_gt_p95_legit` (flag de monto > p95) se calcula con el p95 de transacciones **legítimas (label=0)** del split de **train** únicamente.

**Razón**: si calculamos el p95 sobre el dataset entero, incluye fraudes (que tienden a ser de monto alto), y el "techo habitual" queda inflado. Si lo calculamos sobre todas las transacciones de train (incluyendo fraudes), igual contamina. Solo legit-train da el techo de comportamiento "normal".

---

## ADR-07 — Optuna TPE + MedianPruner

**Decisión**: 50 trials con `TPESampler(multivariate=True)` + `MedianPruner(n_startup_trials=5, n_warmup_steps=20)` + `XGBoostPruningCallback` sobre `validation_0-aucpr`.

**Alternativas consideradas**:
- GridSearch — exponencial en el número de hiperparámetros.
- RandomSearch — competitivo pero menos eficiente que TPE en espacios grandes.
- Hyperopt / Ray Tune — más infraestructura, no aporta sobre Optuna en este alcance.

**Razón**: TPE construye un modelo probabilístico de los buenos hiperparámetros y los explora preferentemente. MedianPruner corta trials que en las primeras 20 iteraciones ya están por debajo de la mediana de los anteriores → ahorro de ~40% del tiempo de cómputo. `multivariate=True` modela correlaciones entre hiperparámetros (importante: `max_depth` y `min_child_weight` están correlacionados).

**Métrica primaria**: PR-AUC en val (no F1, no accuracy). En desbalance fuerte, PR-AUC es la métrica que mejor refleja la calidad del ranking.

---

## ADR-08 — Calibración isotónica vs Platt, fitteada en val

**Decisión**: probar Platt scaling + isotónica, elegir la que minimiza Brier en **val** (no test), persistir en `models/calibrator.pkl`.

**Alternativas consideradas**:
- No calibrar — el score crudo está distorsionado por `scale_pos_weight≈176`.
- Solo Platt — paramétrico (logit lineal), insuficiente cuando la mala calibración no es monotónica simple.
- Solo isotónica — non-parametric, puede sobreajustar con poca data.

**Razón**: probar las dos y elegir empíricamente es robusto. **Fittear en val** evita contaminar test. La isotónica ganó (Brier 0.0015 vs 0.0021 de Platt), persistida.

**Cuándo usar cada score**:
- **Score crudo** → ranking (top-k para revisión manual). PR-AUC y recall@1% son invariantes a calibración monotónica.
- **Score calibrado** → cuando se necesita interpretar como probabilidad real (auto-bloqueo, input a otro modelo, scoring de riesgo).

---

## ADR-09 — Threshold operativo en val, fijo para test

**Decisión**: threshold = 0.6642 (F1 max sobre la curva PR de val), aplicado fijo a test sin re-tunear.

**Razón**: re-elegir el threshold en test contamina la métrica de generalización. El threshold debe quedar fijo desde val, igual que el modelo.

**Alternativa para producción**: threshold de **mínimo costo = 0.52** (recomendado por el análisis costo-beneficio en `model_evaluation.md` § 1). El 0.6642 es académico (F1 max); el 0.52 es de negocio.

---

## ADR-10 — SHAP sobre sample de 5,000 filas

**Decisión**: `shap.TreeExplainer` sobre 5,000 filas muestreadas aleatoriamente del test set, no sobre las 194k completas.

**Alternativas consideradas**:
- Test entero (194k) — explosión de RAM (matrix de 194k × 27 SHAP values).
- Sample muy chico (500) — varianza alta en el top-features.

**Razón**: 5,000 filas es suficiente para que mean(|SHAP|) por feature converja (ley de los grandes números) y mantiene la matriz manejable (~500 KB). Para los force plots se eligen top-3 TPs y top-3 FPs por score dentro del sample.

**Validación**: tests verifican la propiedad de aditividad — `sum(SHAP) + base_value ≈ logit(score)` con tolerancia 1e-4.

---

## ADR-11 — MLflow con backend local file://

**Decisión**: `mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")`, sin server remoto.

**Razón**: el alcance es un capstone single-developer. File backend es trivial de versionar (aunque `mlruns/` está gitignoreado por peso) y suficiente para reproducir runs.

**Para producción**: migraría a un MLflow server con backend Postgres + S3/GCS para artefactos.

---

## ADR-12 — `cc_num` como llave de agregación, nunca como feature

**Decisión**: `cc_num` se usa para agrupar (rolling, distancia consecutiva) pero se dropea antes de pasarle al modelo.

**Razón**: si `cc_num` entra como feature, el modelo memoriza tarjetas individuales — no generaliza a tarjetas nuevas en producción. Es un anti-pattern típico en datasets transaccionales.

**Validación**: test [`tests/test_train.py::test_no_target_in_features`](../tests/test_train.py) verifica que `is_fraud` y `split` no aparezcan en X — extender a `cc_num` sería trivial pero ya está garantizado por la lista `EXTRA_DROP` en [`src/features.py`](../src/features.py).

---

## ADR-13 — Costos: FN = monto, FP = $5 fijo

**Decisión**: en el sweep de threshold, `costo_FN = amt_tx` y `costo_FP = $5`.

**Justificación de los valores**:
- **FN = monto**: si el fraude pasa, la pérdida es el monto completo de la transacción (asumimos sin recuperación — conservador).
- **FP = $5**: costo aproximado de revisión manual por un analista de fraude (5 minutos × salario).

**Sensibilidad**: si FP fuera $50 en lugar de $5, el threshold óptimo subiría (compensaría mejor por menos alertas). En producción habría que medir el costo real (chargeback fees, customer churn, costo de customer service) y recalcular periódicamente.

---

## ADR-14 — API containerizada + UI Streamlit local + UI lee parquet directo

**Decisión**:
1. El API (FastAPI + uvicorn) corre en su **propio container Docker** (`fraud-api`). Modelo y calibrador van **embebidos en la imagen**.
2. La UI (Streamlit) corre **local**, NO en Docker. Se conecta al API por HTTP via `API_URL` (env var).
3. La UI lee `data/curated/transactions_features.parquet` y `data/staging/transactions.parquet` **directo de disco** para cargar transacciones de ejemplo. **No existe** un endpoint `/samples` en el API.
4. El API expone exactamente 3 endpoints: `GET /healthz`, `POST /score`, `POST /score_batch`. Nada más.

**Razón**:

- **API container**: el modelo + calibrador + TreeExplainer son artefactos pesados (~30 MB de modelo, dependencias Python ~700 MB). Quemarlos en una imagen permite que cualquier consumer (UI, batch script, integraciones futuras) hablen el mismo protocolo HTTP sin replicar la carga. CPU-only para portabilidad — el modelo entrena con CUDA (Fase 3) pero infiere fino en CPU.
- **UI local**: Streamlit en Docker es overkill para una demo. Bind del puerto, watch mode, compartir el filesystem para los parquets — todo se vuelve fricción. Local es un comando (`streamlit run app.py`) y el desarrollador tiene hot-reload nativo.
- **UI lee parquet directo**: agregar `/samples` al API mezcla responsabilidades: el API pasa de ser "inferencia" a ser "data + inferencia". En producción los samples vendrían de un sistema separado (kafka, snowflake, etc.) — la UI demo replica eso leyendo parquet local. Mantener el API minimal facilita reuso.
- **Solo 3 endpoints**: scope-creep es el enemigo del software bien diseñado. El API hace una cosa: scorea features.

**Trade-offs**:

- La UI necesita acceso al parquet, no es portable a otra máquina sin los datos. Aceptable para una demo capstone.
- El batch_score.py se conecta al API por HTTP en lugar de cargar el modelo localmente. Es 10× más lento pero garantiza que **la lógica de scoring es la misma** que producción (anti-skew). El sanity check de PR-AUC = 0.8771 sobre los 194,502 rows del test set se reproduce end-to-end.

**Threshold semantics**: `decision_operating` (0.6642) y `decision_cost` (0.52) se aplican al **score crudo**, no al calibrado. Es lo mismo que hace [`src/evaluate.py`](../src/evaluate.py). El score calibrado se reporta como probabilidad interpretable pero no decide; ver ADR-09.

**Manejo de NaN**: las features `rolling_*` y `te_*` admiten `null` JSON (sentinel para NaN) — XGBoost los maneja como missing-value durante la inferencia. Sin esto el ~98 % del test set sería inutilizable (cualquier primera tx en una ventana tiene std/count indefinidos). Las features no-nullable (`hour`, `dow`, `is_night`, `amt_gt_p95_legit`) provienen del timestamp y siempre existen.

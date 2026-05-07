# fraud_ui — Demo Streamlit local

UI mínima para mostrar el modelo en vivo. Carga muestras del test set
(usando los parquets locales), las envía al API por HTTP y renderiza
score, decisión y SHAP top-5.

> **No corre en Docker**. La UI lee los parquets directo de disco y la
> simplicidad de un comando local supera el valor de una imagen extra.
> Ver [ADR-14](../docs/architecture_decisions.md#adr-14--api-containerizada--ui-streamlit-local--ui-lee-parquet-directo).

## Setup

```bash
pip install -r fraud_ui/requirements.txt
```

## Correr

```bash
# El API tiene que estar levantado primero (make api-local | api-docker-run)
make ui
# o:
cd fraud_ui && streamlit run app.py
```

Default: la UI conecta a `http://localhost:8000`. Override con env var:

```bash
API_URL=http://otra-maquina:8000 streamlit run app.py
```

También se puede cambiar el `API_URL` en runtime desde la sidebar.

## Flujo

1. **Sidebar**: configurás `API_URL`, probás conexión, toggle de SHAP.
2. **Selector de tx**: Random fraude / Random legítima / Por índice.
3. **Card** con metadatos legibles: monto, merchant, hora, edad, velocidad.
4. **Expander** con las 27 features que recibe el modelo.
5. **Predecir fraude**: POST `/score`, espera ~25 ms.
6. **Resultado**:
   - Probabilidad calibrada con delta vs 0.5.
   - Score crudo + barra de progreso vs threshold operativo.
   - Decisiones a ambos thresholds (operativo F1* = 0.6642 y costo = 0.52).
   - "Acertó / Falló" comparando con la label real.
   - SHAP top-5 como bar chart horizontal.
   - JSON crudo en un expander.

## Layout

```
fraud_ui/
├── README.md
├── requirements.txt    # streamlit + requests + pandas + pyarrow
├── app.py              # Streamlit entrypoint
├── data_loader.py      # carga + alineamiento de samples del test set
└── api_client.py       # wrapper requests con retry y timeout
```

## Notas técnicas

- **Alineamiento posicional** entre `data/curated/transactions_features.parquet`
  y `data/staging/transactions.parquet`: replica el sort `(cc_num, unix_time)`
  que hace `src/features.py` después del split temporal. Si el orden se rompe
  (e.g. si re-corrés `make features` con un seed distinto), el data_loader
  lanza `RuntimeError`.
- **NaN en features**: la UI envía `null` al API para las features
  `rolling_*` y `te_*` cuando son NaN — XGBoost las maneja como missing-value.
- **`@st.cache_data`** sobre `load_test_samples`: el sample de 250 tx se
  computa una sola vez por sesión.

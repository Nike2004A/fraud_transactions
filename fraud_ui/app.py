"""Streamlit UI: dropdown → POST /score → render score, decisión, SHAP.

Corre LOCAL (no en Docker). Lee parquets directo de ``data/`` y habla con
el API por HTTP. Configurable vía env var ``API_URL``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from api_client import ApiError, health, score_one
from data_loader import (
    FEATURE_NAMES,
    features_dict,
    load_test_samples,
    random_fraud_idx,
    random_legit_idx,
)

st.set_page_config(page_title="Fraud Detection — demo", page_icon=":mag:", layout="wide")

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data(show_spinner="Cargando muestras del test set...")
def _samples_cached() -> pd.DataFrame:
    return load_test_samples(n_fraud=50, n_legit=200, seed=42)


def _format_amt(x: float) -> str:
    return f"${x:,.2f}"


def _decision_box(label: str, decision: dict, real_is_fraud: bool) -> None:
    """Render una decisión con contraste explícito (funciona en theme dark)."""
    is_fraud = decision["is_fraud"]
    thr = decision["threshold"]
    correct = is_fraud == bool(real_is_fraud)

    bg = "#1e7e34" if correct else "#a71d2a"      # verde / rojo saturados
    border = "#28a745" if correct else "#dc3545"
    status_icon = "OK" if correct else "MISS"
    decision_label = "FRAUDE" if is_fraud else "legítima"

    st.markdown(
        f"""<div style="background:{bg};padding:14px 16px;border-radius:8px;
        border-left:6px solid {border};color:#ffffff;line-height:1.45;">
            <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;
                opacity:0.85;color:#ffffff;">{label} &mdash; thr {thr:.4f}</div>
            <div style="font-size:1.6rem;font-weight:700;color:#ffffff;margin-top:2px;">
                {decision_label}</div>
            <div style="font-size:0.85rem;color:#ffffff;opacity:0.95;margin-top:2px;">
                [{status_icon}] {'coincide con la label real' if correct else 'no coincide con la label real'}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _shap_chart(shap_top5: list[dict]) -> None:
    df = pd.DataFrame(shap_top5)
    df["color"] = df["direction"].map(
        {"increases_fraud": "#dc3545", "decreases_fraud": "#28a745"}
    )
    df = df.sort_values("contribution")  # eje vertical: mayor positivo arriba
    df["label"] = df.apply(lambda r: f"{r['feature']} = {r['value']:.3g}", axis=1)
    chart = (
        df[["label", "contribution"]]
        .set_index("label")
    )
    st.bar_chart(chart, horizontal=True, color="#1f77b4")
    st.caption("Contribución SHAP en margen (logit). Positivo → empuja hacia fraude.")


def main() -> None:
    st.title("Fraud Detection — demo capstone")
    st.caption(
        "Modelo XGBoost (Fase 3) + isotónica (Fase 4). Tasa base de fraude **0.58 %**. "
        "Decisiones se toman sobre el **score crudo** (rango [0.31, 0.68]) con "
        "thresholds **0.6642** (F1\\*) y **0.52** (costo mínimo). "
        "La probabilidad calibrada se interpreta vs la tasa base — su máximo empírico es ~0.35."
    )

    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        if st.button("Test connection", use_container_width=True):
            try:
                h = health(api_url)
                st.success(f"OK — calibrator={h['calibrator_kind']}, n_features={h['n_features']}")
            except ApiError as exc:
                st.error(str(exc))
        show_shap = st.toggle("Mostrar SHAP", value=True)
        st.divider()
        st.caption("Datos: `data/curated/transactions_features.parquet` + staging")

    samples = _samples_cached()
    st.write(f"Muestra cargada: **{len(samples)}** tx (fraudes: {int(samples['is_fraud'].sum())})")

    # Selector de modo
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = int(samples.index[0])

    rng = np.random.default_rng()
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Random FRAUDE", use_container_width=True):
            st.session_state.current_idx = random_fraud_idx(samples, rng)
    with col_b:
        if st.button("Random legítima", use_container_width=True):
            st.session_state.current_idx = random_legit_idx(samples, rng)
    with col_c:
        max_idx = int(samples.index.max())
        chosen = st.number_input(
            "Por índice",
            min_value=0,
            max_value=max_idx,
            value=st.session_state.current_idx,
            step=1,
        )
        if chosen != st.session_state.current_idx:
            st.session_state.current_idx = int(chosen)

    row = samples.loc[st.session_state.current_idx]
    real_is_fraud = bool(row["is_fraud"])

    # Card con metadatos legibles
    st.markdown("### Transacción")
    info_cols = st.columns(4)
    info_cols[0].metric("Monto", _format_amt(float(row.get("amt", float("nan")))))
    info_cols[1].metric("Hora", f"{int(row['hour']):02d}:00")
    info_cols[2].metric("Edad", f"{float(row['age']):.0f}")
    info_cols[3].metric("Velocidad", f"{float(row['velocity_kmh']):.1f} km/h")
    if "merchant" in row.index:
        st.write(
            f"**Merchant:** {row['merchant']} &nbsp;|&nbsp; "
            f"**Categoría:** `{row['category']}` &nbsp;|&nbsp; "
            f"**Estado:** {row.get('state', '?')} &nbsp;|&nbsp; "
            f"**Real:** {'**FRAUDE**' if real_is_fraud else 'legítima'}"
        )

    with st.expander("Ver las 27 features que recibe el modelo"):
        feat_view = pd.DataFrame(
            [(f, float(row[f])) for f in FEATURE_NAMES],
            columns=["feature", "value"],
        )
        st.dataframe(feat_view, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("Predecir fraude", type="primary", use_container_width=True):
        try:
            features = features_dict(row)
            with st.spinner("POST /score..."):
                resp = score_one(api_url, features)
        except ApiError as exc:
            st.error(f"Falló el API: {exc}")
            return

        st.markdown("### Resultado")
        score_cal = resp["score_calibrated"]
        score_raw = resp["score_raw"]
        op_thr = resp["decision_operating"]["threshold"]

        # Tasa base de fraude del dataset (Fase 0 EDA). El score calibrado
        # comparado contra 0.5 es engañoso porque la base rate es 0.58 %:
        # 0.35 calibrado ya es ~60× la base rate y eso es altísimo.
        BASE_RATE = 0.0058
        lift_vs_base = score_cal / BASE_RATE if BASE_RATE > 0 else 0

        m_cols = st.columns([2, 1, 1])
        m_cols[0].metric(
            "Probabilidad calibrada",
            f"{score_cal:.4f}",
            delta=f"{lift_vs_base:.0f}x vs tasa base ({BASE_RATE:.2%})",
            delta_color="off",
        )
        m_cols[1].metric("Score crudo", f"{score_raw:.4f}")
        m_cols[2].metric("Real", "FRAUDE" if real_is_fraud else "legítima")

        # El score crudo está comprimido en ~[0.31, 0.68] por scale_pos_weight=176.
        # Mapeamos linealmente a [0, 1] para que la barra cubra el rango útil
        # y el thr operativo caiga visualmente cerca del 75% (no del 66%).
        RAW_LO, RAW_HI = 0.30, 0.70
        progress = min(max((score_raw - RAW_LO) / (RAW_HI - RAW_LO), 0.0), 1.0)
        thr_marker = (op_thr - RAW_LO) / (RAW_HI - RAW_LO)
        st.progress(
            progress,
            text=f"score_raw = {score_raw:.4f}  |  thr operativo = {op_thr:.4f} "
                 f"(escala visual: rango útil del modelo [{RAW_LO}, {RAW_HI}])",
        )
        st.caption(
            f"Marcador del threshold operativo: ~{thr_marker:.0%} de la barra. "
            "Por encima → el modelo decide FRAUDE."
        )

        d_cols = st.columns(2)
        with d_cols[0]:
            _decision_box("Operativo (F1*)", resp["decision_operating"], real_is_fraud)
        with d_cols[1]:
            _decision_box("Costo mínimo", resp["decision_cost"], real_is_fraud)

        # El detalle "OK / MISS" ya viene en cada decision_box.

        if show_shap:
            st.markdown("### Top-5 SHAP (contribución a este score)")
            _shap_chart(resp["shap_top5"])

        with st.expander("JSON crudo del response"):
            st.json(resp)


if __name__ == "__main__":
    main()

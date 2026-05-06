"""Tests de Fase 4 — evaluación.

Tres garantías mínimas:

1. **Round-trip de modelo**: el modelo guardado y cargado predice las
   mismas probabilidades que el original sobre los mismos datos
   (smoke test, dataset chico para no quemar tiempo).
2. **Calibración mejora Brier**: en val, al menos uno de Platt o
   isotónica iguala o mejora el Brier crudo (no es garantizado en
   teoría con datasets sintéticos pequeños, pero sí en la práctica
   cuando el modelo está mal calibrado por ``scale_pos_weight`` ≠ 1).
3. **SHAP suma a score**: para una muestra chica, ``sum(SHAP) +
   base_value ≈ logit(score)`` con tolerancia 1e-4 (propiedad
   fundamental de TreeExplainer en margen).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import brier_score_loss

from src import config, evaluate


# --- helpers --- #


def _has_cuda() -> bool:
    try:
        from src import train
        train._assert_cuda_available()
        return True
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _has_cuda(), reason="XGBoost CUDA no disponible")


def _make_synthetic_split(n: int = 4000, seed: int = 0):
    """Genera un split sintético chico con señal real."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 6)).astype("float32")
    # Señal: f0 + f1 dominan
    logit = -3.5 + 2.0 * X[:, 0] + 1.2 * X[:, 1] - 0.8 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, size=n) < p).astype(int)
    cols = [f"f{i}" for i in range(X.shape[1])]
    Xdf = pd.DataFrame(X, columns=cols)
    return Xdf, y


def _train_small_clf(Xdf, y, **params) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        eval_metric="aucpr",
        verbosity=0,
        device=config.XGBOOST_DEVICE,
        tree_method=config.XGBOOST_TREE_METHOD,
        random_state=0,
        scale_pos_weight=10.0,  # induce mala calibración para que Platt/iso ayuden
        **params,
    )
    clf.fit(Xdf, y, verbose=False)
    return clf


# --- tests --- #


@cuda_only
def test_load_model_predicts(tmp_path):
    """Smoke: serializar un XGBClassifier y volver a cargarlo da los
    mismos predict_proba. Cubre que ``load_model`` reconstruye un
    objeto funcional con la sklearn API.
    """
    Xdf, y = _make_synthetic_split(n=2000)
    clf = _train_small_clf(Xdf, y)
    proba_orig = clf.predict_proba(Xdf)[:, 1]

    model_path = tmp_path / "tmp.json"
    clf.save_model(str(model_path))

    loaded = evaluate.load_model(model_path)
    proba_loaded = loaded.predict_proba(Xdf)[:, 1]

    np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-5, atol=1e-6)


@cuda_only
def test_calibration_improves(tmp_path):
    """Al menos uno entre Platt e isotónica baja el Brier en val cuando
    el modelo fue entrenado con ``scale_pos_weight`` distinto de 1
    (caso típico fraude). Con un modelo bien calibrado puede no
    haber mejora — por eso forzamos scale_pos_weight=10 en el
    classifier sintético.
    """
    Xdf_train, y_train = _make_synthetic_split(n=4000, seed=0)
    Xdf_val, y_val = _make_synthetic_split(n=1500, seed=1)
    Xdf_test, y_test = _make_synthetic_split(n=1500, seed=2)

    clf = _train_small_clf(Xdf_train, y_train)
    val_score = clf.predict_proba(Xdf_val)[:, 1]
    test_score = clf.predict_proba(Xdf_test)[:, 1]

    calib = evaluate.calibration_analysis(val_score, y_val, test_score, y_test)
    # Brier en val (no test) es la métrica de selección.
    val_platt = evaluate._apply_platt(evaluate._fit_platt(val_score, y_val), val_score)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val_score, y_val)
    val_iso = iso.predict(val_score)
    brier_val_raw = brier_score_loss(y_val, val_score)
    brier_val_platt = brier_score_loss(y_val, val_platt)
    brier_val_iso = brier_score_loss(y_val, val_iso)

    assert min(brier_val_platt, brier_val_iso) <= brier_val_raw + 1e-9, (
        f"Ni Platt ({brier_val_platt:.5f}) ni iso ({brier_val_iso:.5f}) "
        f"mejoran Brier crudo ({brier_val_raw:.5f}) en val."
    )


@cuda_only
def test_shap_sums_to_score(tmp_path):
    """Propiedad de aditividad de TreeExplainer en margen:
    ``sum(SHAP_i) + base_value == logit(score)`` para cada fila.

    Tolerancia 1e-4: TreeExplainer es exacto sobre el árbol pero el
    margen final pasa por sigmoid en predict_proba, así que reconstruimos
    el margen con ``output_margin=True`` para no perder precisión.
    """
    Xdf, y = _make_synthetic_split(n=600, seed=42)
    clf = _train_small_clf(Xdf, y)
    booster = clf.get_booster()
    booster.set_param({"device": "cpu"})  # SHAP TreeExplainer en CPU

    sample = Xdf.iloc[:64]
    margin = booster.predict(xgb.DMatrix(sample), output_margin=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = __import__("shap").TreeExplainer(booster)
        shap_values = explainer.shap_values(sample)
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    reconstructed = shap_values.sum(axis=1) + base_value
    np.testing.assert_allclose(reconstructed, margin, rtol=1e-4, atol=1e-4)

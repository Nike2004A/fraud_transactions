"""Carga del modelo + calibrador + TreeExplainer y rutinas de predicción.

Convenciones críticas (heredadas de Fase 4):

- Cargar XGBoost vía ``XGBClassifier.load_model``, NO vía ``Booster.load_model``:
  el segundo respeta el ``best_iteration`` guardado (= 1 en este artefacto)
  y predice con un solo árbol; el wrapper sklearn usa los 32 árboles
  entrenados.
- TreeExplainer se ejecuta sobre el booster forzado a ``device='cpu'`` para
  ser portable (el container no tiene CUDA).
- Calibrador soporta dict ``{'kind': 'isotonic'|'platt', 'model': ...}``:
  isotónica usa ``model.predict``; Platt replica ``_apply_platt`` de
  ``src/evaluate.py`` (clip → logit → ``LR.predict_proba``).
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from .schema import FeaturesIn
from .settings import (
    CALIBRATOR_PATH,
    COST_THRESHOLD,
    FEATURE_NAMES,
    MODEL_PATH,
    OPERATING_THRESHOLD,
)


@dataclass
class ModelBundle:
    clf: xgb.XGBClassifier
    calibrator_kind: str
    calibrator: object  # IsotonicRegression | LogisticRegression
    explainer: shap.TreeExplainer
    base_value: float


def _apply_platt(lr: LogisticRegression, score: np.ndarray) -> np.ndarray:
    eps = 1e-6
    s = np.clip(score, eps, 1 - eps)
    z = np.log(s / (1 - s)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def calibrate(score_raw: np.ndarray, kind: str, model: object) -> np.ndarray:
    """Aplica el calibrador entrenado en val (Fase 4)."""
    if kind == "isotonic":
        return np.asarray(model.predict(score_raw), dtype=float)  # type: ignore[union-attr]
    if kind == "platt":
        return _apply_platt(model, score_raw)  # type: ignore[arg-type]
    raise ValueError(f"Calibrador desconocido: {kind!r}")


def load_bundle(
    model_path: Path = MODEL_PATH,
    calibrator_path: Path = CALIBRATOR_PATH,
) -> ModelBundle:
    """Carga modelo, calibrador y TreeExplainer una sola vez (startup).

    Raises:
        FileNotFoundError: si falta algún artefacto.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrador no encontrado en {calibrator_path}")

    clf = xgb.XGBClassifier(device="cpu", tree_method="hist")
    clf.load_model(str(model_path))

    booster = clf.get_booster()
    booster.set_param({"device": "cpu"})

    with calibrator_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "kind" not in payload or "model" not in payload:
        raise ValueError(
            f"calibrator.pkl tiene formato inesperado (esperado dict con kind/model): {type(payload)}"
        )
    cal_kind = payload["kind"]
    cal_model = payload["model"]
    if cal_kind not in ("isotonic", "platt"):
        raise ValueError(f"Tipo de calibrador no soportado: {cal_kind!r}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(booster)
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    return ModelBundle(
        clf=clf,
        calibrator_kind=cal_kind,
        calibrator=cal_model,
        explainer=explainer,
        base_value=base_value,
    )


def features_to_dataframe(rows: Iterable[FeaturesIn]) -> pd.DataFrame:
    """Convierte una lista de ``FeaturesIn`` en DataFrame con orden estable.

    Los ``None`` (que vinieron como JSON null) se convierten a ``np.nan``
    para que XGBoost los trate como missing-value durante la inferencia.
    """
    data = [r.model_dump() for r in rows]
    df = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    # Forzar columnas a float para que None → NaN. Los enteros lógicos
    # (hour, dow, is_night, amt_gt_p95_legit) son non-nullable, no afecta.
    return df.astype(float, errors="ignore")


def _decision(score: float, threshold: float) -> dict:
    return {"threshold": float(threshold), "is_fraud": bool(score >= threshold)}


def predict_batch(bundle: ModelBundle, X: pd.DataFrame) -> list[dict]:
    """Predicción sin SHAP. Devuelve scores y decisiones a ambos thresholds.

    Los thresholds 0.6642 (F1*) y 0.52 (mín costo) se barrieron en Fase 3/4
    sobre el score CRUDO, así que las decisiones se evalúan contra
    ``score_raw``. El calibrado se reporta para interpretabilidad como
    probabilidad pero NO se usa para decisiones (ver ADR-09 / ADR-14).
    """
    score_raw = bundle.clf.predict_proba(X)[:, 1]
    score_cal = calibrate(score_raw, bundle.calibrator_kind, bundle.calibrator)
    out: list[dict] = []
    for raw, cal in zip(score_raw, score_cal):
        out.append(
            {
                "score_raw": float(raw),
                "score_calibrated": float(cal),
                "decision_operating": _decision(float(raw), OPERATING_THRESHOLD),
                "decision_cost": _decision(float(raw), COST_THRESHOLD),
            }
        )
    return out


def shap_top5(bundle: ModelBundle, X: pd.DataFrame) -> list[dict]:
    """SHAP top-5 features de la primera (y única) fila de ``X``.

    El TreeExplainer en margen (logit) entrega contribuciones en el espacio
    aditivo del booster; reportamos su signo: positivo → empuja hacia
    fraude.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = bundle.explainer.shap_values(X)
    sv = np.asarray(shap_values)[0]
    feat_names = list(X.columns)
    feat_vals = X.iloc[0].to_numpy()

    order = np.argsort(-np.abs(sv))[:5]
    out = []
    for idx in order:
        contrib = float(sv[idx])
        out.append(
            {
                "feature": feat_names[idx],
                "value": float(feat_vals[idx]),
                "contribution": contrib,
                "direction": "increases_fraud" if contrib >= 0 else "decreases_fraud",
            }
        )
    return out


def predict_one(bundle: ModelBundle, features: FeaturesIn) -> dict:
    """Predicción para una sola tx, incluyendo SHAP top-5."""
    X = features_to_dataframe([features])
    base = predict_batch(bundle, X)[0]
    base["shap_top5"] = shap_top5(bundle, X)
    base["base_value"] = bundle.base_value
    return base

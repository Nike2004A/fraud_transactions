"""Constantes de runtime del API.

Los thresholds vienen fijos de Fase 3/4 — no se recalculan en runtime.
Las paths admiten override por env var para soportar tanto el container
(``/app/models``) como ejecución local (``../models``).
"""
from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


_HERE = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = _HERE.parent / "models"
if not _DEFAULT_MODELS_DIR.exists():
    # Ejecución local desde la raíz del repo: usar models/ del proyecto.
    _DEFAULT_MODELS_DIR = _HERE.parents[1] / "models"

MODELS_DIR: Path = _path_from_env("FRAUD_API_MODELS_DIR", _DEFAULT_MODELS_DIR)
MODEL_PATH: Path = _path_from_env("FRAUD_API_MODEL_PATH", MODELS_DIR / "xgb_best.json")
CALIBRATOR_PATH: Path = _path_from_env(
    "FRAUD_API_CALIBRATOR_PATH", MODELS_DIR / "calibrator.pkl"
)

OPERATING_THRESHOLD: float = 0.6642
COST_THRESHOLD: float = 0.52

MAX_BATCH_SIZE: int = 1000

# Orden EXACTO de las 27 features que ve el modelo (Fase 2/3, no tocar).
FEATURE_NAMES: tuple[str, ...] = (
    "log1p_amt",
    "amt_gt_p95_legit",
    "hour",
    "dow",
    "is_night",
    "hour_sin",
    "hour_cos",
    "age",
    "time_since_last_tx",
    "dist_consecutive_km",
    "velocity_kmh",
    "rolling_count_1h",
    "rolling_amt_sum_1h",
    "rolling_amt_mean_1h",
    "rolling_amt_std_1h",
    "rolling_count_24h",
    "rolling_amt_sum_24h",
    "rolling_amt_mean_24h",
    "rolling_amt_std_24h",
    "rolling_count_7d",
    "rolling_amt_sum_7d",
    "rolling_amt_mean_7d",
    "rolling_amt_std_7d",
    "te_merchant",
    "te_category",
    "te_state",
    "te_job",
)

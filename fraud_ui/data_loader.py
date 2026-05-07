"""Carga muestras del test set para la UI.

Une las features curadas (las 27 que ve el modelo + ``is_fraud``) con
metadatos legibles del staging (merchant, category, amt, cc_num, edad).
La unión es **posicional** porque el feature pipeline replica el sort
``(cc_num, unix_time)`` después del split temporal — ver
``src/features.py`` y ``src/evaluate._load_aligned_test_metadata``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Hard-coded. Si la UI corre desde otro CWD, ajustar via env var.
import os

_REPO_ROOT = Path(os.environ.get("FRAUD_REPO_ROOT", Path(__file__).resolve().parents[1]))
CURATED_PATH: Path = _REPO_ROOT / "data" / "curated" / "transactions_features.parquet"
STAGING_PATH: Path = _REPO_ROOT / "data" / "staging" / "transactions.parquet"

# Mismas constantes que src/config.py — duplicadas para mantener la UI
# desacoplada del paquete src.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

FEATURE_NAMES: tuple[str, ...] = (
    "log1p_amt", "amt_gt_p95_legit", "hour", "dow", "is_night", "hour_sin",
    "hour_cos", "age", "time_since_last_tx", "dist_consecutive_km",
    "velocity_kmh", "rolling_count_1h", "rolling_amt_sum_1h",
    "rolling_amt_mean_1h", "rolling_amt_std_1h", "rolling_count_24h",
    "rolling_amt_sum_24h", "rolling_amt_mean_24h", "rolling_amt_std_24h",
    "rolling_count_7d", "rolling_amt_sum_7d", "rolling_amt_mean_7d",
    "rolling_amt_std_7d", "te_merchant", "te_category", "te_state", "te_job",
)


def _staging_test_metadata(stg: pd.DataFrame) -> pd.DataFrame:
    """Replica el alineamiento del split test usado en evaluate.py."""
    stg = stg.sort_values("unix_time", kind="stable").reset_index(drop=True)
    n = len(stg)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    stg["split"] = split
    stg = stg.sort_values(["cc_num", "unix_time"], kind="stable").reset_index(drop=True)
    cols = [
        "merchant", "category", "amt", "cc_num", "trans_date_trans_time",
        "city", "state", "job", "dob", "is_fraud",
    ]
    cols = [c for c in cols if c in stg.columns]
    test = stg.loc[stg["split"] == "test", cols].reset_index(drop=True)
    return test


def load_test_samples(
    n_fraud: int = 50,
    n_legit: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Devuelve un sample aleatorio del test set con features + metadatos.

    Returns:
        DataFrame con las 27 features + ``is_fraud`` (de curated) y los
        metadatos legibles de staging. Sin NaN en las features (XGBoost
        las maneja, pero JSON no las puede serializar).
    """
    if not CURATED_PATH.exists() or not STAGING_PATH.exists():
        raise FileNotFoundError(
            f"Faltan los parquets — {CURATED_PATH} y/o {STAGING_PATH}"
        )

    cur = pd.read_parquet(CURATED_PATH)
    cur_test = cur[cur["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    stg = pd.read_parquet(STAGING_PATH)
    meta = _staging_test_metadata(stg)

    if len(cur_test) != len(meta):
        raise RuntimeError(
            f"Mismatch curated_test={len(cur_test)} vs staging_test={len(meta)}. "
            "El alineamiento posicional se rompió."
        )
    if not np.array_equal(cur_test["is_fraud"].to_numpy(), meta["is_fraud"].to_numpy()):
        raise RuntimeError(
            "Labels de curated test ≠ labels de staging test (orden roto)."
        )

    # Drop is_fraud duplicada (está en cur_test).
    meta = meta.drop(columns=["is_fraud"])
    joined = pd.concat([cur_test.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    # No filtramos por NaN: ``features_dict`` los serializa como None y el
    # API los acepta (XGBoost trata NaN como missing-value).

    rng = np.random.default_rng(seed)
    fraud_idx = joined.index[joined["is_fraud"] == 1].to_numpy()
    legit_idx = joined.index[joined["is_fraud"] == 0].to_numpy()
    pick_fraud = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
    pick_legit = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    sample_idx = np.sort(np.concatenate([pick_fraud, pick_legit]))
    return joined.iloc[sample_idx].reset_index(drop=True)


# Features que XGBoost trata como missing-value (la primera tx de la cc en
# la ventana correspondiente da NaN en std/count). Se envían como JSON null.
_NULLABLE_FEATURES: frozenset[str] = frozenset({
    "time_since_last_tx", "dist_consecutive_km", "velocity_kmh",
    "rolling_count_1h", "rolling_amt_sum_1h", "rolling_amt_mean_1h",
    "rolling_amt_std_1h", "rolling_count_24h", "rolling_amt_sum_24h",
    "rolling_amt_mean_24h", "rolling_amt_std_24h", "rolling_count_7d",
    "rolling_amt_sum_7d", "rolling_amt_mean_7d", "rolling_amt_std_7d",
    "te_merchant", "te_category", "te_state", "te_job",
})


def features_dict(row: pd.Series) -> dict:
    """Extrae las 27 features de una fila lista para POST /score.

    NaN en features ``_NULLABLE_FEATURES`` se serializa como ``None``
    (el API lo acepta y XGBoost lo trata como missing-value).
    """
    out: dict = {}
    for name in FEATURE_NAMES:
        v = row[name]
        if isinstance(v, (np.integer,)):
            out[name] = int(v)
            continue
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            if name in _NULLABLE_FEATURES:
                out[name] = None
            else:
                raise ValueError(f"Feature {name!r} no admite NaN: {f}")
        else:
            out[name] = f
    return out


def random_fraud_idx(df: pd.DataFrame, rng: np.random.Generator) -> int:
    pool = df.index[df["is_fraud"] == 1].to_numpy()
    if len(pool) == 0:
        raise ValueError("Sample sin fraudes")
    return int(rng.choice(pool))


def random_legit_idx(df: pd.DataFrame, rng: np.random.Generator) -> int:
    pool = df.index[df["is_fraud"] == 0].to_numpy()
    if len(pool) == 0:
        raise ValueError("Sample sin legítimas")
    return int(rng.choice(pool))

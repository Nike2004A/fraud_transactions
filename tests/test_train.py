"""Tests de Fase 3 — entrenamiento.

Tres garantías mínimas:

1. **No solapamiento de splits**: cada fila está en *exactamente uno* de
   train/val/test (defensa contra shuffles accidentales en pipelines
   downstream).
2. **Baseline > piso trivial**: en un dataset sintético con señal genuina,
   el baseline supera ``5× tasa_fraude`` en PR-AUC (≈ piso del clasificador
   constante "siempre fraude" agregando un pelín de información).
3. **No leakage del target**: ``is_fraud`` no aparece nunca como feature
   pasada al modelo (sólo en y).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, train


# --- helpers --- #


def _make_fake_curated(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Genera un parquet curado mínimo con señal real para que el baseline
    no degenere. 1 % fraude, 5 features, splits 70/15/15.
    """
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    # Señal: si f1>1.5, prob fraude alta. Resto es ruido.
    f1 = rng.normal(0, 1, size=n).astype("float32")
    f2 = rng.normal(0, 1, size=n).astype("float32")
    f3 = rng.normal(0, 1, size=n).astype("float32")
    f4 = rng.normal(0, 1, size=n).astype("float32")
    f5 = rng.normal(0, 1, size=n).astype("float32")

    base_logit = -5.0 + 3.0 * (f1 > 1.5).astype(float) + 1.5 * f2
    p = 1.0 / (1.0 + np.exp(-base_logit))
    y = (rng.uniform(0, 1, size=n) < p).astype("int8")

    df = pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            config.TARGET_COLUMN: y,
        }
    )
    df["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return df


def _has_cuda() -> bool:
    """Detecta si XGBoost puede usar CUDA en este host."""
    try:
        train._assert_cuda_available()
        return True
    except RuntimeError:
        return False


cuda_only = pytest.mark.skipif(not _has_cuda(), reason="XGBoost CUDA no disponible")


# --- tests --- #


def test_load_splits_no_overlap(tmp_path):
    """Ninguna fila aparece en más de un split.

    Verificación básica de que el split por columna ``split`` particiona el
    dataset (cobertura completa + disjuntez).
    """
    df = _make_fake_curated(n=2000)
    parquet = tmp_path / "fake_curated.parquet"
    df.to_parquet(parquet)

    splits = train.load_curated(parquet)
    n_train, n_val, n_test = (
        len(splits.X_train),
        len(splits.X_val),
        len(splits.X_test),
    )
    assert n_train + n_val + n_test == len(df)

    # Indexes desde el parquet original deben ser disjuntos.
    df_reloaded = pd.read_parquet(parquet).reset_index(drop=True)
    train_idx = set(df_reloaded.index[df_reloaded["split"] == "train"])
    val_idx = set(df_reloaded.index[df_reloaded["split"] == "val"])
    test_idx = set(df_reloaded.index[df_reloaded["split"] == "test"])
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)


def test_no_target_in_features(tmp_path):
    """``is_fraud`` no aparece en X — sólo en y. La columna ``split`` tampoco."""
    df = _make_fake_curated(n=1000)
    parquet = tmp_path / "fake_curated.parquet"
    df.to_parquet(parquet)

    splits = train.load_curated(parquet)
    for X in (splits.X_train, splits.X_val, splits.X_test):
        assert config.TARGET_COLUMN not in X.columns
        assert "split" not in X.columns
    assert config.TARGET_COLUMN not in splits.feature_names


@cuda_only
def test_baseline_beats_dummy(tmp_path):
    """El baseline debe superar 5× la tasa de fraude en val (PR-AUC).

    Piso trivial: un modelo aleatorio o un constante daría ≈ tasa_fraude.
    Si el baseline no supera 5× ese piso sobre un dataset con señal
    sintética clara, algo está roto en el pipeline.
    """
    df = _make_fake_curated(n=8000, seed=42)
    parquet = tmp_path / "fake_curated.parquet"
    df.to_parquet(parquet)

    splits = train.load_curated(parquet)
    fraud_rate = splits.fraud_rate_val
    floor = fraud_rate * train.BASELINE_PR_AUC_FLOOR_FACTOR

    _, metrics = train.train_baseline(splits)
    assert metrics["pr_auc"] > floor, (
        f"Baseline PR-AUC {metrics['pr_auc']:.4f} no supera el piso trivial "
        f"{floor:.4f} (= 5× tasa_fraude={fraud_rate:.4f})"
    )

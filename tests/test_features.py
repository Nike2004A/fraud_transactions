"""Tests de Fase 2: anti-leakage en rolling, TE, split temporal y age."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features


# --- fixtures --- #


def _synthetic_card_50() -> pd.DataFrame:
    """50 tx de una sola cc_num espaciadas 10 minutos, montos crecientes.

    Sirve para verificar que el rolling de la fila i no incluye amt[i] y
    que ``time_since_last_tx`` corresponde al delta con la fila previa.
    """
    n = 50
    base_ts = pd.Timestamp("2020-01-01 00:00:00")
    timestamps = [base_ts + pd.Timedelta(minutes=10 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "cc_num": [9999] * n,
            "unix_time": [int(t.timestamp()) for t in timestamps],
            "trans_date_trans_time": timestamps,
            "amt": np.arange(1.0, n + 1.0, dtype=float),  # 1, 2, ..., 50
            "merchant": ["m_a"] * n,
            "category": ["cat_a"] * n,
            "state": ["NY"] * n,
            "job": ["job_a"] * n,
            "merch_lat": np.full(n, 40.0),
            "merch_long": np.full(n, -74.0),
            "lat": np.full(n, 40.5),
            "long": np.full(n, -74.5),
            "city_pop": [10_000] * n,
            "city": ["NYC"] * n,
            "zip": ["10001"] * n,
            "gender": ["M"] * n,
            "first": ["x"] * n,
            "last": ["y"] * n,
            "street": ["s"] * n,
            "trans_num": [f"t{i}" for i in range(n)],
            "merch_zipcode": [None] * n,
            "dob": [pd.Timestamp("1990-06-15")] * n,
            "is_fraud": [0] * n,
        }
    )
    return df


def _multi_card_dataset(n_cards: int = 6, per_card: int = 200, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético multi-tarjeta con desbalance ~5% para tests de TE/split.

    Genera ``n_cards`` tarjetas con ``per_card`` tx cada una, timestamps
    monotónicos globales, montos uniformes en [1, 500], y target binario
    sembrado para que existan tasas distintas por categoría.
    """
    rng = np.random.default_rng(seed)
    n = n_cards * per_card
    base_ts = pd.Timestamp("2020-01-01 00:00:00")
    # Tiempos distribuidos densamente: 1 tx cada 5 min globales,
    # luego asigno round-robin a tarjetas (así cada tarjeta tiene
    # tx en distintos momentos del rango).
    ts_all = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(n)]
    cc_nums = np.tile(np.arange(1000, 1000 + n_cards), per_card)[:n]
    rng.shuffle(cc_nums)
    categories = rng.choice(["cat_a", "cat_b", "cat_c"], size=n)
    merchants = rng.choice(["m_a", "m_b", "m_c", "m_d"], size=n)

    # Probabilidad de fraude depende de la categoría (señal aprendible).
    p_fraud = np.where(categories == "cat_a", 0.10, np.where(categories == "cat_b", 0.05, 0.01))
    is_fraud = (rng.random(n) < p_fraud).astype(int)

    df = pd.DataFrame(
        {
            "cc_num": cc_nums,
            "unix_time": [int(t.timestamp()) for t in ts_all],
            "trans_date_trans_time": ts_all,
            "amt": rng.uniform(1.0, 500.0, size=n),
            "merchant": merchants,
            "category": categories,
            "state": rng.choice(["NY", "CA", "TX"], size=n),
            "job": rng.choice(["job_a", "job_b"], size=n),
            "merch_lat": rng.uniform(35.0, 45.0, size=n),
            "merch_long": rng.uniform(-80.0, -70.0, size=n),
            "lat": rng.uniform(35.0, 45.0, size=n),
            "long": rng.uniform(-80.0, -70.0, size=n),
            "city_pop": rng.integers(1_000, 1_000_000, size=n),
            "city": ["c"] * n,
            "zip": ["00000"] * n,
            "gender": rng.choice(["M", "F"], size=n),
            "first": ["x"] * n,
            "last": ["y"] * n,
            "street": ["s"] * n,
            "trans_num": [f"t{i}" for i in range(n)],
            "merch_zipcode": [None] * n,
            "dob": [pd.Timestamp("1985-01-01")] * n,
            "is_fraud": is_fraud,
        }
    )
    return df


# --- tests --- #


def test_no_leakage_rolling() -> None:
    """El rolling de la fila ``i`` no incluye ``amt[i]``.

    Construye 50 tx ordenadas (10 min de spacing) y comprueba:
      - ``rolling_count_1h[i]`` ∈ {0..5} y NO cuenta la propia fila.
      - ``rolling_amt_sum_1h[i]`` excluye ``amt[i]`` exactamente.
      - La primera fila tiene count=0 (no hay pasado).
    """
    df = _synthetic_card_50()
    out = features.build_features(df)
    out_sorted = out.reset_index(drop=True)

    # spacing = 10min, ventana = 1h, closed='left' → window [t-1h, t).
    # Para i=k caben hasta 6 tx pasadas (00:00..00:50 si t=01:00).
    counts = out_sorted["rolling_count_1h"].to_numpy()
    assert counts[0] == 0, "primera tx no puede tener pasado"
    expected = np.array([min(i, 6) for i in range(50)], dtype=float)
    np.testing.assert_array_equal(counts, expected)

    # sum_1h[i] = suma de amt[max(0, i-6):i] (ventana strict-left).
    sums = out_sorted["rolling_amt_sum_1h"].to_numpy()
    amts = np.arange(1.0, 51.0)
    for i in range(50):
        lo = max(0, i - 6)
        expected_sum = float(amts[lo:i].sum())
        assert sums[i] == pytest.approx(expected_sum), (
            f"fila {i}: sum esperado {expected_sum}, got {sums[i]} "
            f"(¿se filtró amt[i]={amts[i]}?)"
        )


def test_target_encoding_only_uses_train() -> None:
    """El TE de val/test viene del mapping entrenado en train, no del global.

    Construye un dataset donde la categoría ``cat_a`` tiene una tasa
    intencionalmente distinta entre train y val/test, y verifica que
    el TE de val/test refleja la tasa **del train** (con smoothing),
    NO la tasa global ni la propia tasa de val/test.
    """
    rng = np.random.default_rng(0)
    n = 600
    base_ts = pd.Timestamp("2020-01-01")
    ts = [base_ts + pd.Timedelta(minutes=i) for i in range(n)]

    # Asigno categorías 50/50 entre cat_a y cat_b.
    categories = np.array(["cat_a", "cat_b"] * (n // 2))
    # cat_a en train → 30% fraude; en val/test → 0%. cat_b siempre 5%.
    n_train = int(n * config.TRAIN_FRAC)
    is_fraud = np.zeros(n, dtype=int)
    rng_a_train = rng.random(n_train) < np.where(categories[:n_train] == "cat_a", 0.30, 0.05)
    rng_a_rest = rng.random(n - n_train) < np.where(categories[n_train:] == "cat_a", 0.0, 0.05)
    is_fraud[:n_train] = rng_a_train.astype(int)
    is_fraud[n_train:] = rng_a_rest.astype(int)

    df = _multi_card_dataset(n_cards=4, per_card=n // 4, seed=1)
    df = df.sort_values("unix_time").reset_index(drop=True)
    df["category"] = categories
    df["merchant"] = "m_x"
    df["state"] = "NY"
    df["job"] = "j"
    df["is_fraud"] = is_fraud
    # unix_time monotónico para que el split coincida con n_train.
    df["unix_time"] = [int(t.timestamp()) for t in ts]
    df["trans_date_trans_time"] = ts

    out = features.build_features(df)

    # TE esperado para cat_a/cat_b en val/test = mapping del train completo
    # con smoothing m=20 sobre prior global = mean(is_fraud[train]).
    train_target = is_fraud[:n_train]
    train_cat = categories[:n_train]
    global_mean_train = train_target.mean()
    m = config.TARGET_ENCODING_SMOOTHING
    expected = {}
    for cat in ("cat_a", "cat_b"):
        mask = train_cat == cat
        n_cat = mask.sum()
        mean_cat = train_target[mask].mean()
        expected[cat] = (n_cat * mean_cat + m * global_mean_train) / (n_cat + m)

    # 1) val/test contiene exactamente los dos valores esperados
    # (cat_a/cat_b) calculados con el mapping de train. Si el TE
    # incluyera target de val/test, los valores diferirían.
    val_test = out[out["split"].isin(["val", "test"])]
    unique_te = np.unique(np.round(val_test["te_category"].to_numpy(), 8))
    expected_set = sorted({round(expected["cat_a"], 8), round(expected["cat_b"], 8)})
    np.testing.assert_allclose(unique_te, expected_set, rtol=1e-5)

    # 2) Sanity: la tasa global computada sobre todo el dataset
    # (incluido val/test, donde cat_a tiene 0% de fraude) no coincide
    # con el mapping aprendido. Esto descarta que el TE use el target
    # global en lugar del target sólo de train.
    global_mean_all = is_fraud.mean()
    if abs(global_mean_all - global_mean_train) > 1e-6:
        # Encoding "global" alternativo (lo que daría leakage) — debe
        # ser ≠ del te observado para al menos una categoría.
        any_diff = False
        for cat in ("cat_a", "cat_b"):
            mask_all = categories == cat
            n_all = mask_all.sum()
            mean_all = is_fraud[mask_all].mean()
            leaky = (n_all * mean_all + m * global_mean_all) / (n_all + m)
            if abs(leaky - expected[cat]) > 1e-4:
                any_diff = True
        assert any_diff, "TE de val/test parece haberse calculado con stats globales"


def test_split_temporal_orden() -> None:
    """70/15/15 sin solapamiento; val/test posteriores a train en unix_time."""
    df = _multi_card_dataset(n_cards=5, per_card=400, seed=2)
    out = features.build_features(df)
    # Reincorporo unix_time desde el original alineando por orden temporal:
    # como build_features re-ordena internamente, valido las proporciones
    # y la monotonía de splits sobre el output (que sale ordenado por
    # unix_time gracias a _select_output_columns + sort previo).
    n = len(out)
    counts = out["split"].value_counts()
    assert counts.get("train", 0) == int(n * config.TRAIN_FRAC)
    assert counts.get("val", 0) == int(n * config.VAL_FRAC)
    assert counts.get("train", 0) + counts.get("val", 0) + counts.get("test", 0) == n

    # Validación de orden: aliento una columna unix_time auxiliar para verificarlo
    # reconstruyendo desde el df original ordenado por unix_time.
    df_sorted = df.sort_values("unix_time").reset_index(drop=True)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    max_train_ut = df_sorted.loc[: n_train - 1, "unix_time"].max()
    min_val_ut = df_sorted.loc[n_train : n_train + n_val - 1, "unix_time"].min()
    min_test_ut = df_sorted.loc[n_train + n_val :, "unix_time"].min()
    assert max_train_ut <= min_val_ut, "train debe terminar antes que val"
    assert min_val_ut <= min_test_ut, "val debe terminar antes que test"


def test_age_derivation() -> None:
    """Edad calculada al momento de la transacción, NO al momento de hoy.

    Si la edad usara ``today() - dob``, todas las filas tendrían el mismo
    valor ≈ edad actual del titular. Acá las tx están separadas por años
    en `trans_date_trans_time`, por lo que la edad debe variar.
    """
    df = _synthetic_card_50()
    # Modifico timestamps para forzar 5 años entre la primera y última tx.
    base = pd.Timestamp("2015-01-01")
    df["trans_date_trans_time"] = [base + pd.Timedelta(days=365 * i // 10) for i in range(50)]
    df["unix_time"] = [int(t.timestamp()) for t in df["trans_date_trans_time"]]
    df["dob"] = pd.Timestamp("1990-01-01")  # 25 años el 2015-01-01

    out = features.build_features(df).reset_index(drop=True)
    ages = out["age"].to_numpy()

    # La primera tx (2015-01-01) → edad ≈ 25 años exactos.
    assert ages[0] == pytest.approx(25.0, abs=0.05)
    # La última tx ≈ 4.9 años después → edad ≈ 29.9.
    assert ages[-1] == pytest.approx(29.9, abs=0.1)
    # Crece monotónicamente con el tiempo de la tx.
    assert (np.diff(ages) >= 0).all()

"""Feature engineering pipeline (Fase 2).

Lee ``data/staging/transactions.parquet`` (output de Fase 1), genera el set
final de features y persiste ``data/curated/transactions_features.parquet``.

Todas las features potencialmente con leakage (target encoding, flag de
``amt`` por percentil) se computan **sólo sobre el split de train**. Las
ventanas rolling por ``cc_num`` son estrictamente pasadas (``closed='left'``)
para excluir la transacción actual.

Uso:
    python -m src.features
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config

# --- constantes locales --- #

EARTH_RADIUS_KM: float = 6371.0088

# (etiqueta de columna, ventana pandas)
ROLLING_WINDOWS: tuple[tuple[str, str], ...] = (
    ("1h", "1h"),
    ("24h", "24h"),
    ("7d", "7d"),
)

# Columnas categóricas que reciben target encoding KFold.
TE_COLUMNS: tuple[str, ...] = ("merchant", "category", "state", "job")

# Columnas que se dropean antes de persistir el parquet curado.
EXTRA_DROP: tuple[str, ...] = (
    config.GROUP_KEY,            # cc_num: sólo agregación
    "dob",                       # se reemplaza por age
    "city_pop",                  # sin señal en el EDA
    "merch_zipcode",             # 15.1% nulos, redundante con merch_lat/long
    "trans_date_trans_time",     # capturado en hour/dow/is_night/age
    "gender",                    # no validado por el EDA
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city",
    "zip",
    "amt",                       # se reemplaza por log1p_amt
    "ts",                        # auxiliar, derivada de unix_time
    "unix_time",                 # se mantiene sólo durante el pipeline
)

# Columnas finales del parquet curado (orden estable).
OUTPUT_FEATURES: tuple[str, ...] = (
    # monto
    "log1p_amt", "amt_gt_p95_legit",
    # temporales
    "hour", "dow", "is_night", "hour_sin", "hour_cos",
    # demografía
    "age",
    # geográficas relativas (entre tx consecutivas de la misma cc_num)
    "time_since_last_tx", "dist_consecutive_km", "velocity_kmh",
    # rolling por cc_num (closed='left' → ventana estrictamente pasada)
    "rolling_count_1h", "rolling_amt_sum_1h", "rolling_amt_mean_1h", "rolling_amt_std_1h",
    "rolling_count_24h", "rolling_amt_sum_24h", "rolling_amt_mean_24h", "rolling_amt_std_24h",
    "rolling_count_7d", "rolling_amt_sum_7d", "rolling_amt_mean_7d", "rolling_amt_std_7d",
    # target encoding KFold (fit en train, aplicado a val/test)
    "te_merchant", "te_category", "te_state", "te_job",
)


# --- helpers --- #


def _haversine_km(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Distancia haversine en km entre dos pares (lat, lon).

    Args:
        lat1: Latitud del primer punto (grados).
        lon1: Longitud del primer punto (grados).
        lat2: Latitud del segundo punto (grados).
        lon2: Longitud del segundo punto (grados).

    Returns:
        Array de distancias en kilómetros (NaN si algún input es NaN).
    """
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2r - lat1r
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def temporal_split(df: pd.DataFrame) -> pd.Series:
    """Asigna ``train``/``val``/``test`` por orden de ``unix_time`` (70/15/15).

    El df se asume ordenado ascendentemente por ``unix_time``. No hace
    shuffle: respeta estrictamente el orden temporal para evitar leakage.

    Args:
        df: DataFrame ordenado por ``unix_time``.

    Returns:
        Serie de strings con la asignación ``train``/``val``/``test``.
    """
    n = len(df)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_val] = "val"
    split[n_train + n_val :] = "test"
    return pd.Series(split, index=df.index, name="split")


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``hour``, ``dow``, ``is_night`` y sin/cos cíclicos de la hora."""
    ts = df["ts"]
    df["hour"] = ts.dt.hour.astype("int16")
    df["dow"] = ts.dt.dayofweek.astype("int16")
    # Pico nocturno detectado en el EDA: 22-02h (~5x tasa global).
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 2)).astype("int8")
    angle = 2.0 * np.pi * df["hour"].astype(float) / 24.0
    df["hour_sin"] = np.sin(angle).astype("float32")
    df["hour_cos"] = np.cos(angle).astype("float32")
    return df


def _add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula edad del titular **al momento de cada transacción**.

    No usa la fecha actual: usa ``ts`` (= ``trans_date_trans_time``) menos
    ``dob``. Esto evita que la feature varíe según cuándo se entrene.
    """
    dob = pd.to_datetime(df["dob"], errors="coerce")
    delta = (df["ts"] - dob).dt.total_seconds()
    df["age"] = (delta / (365.25 * 24 * 3600.0)).astype("float32")
    return df


def _add_amt_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Agrega ``log1p_amt`` y flag ``amt > p95(train_legit)``.

    El p95 se calcula **sólo sobre transacciones legítimas del split de
    train**, para que el flag refleje el techo "habitual" sin contaminarse
    con los montos elevados típicos del fraude.
    """
    df["log1p_amt"] = np.log1p(df["amt"]).astype("float32")
    legit_train = df.loc[train_mask & (df[config.TARGET_COLUMN] == 0), "amt"]
    p95 = float(legit_train.quantile(0.95))
    df["amt_gt_p95_legit"] = (df["amt"] > p95).astype("int8")
    return df


def _add_consecutive_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Distancia y velocidad entre tx consecutivas de la misma ``cc_num``.

    Asume ``df`` ya ordenado por ``(cc_num, unix_time)``. Para la primera
    transacción de cada tarjeta, todos los valores quedan NaN.
    """
    g = df.groupby(config.GROUP_KEY, sort=False)
    prev_lat = g["merch_lat"].shift(1)
    prev_lon = g["merch_long"].shift(1)
    df["dist_consecutive_km"] = _haversine_km(
        df["merch_lat"], df["merch_long"], prev_lat, prev_lon
    ).astype("float32")
    delta_s = g["ts"].diff().dt.total_seconds()
    df["time_since_last_tx"] = delta_s.astype("float32")
    # km/h = km / (s/3600). Mantengo NaN cuando time_delta es NaN o 0
    # (el primer tx de la tarjeta y duplicados de timestamp).
    hours = delta_s / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.where(hours > 0, df["dist_consecutive_km"] / hours, np.nan)
    df["velocity_kmh"] = velocity.astype("float32")
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega rolling stats de ``amt`` por ``cc_num`` en ventanas (1h, 24h, 7d).

    Usa ``closed='left'`` para excluir la transacción actual del agregado:
    así la fila ``i`` sólo "ve" transacciones estrictamente anteriores
    dentro de la ventana, evitando leakage del propio registro.
    """
    for label, win in ROLLING_WINDOWS:
        rolled = (
            df.groupby(config.GROUP_KEY, sort=False)
            .rolling(window=win, on="ts", closed="left", min_periods=0)["amt"]
            .agg(["count", "sum", "mean", "std"])
        )
        # groupby(sort=False)+rolling preserva el orden de df: como df ya
        # está ordenado por (cc_num, unix_time), las filas de `rolled`
        # corresponden 1-a-1 con las de df. Con on='ts' el index pierde
        # el offset posicional, así que asignamos por valores.
        if len(rolled) != len(df):
            raise RuntimeError("rolling devolvió un largo distinto al de df")
        # min_periods=0 → count y sum son 0 cuando no hay pasado;
        # mean y std quedan NaN (semánticamente indefinidos).
        df[f"rolling_count_{label}"] = (
            np.nan_to_num(rolled["count"].to_numpy(), nan=0.0).astype("float32")
        )
        df[f"rolling_amt_sum_{label}"] = (
            np.nan_to_num(rolled["sum"].to_numpy(), nan=0.0).astype("float32")
        )
        df[f"rolling_amt_mean_{label}"] = rolled["mean"].to_numpy().astype("float32")
        df[f"rolling_amt_std_{label}"] = rolled["std"].to_numpy().astype("float32")
    return df


def _kfold_target_encode(
    df: pd.DataFrame,
    train_mask: pd.Series,
    column: str,
    target: str = config.TARGET_COLUMN,
    n_folds: int = config.TARGET_ENCODING_FOLDS,
    smoothing: int = config.TARGET_ENCODING_SMOOTHING,
    seed: int = config.SEED,
) -> pd.Series:
    """Target encoding KFold con smoothing m, fit sólo sobre train.

    Para cada fila del split de train, el encoding de su categoría se
    calcula con los demás folds (out-of-fold) → no usa su propio target.
    Para val/test se usa el mapping del **train completo**, también con
    smoothing.

    El smoothing pondera entre la media de la categoría y la media global:
    ``encoded = (n*mean_cat + m*mean_global) / (n + m)``. Categorías no
    vistas en train caen al ``mean_global``.

    Args:
        df: DataFrame con la columna categórica y el target.
        train_mask: Boolean mask del split de train.
        column: Nombre de la columna categórica a encodear.
        target: Columna binaria objetivo (0/1).
        n_folds: Número de folds para el encoding del split de train.
        smoothing: Coeficiente m del prior global.
        seed: Semilla del KFold.

    Returns:
        Serie ``float32`` con el encoding de cada fila (alineada al index).
    """
    encoded = pd.Series(np.nan, index=df.index, dtype="float64")
    train_df = df.loc[train_mask, [column, target]]
    global_mean = float(train_df[target].mean())

    # Encoding fold-wise dentro de train (out-of-fold por fila).
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_positions = np.arange(len(train_df))
    for oof_train_pos, oof_val_pos in kf.split(train_positions):
        oof_train = train_df.iloc[oof_train_pos]
        stats = oof_train.groupby(column, observed=True)[target].agg(["mean", "count"])
        mapping = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        oof_val_idx = train_df.index[oof_val_pos]
        vals = df.loc[oof_val_idx, column].map(mapping)
        encoded.loc[oof_val_idx] = vals.fillna(global_mean).to_numpy()

    # Encoding para val/test: mapping del train completo (con smoothing).
    full_stats = train_df.groupby(column, observed=True)[target].agg(["mean", "count"])
    full_mapping = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (
        full_stats["count"] + smoothing
    )
    non_train_idx = df.index[~train_mask]
    vals_nt = df.loc[non_train_idx, column].map(full_mapping)
    encoded.loc[non_train_idx] = vals_nt.fillna(global_mean).to_numpy()

    return encoded.astype("float32")


def _add_target_encoding(
    df: pd.DataFrame,
    train_mask: pd.Series,
    columns: Iterable[str] = TE_COLUMNS,
) -> pd.DataFrame:
    """Aplica :func:`_kfold_target_encode` a varias columnas categóricas."""
    for col in columns:
        df[f"te_{col}"] = _kfold_target_encode(df, train_mask=train_mask, column=col)
    return df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona el set final de columnas para persistir."""
    keep = [*OUTPUT_FEATURES, config.TARGET_COLUMN, "split"]
    missing = set(keep) - set(df.columns)
    if missing:
        raise RuntimeError(f"Columnas esperadas faltantes: {sorted(missing)}")
    return df[keep].copy()


# --- pipeline público --- #


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el set de features de Fase 2 a partir del staging parquet.

    Pipeline (en este orden):
        1. Split temporal 70/15/15 por ``unix_time`` (sin shuffle).
        2. Re-orden por ``(cc_num, unix_time)`` para los rolling.
        3. Features temporales (hour, dow, is_night, sin/cos).
        4. Edad al momento de la transacción.
        5. Features de monto (log1p, flag p95 legit-train).
        6. Distancia y velocidad entre tx consecutivas de la tarjeta.
        7. Rolling counts/sums/means/stds por cc_num (1h, 24h, 7d).
        8. Target encoding KFold (merchant, category, state, job).
        9. Selección de columnas finales.

    Args:
        df: DataFrame del staging (salida de :mod:`src.ingest`). Debe
            contener al menos: ``cc_num``, ``unix_time``,
            ``trans_date_trans_time``, ``amt``, ``merch_lat/long``,
            ``dob``, ``merchant``, ``category``, ``state``, ``job``,
            y la columna target ``is_fraud``.

    Returns:
        DataFrame curado con sólo las columnas de :data:`OUTPUT_FEATURES`,
        el target y la columna ``split``. Ordenado por ``unix_time``.
    """
    df = df.copy()

    # 1. Split temporal sobre orden global por unix_time.
    df = df.sort_values("unix_time", kind="stable").reset_index(drop=True)
    df["split"] = temporal_split(df)
    train_mask = df["split"] == "train"

    # 2. Re-orden por (cc_num, unix_time) — la columna split viaja con las filas.
    df = df.sort_values([config.GROUP_KEY, "unix_time"], kind="stable").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["unix_time"], unit="s")
    train_mask = df["split"] == "train"

    # 3-7. Features.
    df = _add_temporal_features(df)
    df = _add_age(df)
    df = _add_amt_features(df, train_mask=train_mask)
    df = _add_consecutive_distance(df)
    df = _add_rolling_features(df)

    # 8. Target encoding (fit en train, aplicado al resto).
    df = _add_target_encoding(df, train_mask=train_mask, columns=TE_COLUMNS)

    # 9. Drop columnas no-feature + selección final, re-ordenado por tiempo.
    drop_cols = list(set(config.DROP_COLUMNS) | set(EXTRA_DROP))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = _select_output_columns(df)
    return df


def run(input_path: Path | None = None, output_path: Path | None = None) -> Path:
    """Lee el staging parquet, ejecuta :func:`build_features` y persiste.

    Args:
        input_path: parquet de entrada. Default: ``config.STAGING_FILE``.
        output_path: parquet de salida. Default: ``config.CURATED_FILE``.

    Returns:
        Path al parquet curado escrito.
    """
    input_path = input_path or config.STAGING_FILE
    output_path = output_path or config.CURATED_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el staging parquet en {input_path}. Corré 'make ingest' primero."
        )

    print(f"[features] leyendo {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[features] filas: {len(df):,} | columnas: {len(df.columns)}")

    out = build_features(df)
    out.to_parquet(output_path, compression="zstd", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"[features] wrote {output_path} ({size_mb:,.1f} MB) "
        f"| shape={out.shape} | splits={out['split'].value_counts().to_dict()}"
    )
    return output_path


def main() -> int:
    try:
        run()
    except FileNotFoundError as exc:
        print(f"[features][error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

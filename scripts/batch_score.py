"""CLI: parquet curado → parquet con scores via API.

Lee features de un parquet, llama ``/score_batch`` en chunks, y escribe un
parquet con las columnas originales + ``score_raw``, ``score_calibrated``,
``decision_operating``, ``decision_cost``.

Uso típico:

    python -m scripts.batch_score \\
        --input data/curated/transactions_features.parquet \\
        --output data/scored/test_scored.parquet \\
        --api-url http://localhost:8000 \\
        --batch-size 500 \\
        --filter-split test
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FEATURE_NAMES: tuple[str, ...] = (
    "log1p_amt", "amt_gt_p95_legit", "hour", "dow", "is_night", "hour_sin",
    "hour_cos", "age", "time_since_last_tx", "dist_consecutive_km",
    "velocity_kmh", "rolling_count_1h", "rolling_amt_sum_1h",
    "rolling_amt_mean_1h", "rolling_amt_std_1h", "rolling_count_24h",
    "rolling_amt_sum_24h", "rolling_amt_mean_24h", "rolling_amt_std_24h",
    "rolling_count_7d", "rolling_amt_sum_7d", "rolling_amt_mean_7d",
    "rolling_amt_std_7d", "te_merchant", "te_category", "te_state", "te_job",
)


_NULLABLE_FEATURES: frozenset[str] = frozenset({
    "time_since_last_tx", "dist_consecutive_km", "velocity_kmh",
    "rolling_count_1h", "rolling_amt_sum_1h", "rolling_amt_mean_1h",
    "rolling_amt_std_1h", "rolling_count_24h", "rolling_amt_sum_24h",
    "rolling_amt_mean_24h", "rolling_amt_std_24h", "rolling_count_7d",
    "rolling_amt_sum_7d", "rolling_amt_mean_7d", "rolling_amt_std_7d",
    "te_merchant", "te_category", "te_state", "te_job",
})


def _row_features(row: pd.Series) -> dict:
    """Convierte una fila a dict JSON-serializable (NaN → None)."""
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
                raise ValueError(f"NaN/inf en feature non-nullable {name!r}")
        else:
            out[name] = f
    return out


def _try_tqdm(it, total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm(it, total=total, desc=desc)
    except ImportError:
        return it


def score_dataframe(
    df: pd.DataFrame,
    api_url: str,
    batch_size: int,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Llama /score_batch en chunks, devuelve df original + columnas de score.

    Las filas con NaN en alguna de las 27 features se omiten (no son
    JSON-serializables); su contribución al score quedará como NaN en el
    output.
    """
    score_raw = np.full(len(df), np.nan)
    score_cal = np.full(len(df), np.nan)
    op_flag = np.full(len(df), False, dtype=bool)
    cost_flag = np.full(len(df), False, dtype=bool)
    op_thr = np.nan
    cost_thr = np.nan

    valid_idx = np.arange(len(df))
    n_batches = (len(valid_idx) + batch_size - 1) // batch_size

    for batch_pos in _try_tqdm(range(n_batches), total=n_batches, desc="scoring"):
        sl = valid_idx[batch_pos * batch_size : (batch_pos + 1) * batch_size]
        rows = [_row_features(df.iloc[int(i)]) for i in sl]
        resp = requests.post(
            f"{api_url.rstrip('/')}/score_batch",
            json={"features": rows},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/score_batch status {resp.status_code}: {resp.text[:300]}")
        results = resp.json()["results"]
        for offset, item in enumerate(results):
            i = int(sl[offset])
            score_raw[i] = item["score_raw"]
            score_cal[i] = item["score_calibrated"]
            op_flag[i] = item["decision_operating"]["is_fraud"]
            cost_flag[i] = item["decision_cost"]["is_fraud"]
            op_thr = item["decision_operating"]["threshold"]
            cost_thr = item["decision_cost"]["threshold"]

    out = df.copy()
    out["score_raw"] = score_raw
    out["score_calibrated"] = score_cal
    out["decision_operating"] = op_flag
    out["decision_cost"] = cost_flag
    out.attrs["op_threshold"] = float(op_thr) if not math.isnan(op_thr) else None
    out.attrs["cost_threshold"] = float(cost_thr) if not math.isnan(cost_thr) else None
    return out


def maybe_sanity_check(df: pd.DataFrame, expected_pr_auc: float | None = 0.8771) -> None:
    """Si hay ``is_fraud``, calcula PR-AUC sobre las filas scoreadas.

    Lo usamos como sanity check end-to-end del API: si el batch reproduce el
    PR-AUC reportado en Fase 4, entonces el modelo + serialización + I/O
    estuvieron bien.
    """
    if "is_fraud" not in df.columns:
        return
    mask = ~df["score_raw"].isna()
    if mask.sum() == 0:
        return
    y = df.loc[mask, "is_fraud"].astype(int).to_numpy()
    s = df.loc[mask, "score_raw"].to_numpy()
    if y.sum() == 0:
        print("[batch_score] (sanity) sin positivos, skip PR-AUC")
        return
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except ImportError:
        return
    pr = float(average_precision_score(y, s))
    roc = float(roc_auc_score(y, s))
    print(f"[batch_score] PR-AUC={pr:.4f} | ROC-AUC={roc:.4f} (n={mask.sum():,})")
    if expected_pr_auc is not None and abs(pr - expected_pr_auc) > 0.01:
        print(
            f"[batch_score][warn] PR-AUC {pr:.4f} no matchea Fase 4 ({expected_pr_auc:.4f})",
            file=sys.stderr,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="parquet con las 27 features")
    p.add_argument("--output", required=True, type=Path, help="parquet de salida")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument(
        "--filter-split",
        choices=["train", "val", "test"],
        default=None,
        help="si el parquet tiene columna 'split', filtrar antes de scorear",
    )
    p.add_argument("--limit", type=int, default=None, help="solo procesar las primeras N filas (debug)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"[batch_score][error] no existe {args.input}", file=sys.stderr)
        return 2
    df = pd.read_parquet(args.input)
    if args.filter_split and "split" in df.columns:
        df = df.loc[df["split"] == args.filter_split].drop(columns=["split"]).reset_index(drop=True)
        print(f"[batch_score] filtrado split={args.filter_split} → {len(df):,} filas")
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[batch_score] --limit {args.limit} aplicado")

    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        print(f"[batch_score][error] faltan features: {sorted(missing)}", file=sys.stderr)
        return 2

    out = score_dataframe(df, args.api_url, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[batch_score] escritos {len(out):,} rows → {args.output}")
    maybe_sanity_check(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

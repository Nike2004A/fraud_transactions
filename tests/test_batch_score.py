"""Smoke test del script batch_score con monkeypatch sobre requests.

No depende del API levantado: simula la respuesta /score_batch para
chequear que el script:
- arma los chunks correctamente,
- preserva el orden,
- escribe el parquet con las columnas esperadas,
- maneja NaN en features (envía None, no rompe).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import batch_score


def _fake_curated(n: int = 100, n_pos: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(size=n).astype("float32") for name in batch_score.FEATURE_NAMES})
    # Algunos enteros lógicos
    for col in ("amt_gt_p95_legit", "is_night"):
        df[col] = rng.integers(0, 2, size=n).astype("int8")
    for col in ("hour", "dow"):
        df[col] = rng.integers(0, 24, size=n).astype("int16")
    df["is_fraud"] = 0
    df.loc[: n_pos - 1, "is_fraud"] = 1
    # Inyectar algunos NaN en las features nullables para verificar que se serializan como None.
    df.loc[3, "rolling_amt_std_1h"] = np.nan
    df.loc[7, "te_merchant"] = np.nan
    df["split"] = "test"
    return df


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload


def _make_post_handler():
    """Genera una respuesta determinística por fila a partir del request real."""
    seen_chunks: list[int] = []

    def fake_post(url, json, timeout):  # noqa: A002 — keyword `json`, lo definimos así para matchear requests.post
        rows = json["features"]
        seen_chunks.append(len(rows))
        results = []
        for r in rows:
            # Verificar que NaN viene como None (no como string ni excepción).
            for k, v in r.items():
                assert v is None or isinstance(v, (int, float)), f"valor inválido en {k}: {v!r}"
            # Score arbitrario reproducible: log1p_amt + offset.
            raw = float((r["log1p_amt"] + 5.0) / 10.0)
            results.append(
                {
                    "score_raw": raw,
                    "score_calibrated": raw / 2,
                    "decision_operating": {"threshold": 0.6642, "is_fraud": raw >= 0.6642},
                    "decision_cost": {"threshold": 0.52, "is_fraud": raw >= 0.52},
                }
            )
        return _FakeResponse({"results": results})

    return fake_post, seen_chunks


def test_batch_score_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _fake_curated(n=100)
    inp = tmp_path / "curated.parquet"
    out = tmp_path / "scored.parquet"
    df.to_parquet(inp, index=False)

    fake_post, chunks = _make_post_handler()
    monkeypatch.setattr(batch_score.requests, "post", fake_post)

    rc = batch_score.main([
        "--input", str(inp),
        "--output", str(out),
        "--api-url", "http://fake:9999",
        "--batch-size", "30",
        "--filter-split", "test",
    ])
    assert rc == 0
    assert out.exists()

    res = pd.read_parquet(out)
    assert len(res) == 100
    expected_cols = {"score_raw", "score_calibrated", "decision_operating", "decision_cost"}
    assert expected_cols <= set(res.columns)
    # 100 filas con batch=30 → chunks 30,30,30,10
    assert chunks == [30, 30, 30, 10]
    # Sin NaN en score_raw: todas las filas se scorearon (NaN se mandó como None).
    assert res["score_raw"].isna().sum() == 0
    # Orden preservado: la primera fila tiene el log1p_amt original.
    assert abs(res.iloc[0]["score_raw"] - (df.iloc[0]["log1p_amt"] + 5.0) / 10.0) < 1e-5


def test_batch_score_rejects_missing_input(tmp_path: Path) -> None:
    rc = batch_score.main([
        "--input", str(tmp_path / "noexiste.parquet"),
        "--output", str(tmp_path / "out.parquet"),
    ])
    assert rc == 2

"""Tests del API con FastAPI TestClient.

Las fixtures (``_fixtures.json``) se generaron offline con el modelo y
calibrador reales — ver el bloque al final de ``conftest.py`` /
``tests/_fixtures.json`` para regenerarlas si cambia el modelo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.settings import COST_THRESHOLD, FEATURE_NAMES, OPERATING_THRESHOLD


_FIXTURES = Path(__file__).parent / "_fixtures.json"


@pytest.fixture(scope="module")
def fixtures() -> dict:
    if not _FIXTURES.exists():
        pytest.skip(f"Falta {_FIXTURES} — regenerar con tests/regen_fixtures.py")
    with _FIXTURES.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Context manager dispara el lifespan (startup) → carga real del modelo.
    with TestClient(app) as c:
        yield c


# --- /healthz --- #


def test_healthz_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 27
    assert body["calibrator_kind"] in ("isotonic", "platt")


# --- /score happy path --- #


def test_score_known_fraud(client: TestClient, fixtures: dict) -> None:
    feats = fixtures["fraud_high_score"]
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected_raw = fixtures["expected"]["fraud_raw"]
    assert abs(body["score_raw"] - expected_raw) < 1e-4
    assert body["score_raw"] > OPERATING_THRESHOLD
    assert body["decision_operating"]["threshold"] == OPERATING_THRESHOLD
    assert body["decision_operating"]["is_fraud"] is True
    assert body["decision_cost"]["threshold"] == COST_THRESHOLD
    assert body["decision_cost"]["is_fraud"] is True


def test_score_known_legit(client: TestClient, fixtures: dict) -> None:
    feats = fixtures["legit_low_score"]
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 200
    body = resp.json()
    assert body["score_raw"] < COST_THRESHOLD
    assert body["decision_operating"]["is_fraud"] is False
    assert body["decision_cost"]["is_fraud"] is False


def test_score_shap_returns_5(client: TestClient, fixtures: dict) -> None:
    resp = client.post("/score", json={"features": fixtures["fraud_high_score"]})
    body = resp.json()
    assert len(body["shap_top5"]) == 5
    feature_set = set(FEATURE_NAMES)
    for entry in body["shap_top5"]:
        assert set(entry.keys()) == {"feature", "value", "contribution", "direction"}
        assert entry["feature"] in feature_set
        assert entry["direction"] in ("increases_fraud", "decreases_fraud")
    # |contrib| ordenado desc
    abs_contribs = [abs(e["contribution"]) for e in body["shap_top5"]]
    assert abs_contribs == sorted(abs_contribs, reverse=True)
    assert isinstance(body["base_value"], float)


# --- /score validation errors --- #


def test_score_missing_feature(client: TestClient, fixtures: dict) -> None:
    feats = dict(fixtures["fraud_high_score"])
    feats.pop("log1p_amt")
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any("log1p_amt" in str(item) for item in detail)


def test_score_extra_feature(client: TestClient, fixtures: dict) -> None:
    feats = dict(fixtures["fraud_high_score"])
    feats["extra_field"] = 1.0
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 422


# --- /score_batch --- #


def test_score_batch_returns_n(client: TestClient, fixtures: dict) -> None:
    rows = [fixtures["fraud_high_score"], fixtures["legit_low_score"]] * 3  # 6 filas
    resp = client.post("/score_batch", json={"features": rows[:5]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 5
    for item in body["results"]:
        assert set(item.keys()) == {
            "score_raw",
            "score_calibrated",
            "decision_operating",
            "decision_cost",
        }
    # Orden preservado: la primera fila es fraude, la segunda es legit.
    assert body["results"][0]["decision_operating"]["is_fraud"] is True
    assert body["results"][1]["decision_operating"]["is_fraud"] is False


def test_score_batch_too_large(client: TestClient, fixtures: dict) -> None:
    rows = [fixtures["legit_low_score"]] * 1001
    resp = client.post("/score_batch", json={"features": rows})
    # pydantic max_length=1000 → 422 antes de llegar al handler.
    assert resp.status_code in (413, 422)


def test_score_batch_empty(client: TestClient, fixtures: dict) -> None:
    resp = client.post("/score_batch", json={"features": []})
    assert resp.status_code == 422


def test_score_accepts_null_for_nullable(client: TestClient, fixtures: dict) -> None:
    """JSON null (NaN sentinel) en features rolling/std debe ser aceptado.

    XGBoost trata el NaN como missing-value durante la inferencia. Sin esto
    el batch_score sería inutilizable para el ~98 % del test set
    (cualquier primera tx en la ventana tiene std indefinido).
    """
    feats = dict(fixtures["fraud_high_score"])
    feats["rolling_amt_std_1h"] = None
    feats["te_merchant"] = None
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 200, resp.text


def test_score_rejects_null_for_non_nullable(client: TestClient, fixtures: dict) -> None:
    """``hour`` y otros enteros lógicos vienen del timestamp; nunca null."""
    feats = dict(fixtures["fraud_high_score"])
    feats["hour"] = None
    resp = client.post("/score", json={"features": feats})
    assert resp.status_code == 422

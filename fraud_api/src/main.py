"""FastAPI app: /healthz, /score, /score_batch.

Carga modelo + calibrador + TreeExplainer una sola vez en el lifespan.
Si algo falla al iniciar, el endpoint /healthz devuelve 503 y los
endpoints de inferencia 503 también — el container no debe servir
predicciones inconsistentes.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .inference import (
    ModelBundle,
    features_to_dataframe,
    load_bundle,
    predict_batch,
    predict_one,
)
from .schema import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from .settings import FEATURE_NAMES, MAX_BATCH_SIZE

log = logging.getLogger("fraud_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = load_bundle()
        app.state.bundle = bundle
        app.state.bundle_error = None
        log.info(
            "Modelo cargado: calibrator=%s, n_features=%d, base_value=%.4f",
            bundle.calibrator_kind,
            len(FEATURE_NAMES),
            bundle.base_value,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Falló la carga del modelo en startup: %s", exc)
        app.state.bundle = None
        app.state.bundle_error = str(exc)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Inferencia para el modelo XGBoost de Fase 3-4 del capstone.",
    version="1.0.0",
    lifespan=lifespan,
)


def _bundle(request: Request) -> ModelBundle:
    bundle: ModelBundle | None = request.app.state.bundle
    if bundle is None:
        err = request.app.state.bundle_error or "modelo no inicializado"
        raise HTTPException(status_code=503, detail=f"modelo no disponible: {err}")
    return bundle


@app.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> JSONResponse:
    bundle: ModelBundle | None = request.app.state.bundle
    if bundle is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "model_loaded": False,
                "calibrator_kind": "",
                "n_features": len(FEATURE_NAMES),
            },
        )
    payload = HealthResponse(
        status="ok",
        model_loaded=True,
        calibrator_kind=bundle.calibrator_kind,
        n_features=len(FEATURE_NAMES),
    )
    return JSONResponse(status_code=200, content=payload.model_dump())


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    bundle = _bundle(request)
    out = predict_one(bundle, req.features)
    return ScoreResponse(**out)


@app.post("/score_batch", response_model=BatchResponse)
def score_batch(req: BatchRequest, request: Request) -> BatchResponse:
    if len(req.features) > MAX_BATCH_SIZE:
        # pydantic ya rechaza esto a nivel schema, pero defendemos por si
        # alguien sube MAX_BATCH_SIZE sin tocar el schema.
        raise HTTPException(
            status_code=413,
            detail=f"batch supera el máximo de {MAX_BATCH_SIZE} filas",
        )
    bundle = _bundle(request)
    X = features_to_dataframe(req.features)
    items = predict_batch(bundle, X)
    return BatchResponse(results=items)

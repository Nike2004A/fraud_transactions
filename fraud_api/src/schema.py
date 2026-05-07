"""Schemas Pydantic v2 para request/response del API.

Reglas:
- ``FeaturesIn`` exige los 27 campos exactos. ``extra='forbid'`` rechaza
  campos desconocidos con 422 antes de llegar al modelo.
- ``None`` es el sentinel JSON para NaN (JSON no tiene NaN nativo). El
  modelo XGBoost maneja NaN como missing-value durante la inferencia, así
  que features como ``rolling_amt_std_1h`` pueden faltar legítimamente
  (primera tx de la cc en la ventana de 1h ⇒ std indefinido). Los enteros
  lógicos (``hour``, ``dow``, ``is_night``, ``amt_gt_p95_legit``) NO admiten
  None: provienen del timestamp y siempre existen.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, conlist


class FeaturesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log1p_amt: float
    amt_gt_p95_legit: int
    hour: int
    dow: int
    is_night: int
    hour_sin: float
    hour_cos: float
    age: float
    time_since_last_tx: Optional[float]
    dist_consecutive_km: Optional[float]
    velocity_kmh: Optional[float]
    rolling_count_1h: Optional[float]
    rolling_amt_sum_1h: Optional[float]
    rolling_amt_mean_1h: Optional[float]
    rolling_amt_std_1h: Optional[float]
    rolling_count_24h: Optional[float]
    rolling_amt_sum_24h: Optional[float]
    rolling_amt_mean_24h: Optional[float]
    rolling_amt_std_24h: Optional[float]
    rolling_count_7d: Optional[float]
    rolling_amt_sum_7d: Optional[float]
    rolling_amt_mean_7d: Optional[float]
    rolling_amt_std_7d: Optional[float]
    te_merchant: Optional[float]
    te_category: Optional[float]
    te_state: Optional[float]
    te_job: Optional[float]


class ScoreRequest(BaseModel):
    features: FeaturesIn


class BatchRequest(BaseModel):
    features: conlist(FeaturesIn, min_length=1, max_length=1000)


class Decision(BaseModel):
    threshold: float
    is_fraud: bool


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str  # "increases_fraud" | "decreases_fraud"


class ScoreResponse(BaseModel):
    score_raw: float
    score_calibrated: float
    decision_operating: Decision
    decision_cost: Decision
    shap_top5: List[ShapContribution]
    base_value: float


class BatchScoreItem(BaseModel):
    score_raw: float
    score_calibrated: float
    decision_operating: Decision
    decision_cost: Decision


class BatchResponse(BaseModel):
    results: List[BatchScoreItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibrator_kind: str
    n_features: int

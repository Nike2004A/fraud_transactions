"""Cliente HTTP del API. Solo conoce los endpoints; no sabe nada del modelo."""
from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_HEALTH_TIMEOUT_S = 2.0


class ApiError(RuntimeError):
    """Error de red o HTTP del API."""


def health(api_url: str, timeout: float = DEFAULT_HEALTH_TIMEOUT_S) -> dict[str, Any]:
    """GET /healthz. Devuelve el body parseado o lanza ApiError."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/healthz", timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"no se pudo conectar a {api_url}: {exc}") from exc
    if resp.status_code != 200:
        raise ApiError(f"healthz status {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def score_one(
    api_url: str,
    features: dict[str, float],
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = 2,
) -> dict[str, Any]:
    """POST /score con retry exponencial. Re-lanza ApiError en fallo."""
    url = f"{api_url.rstrip('/')}/score"
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retries:
        try:
            resp = requests.post(url, json={"features": features}, timeout=timeout)
            if resp.status_code == 422:
                # No reintentar errores de validación: la falla es del input.
                raise ApiError(f"422 validation: {resp.text[:300]}")
            if resp.status_code >= 500:
                raise requests.RequestException(f"status {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(0.3 * 2 ** (attempt - 1))
    raise ApiError(f"score_one falló tras {max_retries + 1} intentos: {last_exc}")


def score_batch(
    api_url: str,
    rows: list[dict[str, float]],
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """POST /score_batch. Devuelve la lista ``results``."""
    url = f"{api_url.rstrip('/')}/score_batch"
    resp = requests.post(url, json={"features": rows}, timeout=timeout)
    if resp.status_code != 200:
        raise ApiError(f"score_batch status {resp.status_code}: {resp.text[:200]}")
    return resp.json()["results"]

# Imagen CPU-only para portabilidad del API. xgboost/shap/sklearn son los
# paquetes pesados; sin las wheels de CUDA la imagen final queda < 1.5 GB.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 es necesario para xgboost (OpenMP). El resto se instala vía pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Build context: la RAÍZ del repo (`docker build -f fraud_api/Dockerfile .`).
# Esto es necesario para poder copiar también `models/` que vive afuera de
# `fraud_api/`. El target `api-docker-build` del Makefile lo configura así.

COPY fraud_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud_api/src/ /app/src/

COPY models/xgb_best.json /app/models/xgb_best.json
COPY models/calibrator.pkl /app/models/calibrator.pkl

ENV FRAUD_API_MODELS_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

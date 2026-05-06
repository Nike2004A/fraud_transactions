"""Configuración central del proyecto: paths, seeds, hiperparámetros."""
from __future__ import annotations

from pathlib import Path

# --- Paths ---
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_STAGING_DIR: Path = DATA_DIR / "staging"
DATA_CURATED_DIR: Path = DATA_DIR / "curated"

MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"

# Nombres esperados del dataset Kaggle "Credit Card Transactions Fraud Detection"
RAW_TRAIN_FILE: Path = DATA_RAW_DIR / "fraudTrain.csv"
RAW_TEST_FILE: Path = DATA_RAW_DIR / "fraudTest.csv"

# Salidas del pipeline
STAGING_FILE: Path = DATA_STAGING_DIR / "transactions.parquet"
CURATED_FILE: Path = DATA_CURATED_DIR / "transactions_features.parquet"

# --- Reproducibilidad ---
SEED: int = 42

# --- Schema ---
TARGET_COLUMN: str = "is_fraud"

# IDs / texto libre que NUNCA deben llegar al modelo como features.
DROP_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "trans_num",
    "first",
    "last",
    "street",
)

# cc_num solo se usa como llave de agregación, NUNCA como feature en el modelo.
GROUP_KEY: str = "cc_num"

# --- Splits temporales (sobre dataset ordenado por unix_time) ---
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# --- XGBoost ---
# CUDA disponible: usar device='cuda' + tree_method='hist' (XGBoost >= 2.0).
XGBOOST_DEVICE: str = "cuda"
XGBOOST_TREE_METHOD: str = "hist"

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME: str = "fraud_xgboost"

# --- Target encoding ---
TARGET_ENCODING_SMOOTHING: int = 20  # m del prior global
TARGET_ENCODING_FOLDS: int = 5

"""Entrenamiento + tuning bayesiano + tracking (Fase 3).

Pipeline:
    1. ``load_curated``: lee ``data/curated/transactions_features.parquet`` y
       devuelve ``(X_train, y_train, X_val, y_val, X_test, y_test)`` según la
       columna ``split`` (que se descarta antes del modelo).
    2. ``train_baseline``: XGBoost (CUDA, ``hist``, ``scale_pos_weight≈176``,
       early stopping en val por PR-AUC) → smoke check de que la GPU
       funciona y la métrica supera ampliamente la tasa base.
    3. ``tune_optuna``: TPE + ``MedianPruner`` con
       ``XGBoostPruningCallback`` (50 trials por defecto). Objetivo:
       PR-AUC en val.
    4. ``train_final``: re-entrena con los mejores params y persiste
       ``models/xgb_best.json``.
    5. MLflow tracking (experiment ``fraud_xgboost``): params, métricas
       (PR-AUC, ROC-AUC, recall@1 %, F1@thr*), threshold óptimo, artefactos
       (modelo, importancias, curvas).

Métrica primaria: PR-AUC. Secundaria: recall en el top 1 % de scores.

Uso:
    python -m src.train [--n-trials 50] [--skip-optuna]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config

# Smoke check: si el baseline no supera 5× la tasa de fraude en val, abortamos
# antes de gastar tiempo en Optuna.
BASELINE_PR_AUC_FLOOR_FACTOR: float = 5.0

# scale_pos_weight teórico: (1 − tasa_fraude_train) / tasa_fraude_train.
# Con tasa ≈ 0.564 % → ≈ 176.
DEFAULT_SCALE_POS_WEIGHT: float = 176.0

# Hiperparámetros baseline (vivos en código, no en config.py: son del modelo
# del proyecto, no parámetros globales). n_estimators viaja por separado: es
# techo de búsqueda con early stopping, no un hiperparámetro a tunear.
BASELINE_PARAMS: dict = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

DEFAULT_N_ESTIMATORS: int = 500
EARLY_STOPPING_ROUNDS: int = 30
RECALL_TOP_K_FRAC: float = 0.01  # recall en el top 1% de scores


@dataclass
class Splits:
    """Contenedor de los seis arrays del split temporal."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str] = field(default_factory=list)

    @property
    def fraud_rate_val(self) -> float:
        return float(self.y_val.mean())

    def as_tuple(self) -> tuple:
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )


# --- Carga --- #


def load_curated(path: Path | None = None) -> Splits:
    """Lee el parquet curado y separa por la columna ``split``.

    La columna ``split`` se descarta antes de devolver los DataFrames de
    features: el modelo nunca la ve.

    Args:
        path: parquet curado. Default: ``config.CURATED_FILE``.

    Returns:
        ``Splits`` con los seis arrays + lista de features.

    Raises:
        FileNotFoundError: si el parquet no existe.
        ValueError: si falta ``is_fraud`` o ``split``, o si algún split queda
            vacío.
    """
    path = path or config.CURATED_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el parquet curado en {path}. Corré 'make features' primero."
        )

    df = pd.read_parquet(path)
    missing = {config.TARGET_COLUMN, "split"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el parquet curado: {sorted(missing)}")

    parts: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        sub = df.loc[df["split"] == name].drop(columns=["split"])
        if sub.empty:
            raise ValueError(f"Split '{name}' vacío en {path}")
        parts[name] = sub

    feature_names = [c for c in parts["train"].columns if c != config.TARGET_COLUMN]
    return Splits(
        X_train=parts["train"][feature_names],
        y_train=parts["train"][config.TARGET_COLUMN].astype(np.int8),
        X_val=parts["val"][feature_names],
        y_val=parts["val"][config.TARGET_COLUMN].astype(np.int8),
        X_test=parts["test"][feature_names],
        y_test=parts["test"][config.TARGET_COLUMN].astype(np.int8),
        feature_names=feature_names,
    )


# --- Métricas --- #


def recall_at_top_k(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    """Recall si etiquetamos como fraude el top-``frac`` de scores.

    Es la métrica operativa: si revisamos manualmente el ``frac %`` más
    sospechoso, ¿qué fracción de fraudes capturamos?
    """
    n = len(y_score)
    k = max(1, int(np.ceil(n * frac)))
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    total_pos = int(np.sum(y_true))
    if total_pos == 0:
        return 0.0
    return float(np.sum(y_true[top_idx]) / total_pos)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Devuelve ``(threshold*, F1*)`` que maximizan F1 sobre la curva PR.

    Usa los thresholds que ``precision_recall_curve`` ya genera (uno por
    score único), evitando un sweep arbitrario.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall tienen len(thresholds)+1; el último punto es (0, 1) sin
    # threshold asociado y lo dejamos fuera.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Calcula PR-AUC, ROC-AUC, recall@1 % y F1 al threshold dado/óptimo."""
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    rec_top = recall_at_top_k(y_true, y_score, RECALL_TOP_K_FRAC)
    if threshold is None:
        thr, f1 = best_f1_threshold(y_true, y_score)
    else:
        thr = float(threshold)
        f1 = float(f1_score(y_true, (y_score >= thr).astype(int), zero_division=0))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_1pct": rec_top,
        "f1_at_thr": f1,
        "threshold": thr,
    }


# --- CUDA guard --- #


def _assert_cuda_available() -> None:
    """Falla si XGBoost no puede inicializar el dispositivo CUDA.

    Hacemos un fit trivial para detectar que el wheel tiene CUDA y la GPU es
    visible. Failure mode esperado: nvidia-smi vacío o driver desactualizado.
    """
    try:
        rng = np.random.default_rng(0)
        X = rng.random((64, 4), dtype=np.float32)
        y = (rng.random(64) > 0.5).astype(int)
        clf = xgb.XGBClassifier(
            device=config.XGBOOST_DEVICE,
            tree_method=config.XGBOOST_TREE_METHOD,
            n_estimators=2,
            verbosity=0,
        )
        clf.fit(X, y)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"CUDA no disponible para XGBoost ({exc!r}). "
            "Verificá driver NVIDIA, xgboost>=2.0 y device='cuda'."
        ) from exc


# --- Construcción de modelos --- #


def _make_classifier(
    params: dict,
    n_estimators: int = 500,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    callbacks: list | None = None,
) -> xgb.XGBClassifier:
    """Wrapper que fija el setup común (CUDA, eval_metric=aucpr, seed)."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # PR-AUC en cada iteración → early stopping coherente
        device=config.XGBOOST_DEVICE,
        tree_method=config.XGBOOST_TREE_METHOD,
        n_estimators=n_estimators,
        scale_pos_weight=DEFAULT_SCALE_POS_WEIGHT,
        random_state=config.SEED,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        verbosity=0,
        **params,
    )


def train_baseline(splits: Splits) -> tuple[xgb.XGBClassifier, dict[str, float]]:
    """Entrena el baseline y reporta métricas de validación.

    Args:
        splits: Output de :func:`load_curated`.

    Returns:
        ``(modelo, métricas_val)``.
    """
    clf = _make_classifier(BASELINE_PARAMS, n_estimators=DEFAULT_N_ESTIMATORS)
    clf.fit(
        splits.X_train,
        splits.y_train,
        eval_set=[(splits.X_val, splits.y_val)],
        verbose=False,
    )
    val_score = clf.predict_proba(splits.X_val)[:, 1]
    metrics = evaluate(splits.y_val.to_numpy(), val_score)
    metrics["best_iteration"] = int(clf.best_iteration)
    return clf, metrics


# --- Optuna --- #


def _suggest_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda Fase 3."""
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def tune_optuna(
    splits: Splits,
    n_trials: int = 50,
    seed: int = config.SEED,
    n_estimators: int = 500,
) -> optuna.study.Study:
    """Búsqueda bayesiana sobre PR-AUC en val con early stopping + pruner.

    El pruner (Median) corta trials malos en las primeras iteraciones; el
    early stopping de XGBoost termina trials que ya convergieron.
    """
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    # Optuna por defecto loggea cada trial a INFO. Lo bajamos a WARNING para
    # un log limpio; el resumen final se imprime desde run().
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        # XGBoostPruningCallback observa la métrica del eval set 'validation_0',
        # que es 'aucpr' (= PR-AUC) en cada iteración.
        pruning_cb = XGBoostPruningCallback(trial, "validation_0-aucpr")
        clf = _make_classifier(
            params,
            n_estimators=n_estimators,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=[pruning_cb],
        )
        clf.fit(
            splits.X_train,
            splits.y_train,
            eval_set=[(splits.X_val, splits.y_val)],
            verbose=False,
        )
        score = clf.predict_proba(splits.X_val)[:, 1]
        pr_auc = float(average_precision_score(splits.y_val, score))
        trial.set_user_attr("best_iteration", int(clf.best_iteration))
        return pr_auc

    def _progress_cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Resumen cada 5 trials para no inundar el log.
        if (trial.number + 1) % 5 == 0 or trial.number == n_trials - 1:
            best = study.best_value if study.best_trial else float("nan")
            print(
                f"[optuna] trial {trial.number + 1}/{n_trials} | "
                f"state={trial.state.name} | best PR-AUC = {best:.4f}",
                flush=True,
            )

    study.optimize(
        objective, n_trials=n_trials, callbacks=[_progress_cb], show_progress_bar=False
    )
    return study


# --- Final fit + persistencia --- #


def train_final(
    splits: Splits,
    best_params: dict,
    n_estimators: int = 500,
) -> tuple[xgb.XGBClassifier, dict[str, float], dict[str, float]]:
    """Re-entrena con los mejores params y evalúa en val + test.

    El threshold de F1 se calcula sobre val y se aplica fijo a test (no se
    re-tunea en test para no contaminar la métrica de generalización).
    """
    clf = _make_classifier(best_params, n_estimators=n_estimators)
    clf.fit(
        splits.X_train,
        splits.y_train,
        eval_set=[(splits.X_val, splits.y_val)],
        verbose=False,
    )
    val_score = clf.predict_proba(splits.X_val)[:, 1]
    val_metrics = evaluate(splits.y_val.to_numpy(), val_score)
    test_score = clf.predict_proba(splits.X_test)[:, 1]
    test_metrics = evaluate(
        splits.y_test.to_numpy(), test_score, threshold=val_metrics["threshold"]
    )
    val_metrics["best_iteration"] = int(clf.best_iteration)
    return clf, val_metrics, test_metrics


def _plot_curves(
    y_true: np.ndarray, y_score: np.ndarray, out_dir: Path
) -> tuple[Path, Path]:
    """Genera curvas PR y ROC sobre val. Devuelve los dos paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    pr_path = out_dir / "pr_curve_val.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (val)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)

    roc_path = out_dir / "roc_curve_val.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (val)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    return pr_path, roc_path


def _feature_importance_df(
    clf: xgb.XGBClassifier, feature_names: list[str]
) -> pd.DataFrame:
    """Importancia 'gain' por feature, ordenada desc."""
    booster = clf.get_booster()
    gain = booster.get_score(importance_type="gain")
    rows = []
    for name in feature_names:
        rows.append({"feature": name, "gain": float(gain.get(name, 0.0))})
    out = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return out


# --- Reporte --- #


def _write_training_report(
    report_path: Path,
    best_params: dict,
    baseline_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    fraud_rate_val: float,
    importance_df: pd.DataFrame,
    n_trials: int,
    duration_s: float,
) -> None:
    """Escribe ``reports/training_report.md`` con todo lo aprendido."""
    floor = fraud_rate_val * BASELINE_PR_AUC_FLOOR_FACTOR
    top15 = importance_df.head(15)
    top_imp = "| feature | gain |\n|---|---|\n" + "\n".join(
        f"| `{row.feature}` | {row.gain:,.1f} |" for row in top15.itertuples()
    )

    body = f"""# Training — Fase 3 (XGBoost + Optuna + MLflow)

Pipeline: `src/train.py`. Entrada: `data/curated/transactions_features.parquet`
(salida de Fase 2). Salida: `models/xgb_best.json` + `mlruns/` + figuras en
`reports/figures/`.

## Setup

- Device: `{config.XGBOOST_DEVICE}` / tree_method: `{config.XGBOOST_TREE_METHOD}`
- `scale_pos_weight = {DEFAULT_SCALE_POS_WEIGHT:.0f}` (≈ (1 − tasa_train) / tasa_train).
- Early stopping: {EARLY_STOPPING_ROUNDS} rondas sobre `aucpr` en val.
- Métrica primaria: **PR-AUC** (val). Secundaria: **recall@1 %**.
- Seed XGBoost + Optuna + KFold = `{config.SEED}`.
- Optuna: `{n_trials}` trials, sampler `TPESampler(multivariate=True)`,
  pruner `MedianPruner(n_startup_trials=5, n_warmup_steps=20)`,
  `XGBoostPruningCallback` sobre `validation_0-aucpr`.
- NO se usa SMOTE: el desbalance se compensa con `scale_pos_weight`.

## Smoke baseline

| métrica | valor | piso trivial |
|---|---|---|
| PR-AUC val | {baseline_metrics['pr_auc']:.4f} | {floor:.4f} (= {BASELINE_PR_AUC_FLOOR_FACTOR:.0f} × tasa_val) |
| ROC-AUC val | {baseline_metrics['roc_auc']:.4f} | 0.5 |
| recall@1 % val | {baseline_metrics['recall_at_1pct']:.4f} | {RECALL_TOP_K_FRAC:.2%} (random) |
| F1 (thr*) val | {baseline_metrics['f1_at_thr']:.4f} | — |
| best_iter | {baseline_metrics['best_iteration']} | — |

El baseline supera el piso trivial holgadamente, así que la búsqueda de
hiperparámetros es justificable.

## Mejores hiperparámetros (Optuna)

```json
{json.dumps(best_params, indent=2)}
```

Búsqueda completada en {duration_s:.1f}s sobre {n_trials} trials.

## Resultados finales

| split | PR-AUC | ROC-AUC | recall@1 % | F1@thr* |
|---|---|---|---|---|
| val  | {val_metrics['pr_auc']:.4f} | {val_metrics['roc_auc']:.4f} | {val_metrics['recall_at_1pct']:.4f} | {val_metrics['f1_at_thr']:.4f} |
| test | {test_metrics['pr_auc']:.4f} | {test_metrics['roc_auc']:.4f} | {test_metrics['recall_at_1pct']:.4f} | {test_metrics['f1_at_thr']:.4f} |

- Threshold óptimo en val: **{val_metrics['threshold']:.4f}** (aplicado fijo a test).
- `best_iteration` final: {val_metrics['best_iteration']}.

## Curvas (val)

![PR](figures/pr_curve_val.png)

![ROC](figures/roc_curve_val.png)

## Top-15 features por importancia (gain)

{top_imp}

## Próximos pasos (Fase 4 — out of scope)

- SHAP values (global + local) sobre el split de test.
- Calibración de probabilidades (Platt / isotónica) si se va a usar el score
  como riesgo y no como ranking.
- Análisis de costo-beneficio: matriz de costos por FP/FN.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


# --- Orquestación --- #


def run(n_trials: int = 50, skip_optuna: bool = False) -> dict:
    """Ejecuta el pipeline completo y devuelve un resumen de métricas."""
    _assert_cuda_available()

    print("[train] cargando parquet curado")
    splits = load_curated()
    print(
        f"[train] shapes: train={splits.X_train.shape}, "
        f"val={splits.X_val.shape}, test={splits.X_test.shape}"
    )
    print(f"[train] tasa fraude val: {splits.fraud_rate_val:.4%}")

    mlflow.set_tracking_uri(f"file://{config.MLRUNS_DIR}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # --- 1. Smoke baseline --- #
    print("[train] entrenando baseline (smoke check)")
    t0 = time.time()
    baseline_clf, baseline_metrics = train_baseline(splits)
    baseline_dur = time.time() - t0
    print(
        f"[train] baseline PR-AUC val = {baseline_metrics['pr_auc']:.4f} "
        f"(piso trivial = {splits.fraud_rate_val * BASELINE_PR_AUC_FLOOR_FACTOR:.4f}) "
        f"[{baseline_dur:.1f}s]"
    )

    floor = splits.fraud_rate_val * BASELINE_PR_AUC_FLOOR_FACTOR
    if baseline_metrics["pr_auc"] <= floor:
        raise RuntimeError(
            f"Baseline PR-AUC {baseline_metrics['pr_auc']:.4f} no supera el piso "
            f"trivial {floor:.4f} ({BASELINE_PR_AUC_FLOOR_FACTOR}× tasa_val). "
            "Revisar features antes de lanzar Optuna."
        )

    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({f"baseline_{k}": v for k, v in BASELINE_PARAMS.items()})
        mlflow.log_param("scale_pos_weight", DEFAULT_SCALE_POS_WEIGHT)
        mlflow.log_metrics({f"val_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metric("train_seconds", baseline_dur)

    # --- 2. Optuna --- #
    if skip_optuna:
        print("[train] --skip-optuna: usando params baseline como 'best'")
        best_params = {k: v for k, v in BASELINE_PARAMS.items() if k != "n_estimators"}
        n_trials_done = 0
        optuna_dur = 0.0
    else:
        print(f"[train] lanzando Optuna ({n_trials} trials)")
        t0 = time.time()
        study = tune_optuna(splits, n_trials=n_trials)
        optuna_dur = time.time() - t0
        best_params = study.best_params
        n_trials_done = len(study.trials)
        print(
            f"[train] Optuna best PR-AUC = {study.best_value:.4f} "
            f"({n_trials_done} trials, {optuna_dur:.1f}s)"
        )

    # --- 3. Final fit --- #
    print("[train] re-entrenando con best_params")
    t0 = time.time()
    final_clf, val_metrics, test_metrics = train_final(splits, best_params)
    final_dur = time.time() - t0
    print(f"[train] final val PR-AUC = {val_metrics['pr_auc']:.4f} ({final_dur:.1f}s)")
    print(f"[train] final test PR-AUC = {test_metrics['pr_auc']:.4f}")

    # --- 4. Persistencia + MLflow --- #
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "xgb_best.json"
    final_clf.get_booster().save_model(str(model_path))
    print(f"[train] modelo persistido en {model_path}")

    pr_path, roc_path = _plot_curves(
        splits.y_val.to_numpy(),
        final_clf.predict_proba(splits.X_val)[:, 1],
        config.FIGURES_DIR,
    )
    importance_df = _feature_importance_df(final_clf, splits.feature_names)
    importance_csv = config.REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    with mlflow.start_run(run_name="best"):
        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", DEFAULT_SCALE_POS_WEIGHT)
        mlflow.log_param("n_trials", n_trials_done)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("optuna_seconds", optuna_dur)
        mlflow.log_metric("train_seconds", final_dur)
        mlflow.xgboost.log_model(final_clf.get_booster(), artifact_path="model")
        mlflow.log_artifact(str(pr_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(importance_csv))
        mlflow.log_artifact(str(model_path))

    # --- 5. Reporte --- #
    _write_training_report(
        config.REPORTS_DIR / "training_report.md",
        best_params=best_params,
        baseline_metrics=baseline_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        fraud_rate_val=splits.fraud_rate_val,
        importance_df=importance_df,
        n_trials=n_trials_done,
        duration_s=optuna_dur,
    )
    print(f"[train] reporte escrito en {config.REPORTS_DIR / 'training_report.md'}")

    return {
        "baseline": baseline_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_params": best_params,
        "n_trials": n_trials_done,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 3: train + tune + log")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Usa params baseline como 'best' (debug rápido).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(n_trials=args.n_trials, skip_optuna=args.skip_optuna)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[train][error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

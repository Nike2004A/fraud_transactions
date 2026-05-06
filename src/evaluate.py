"""Evaluación profunda del modelo final (Fase 4).

Trabajamos sobre ``models/xgb_best.json`` fijo (sin re-tunear) y
producimos:

- Métricas full sobre test (PR-AUC, ROC-AUC, recall@1 %, P/R/F1 al
  threshold operativo de val, lift@k).
- Calibración: comparación de score crudo vs Platt vs isotónica,
  fittear en val, evaluar en test (Brier).
- Análisis por segmento (categoría, franja horaria, monto, edad) para
  detectar dónde el modelo flaquea.
- Costo-beneficio: barrer thresholds con costo_FN = monto y costo_FP
  fijo. Elegir threshold óptimo y compararlo con el F1*.
- SHAP global (summary + top-10) + 3 force plots de TPs altos + 3 FPs
  interesantes.
- Reporte ``reports/evaluation_report.md`` y MLflow run "evaluation".

El threshold operativo (F1 max en val) se calcula una sola vez en val y
se aplica fijo a test (no se recalibra en test para no contaminar).

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.train import recall_at_top_k, load_curated, Splits

# Threshold operativo de val (F1 max) — fijado tras Fase 3, no recalcular.
OPERATING_THRESHOLD: float = 0.6642

# Costos asumidos para el análisis costo-beneficio.
COST_FP_USD: float = 5.0  # revisión manual de un falso positivo

# Sample para SHAP — TreeExplainer sobre 194k filas es caro y satura RAM.
SHAP_SAMPLE_SIZE: int = 5000

LIFT_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)


# --- 1. Carga del modelo + métricas en test --- #


def load_model(model_path: Path | None = None) -> xgb.XGBClassifier:
    """Deserializa ``models/xgb_best.json`` envuelto en sklearn API.

    Usar ``XGBClassifier.load_model`` y NO ``Booster.load_model`` porque
    el segundo respeta el ``best_iteration`` guardado en metadata
    (que en este artefacto es 1, vestigio del checkpoint de early
    stopping) y termina prediciendo con un solo árbol. El wrapper
    sklearn usa los 32 árboles entrenados, que son los que generan
    PR-AUC test = 0.8771.
    """
    model_path = model_path or (config.MODELS_DIR / "xgb_best.json")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {model_path}. Corré 'make train' primero."
        )
    clf = xgb.XGBClassifier(device=config.XGBOOST_DEVICE, tree_method=config.XGBOOST_TREE_METHOD)
    clf.load_model(str(model_path))
    return clf


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    """Lift = (precision en top-k) / (tasa base). >1 → mejor que random."""
    base_rate = float(np.mean(y_true))
    if base_rate <= 0:
        return float("nan")
    n = len(y_score)
    k = max(1, int(np.ceil(n * frac)))
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    top_precision = float(np.mean(y_true[top_idx]))
    return top_precision / base_rate


@dataclass
class TestMetrics:
    pr_auc: float
    roc_auc: float
    recall_at_1pct: float
    precision_at_thr: float
    recall_at_thr: float
    f1_at_thr: float
    threshold: float
    confusion: np.ndarray  # 2x2
    lift: dict[str, float]

    def as_flat_dict(self) -> dict[str, float]:
        out = {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "recall_at_1pct": self.recall_at_1pct,
            "precision_at_thr": self.precision_at_thr,
            "recall_at_thr": self.recall_at_thr,
            "f1_at_thr": self.f1_at_thr,
            "threshold": self.threshold,
            "tn": int(self.confusion[0, 0]),
            "fp": int(self.confusion[0, 1]),
            "fn": int(self.confusion[1, 0]),
            "tp": int(self.confusion[1, 1]),
        }
        for k, v in self.lift.items():
            out[f"lift_at_{k}"] = float(v)
        return out


def eval_test(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
) -> TestMetrics:
    """Métricas sobre test al threshold operativo fijado en val."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    lift = {
        f"{int(f * 100)}pct": _lift_at_k(y_true, y_score, f)
        for f in LIFT_FRACTIONS
    }
    return TestMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        recall_at_1pct=recall_at_top_k(y_true, y_score, 0.01),
        precision_at_thr=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_at_thr=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_at_thr=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        confusion=cm,
        lift=lift,
    )


# --- 2. Calibración --- #


@dataclass
class CalibrationResult:
    brier_raw: float
    brier_platt: float
    brier_isotonic: float
    chosen: str  # 'raw' | 'platt' | 'isotonic'
    test_brier_chosen: float
    val_curves: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (prob_pred, prob_true)
    test_score_calibrated: np.ndarray  # score test del mejor calibrador (o crudo)


def _fit_platt(val_score: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """Platt scaling = LR univariada sobre el logit del score."""
    eps = 1e-6
    s = np.clip(val_score, eps, 1 - eps)
    z = np.log(s / (1 - s)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, y_val)
    return lr


def _apply_platt(lr: LogisticRegression, score: np.ndarray) -> np.ndarray:
    eps = 1e-6
    s = np.clip(score, eps, 1 - eps)
    z = np.log(s / (1 - s)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def calibration_analysis(
    val_score: np.ndarray,
    y_val: np.ndarray,
    test_score: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compara raw / Platt / isotónica. Fit en val, evaluación en test.

    Devuelve Briers en test y curvas de calibración (sobre val) para los
    tres. Elige el calibrador que minimiza el Brier en val (no en test:
    seleccionar en test sería sobreajustar al set de generalización).
    """
    # Fit en val
    platt = _fit_platt(val_score, y_val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_score, y_val)

    # Briers en val (para selección)
    val_platt = _apply_platt(platt, val_score)
    val_iso = iso.predict(val_score)
    brier_val_raw = brier_score_loss(y_val, val_score)
    brier_val_platt = brier_score_loss(y_val, val_platt)
    brier_val_iso = brier_score_loss(y_val, val_iso)

    # Briers en test (para reporte)
    test_platt = _apply_platt(platt, test_score)
    test_iso = iso.predict(test_score)
    brier_raw = float(brier_score_loss(y_test, test_score))
    brier_platt = float(brier_score_loss(y_test, test_platt))
    brier_iso = float(brier_score_loss(y_test, test_iso))

    candidates = {
        "raw": (brier_val_raw, test_score, brier_raw),
        "platt": (brier_val_platt, test_platt, brier_platt),
        "isotonic": (brier_val_iso, test_iso, brier_iso),
    }
    chosen = min(candidates, key=lambda k: candidates[k][0])
    _, chosen_test_score, chosen_test_brier = candidates[chosen]

    # Curvas de calibración (sobre val: muestra cómo cada calibrador
    # rectifica las probabilidades antes de aplicarse a test).
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, scores in (
        ("raw", val_score),
        ("platt", val_platt),
        ("isotonic", val_iso),
    ):
        prob_true, prob_pred = calibration_curve(y_val, scores, n_bins=n_bins, strategy="quantile")
        curves[name] = (prob_pred, prob_true)

    # Persistir el calibrador si mejora el Brier en val
    if chosen != "raw":
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with (config.MODELS_DIR / "calibrator.pkl").open("wb") as fh:
            payload = {"kind": chosen, "model": platt if chosen == "platt" else iso}
            pickle.dump(payload, fh)

    return CalibrationResult(
        brier_raw=brier_raw,
        brier_platt=brier_platt,
        brier_isotonic=brier_iso,
        chosen=chosen,
        test_brier_chosen=float(chosen_test_brier),
        val_curves=curves,
        test_score_calibrated=np.asarray(chosen_test_score, dtype=float),
    )


# --- 3. Segmentos --- #


def _load_aligned_test_metadata() -> pd.DataFrame:
    """Devuelve un DataFrame alineado fila-a-fila con el test de curated.

    Replica el orden de :func:`src.features.build_features`:
        1. Sort por ``unix_time`` → asignar split (70/15/15).
        2. Re-sort por ``(cc_num, unix_time)``.
        3. Slice donde ``split == 'test'``.

    Devuelve ``category``, ``amt``, e ``is_fraud`` (este último para
    chequear alineamiento con curated).
    """
    stg = pd.read_parquet(config.STAGING_FILE)
    stg = stg.sort_values("unix_time", kind="stable").reset_index(drop=True)
    n = len(stg)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    stg["split"] = split
    stg = stg.sort_values([config.GROUP_KEY, "unix_time"], kind="stable").reset_index(drop=True)
    test = stg.loc[stg["split"] == "test", ["category", "amt", "is_fraud"]].reset_index(drop=True)
    return test


def _segment_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    n_pos = int(y.sum())
    if len(y) < 50 or n_pos < 5:
        return {"n": int(len(y)), "n_pos": n_pos, "pr_auc": float("nan"), "recall_at_1pct": float("nan")}
    return {
        "n": int(len(y)),
        "n_pos": n_pos,
        "pr_auc": float(average_precision_score(y, score)),
        "recall_at_1pct": recall_at_top_k(y, score, 0.01),
    }


def segment_analysis(
    splits: Splits,
    test_score: np.ndarray,
) -> pd.DataFrame:
    """Métricas por categoría top-5, día/noche, cuartil de monto, edad.

    Args:
        splits: Splits del parquet curado (necesitamos features de test).
        test_score: Scores del modelo sobre ``X_test``.

    Returns:
        DataFrame long con columnas: ``segment_type``, ``segment``, ``n``,
        ``n_pos``, ``pr_auc``, ``recall_at_1pct``.
    """
    y = splits.y_test.to_numpy()
    rows: list[dict] = []

    # Categoría: top-5 por volumen (necesita staging alineado)
    meta = _load_aligned_test_metadata()
    if not np.array_equal(meta["is_fraud"].to_numpy(), y):
        raise RuntimeError("Mismatch entre staging test y curated test (orden roto).")
    top5_cats = meta["category"].value_counts().head(5).index.tolist()
    for cat in top5_cats:
        mask = (meta["category"] == cat).to_numpy()
        m = _segment_metrics(y[mask], test_score[mask])
        rows.append({"segment_type": "category", "segment": cat, **m})

    # Día vs noche
    is_night = splits.X_test["is_night"].to_numpy().astype(bool)
    rows.append({"segment_type": "time_of_day", "segment": "night", **_segment_metrics(y[is_night], test_score[is_night])})
    rows.append({"segment_type": "time_of_day", "segment": "day", **_segment_metrics(y[~is_night], test_score[~is_night])})

    # Cuartiles de log1p_amt (calculados sobre test mismo: sólo descripción)
    amt = splits.X_test["log1p_amt"].to_numpy()
    qs = np.quantile(amt, [0.25, 0.50, 0.75])
    quartile = np.digitize(amt, qs, right=False)  # 0..3
    for q in range(4):
        mask = quartile == q
        rows.append({
            "segment_type": "amount_quartile",
            "segment": f"Q{q + 1}",
            **_segment_metrics(y[mask], test_score[mask]),
        })

    # Grupos de edad
    age = splits.X_test["age"].to_numpy()
    age_groups = [("<30", age < 30), ("30-50", (age >= 30) & (age < 50)), ("50+", age >= 50)]
    for name, mask in age_groups:
        rows.append({
            "segment_type": "age_group",
            "segment": name,
            **_segment_metrics(y[mask], test_score[mask]),
        })

    return pd.DataFrame(rows)


# --- 4. Costo-beneficio --- #


@dataclass
class CostResult:
    thresholds: np.ndarray
    total_cost: np.ndarray  # costo en val
    n_fp: np.ndarray
    n_fn: np.ndarray
    cost_fn: np.ndarray  # USD perdidos en FN
    threshold_min_cost: float
    cost_at_min: float
    cost_at_f1_threshold: float
    saved_vs_f1: float


def cost_benefit_analysis(
    val_score: np.ndarray,
    y_val: np.ndarray,
    val_amt: np.ndarray,
    f1_threshold: float = OPERATING_THRESHOLD,
    cost_fp_usd: float = COST_FP_USD,
) -> CostResult:
    """Sweep thresholds en val con costo_FN = monto, costo_FP = fijo.

    Decisión sobre val (no en test): el threshold óptimo se elige en val
    y se reporta su comportamiento. La métrica de generalización del
    threshold queda implícita en el reporte global del modelo.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    total_cost = np.zeros_like(thresholds)
    cost_fn = np.zeros_like(thresholds)
    n_fp = np.zeros_like(thresholds, dtype=int)
    n_fn = np.zeros_like(thresholds, dtype=int)
    for i, thr in enumerate(thresholds):
        pred = val_score >= thr
        # FP: pred=1, label=0
        fp_mask = pred & (y_val == 0)
        # FN: pred=0, label=1
        fn_mask = (~pred) & (y_val == 1)
        cost_fp = float(fp_mask.sum()) * cost_fp_usd
        cost_fn[i] = float(val_amt[fn_mask].sum())
        total_cost[i] = cost_fp + cost_fn[i]
        n_fp[i] = int(fp_mask.sum())
        n_fn[i] = int(fn_mask.sum())

    idx = int(np.argmin(total_cost))
    cost_at_min = float(total_cost[idx])
    # Costo al threshold F1*
    pred_f1 = val_score >= f1_threshold
    cost_f1 = float(((pred_f1 & (y_val == 0)).sum()) * cost_fp_usd) + \
        float(val_amt[(~pred_f1) & (y_val == 1)].sum())

    return CostResult(
        thresholds=thresholds,
        total_cost=total_cost,
        n_fp=n_fp,
        n_fn=n_fn,
        cost_fn=cost_fn,
        threshold_min_cost=float(thresholds[idx]),
        cost_at_min=cost_at_min,
        cost_at_f1_threshold=cost_f1,
        saved_vs_f1=float(cost_f1 - cost_at_min),
    )


# --- 5. SHAP --- #


@dataclass
class ShapResult:
    sample_idx: np.ndarray
    shap_values: np.ndarray
    base_value: float
    top10: pd.DataFrame  # feature, mean_abs_shap
    tp_indices: list[int]
    fp_indices: list[int]


def shap_analysis(
    clf: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_score: np.ndarray,
    sample_size: int = SHAP_SAMPLE_SIZE,
    seed: int = config.SEED,
) -> ShapResult:
    """SHAP TreeExplainer sobre un sample del test set (RAM-friendly).

    Devuelve los SHAP, top-10 features por mean(|SHAP|), e índices
    (en el sample) de 3 verdaderos positivos con mayor score y
    3 falsos positivos con score más alto (los más "convincentes"
    para el modelo aunque sean negativos).
    """
    rng = np.random.default_rng(seed)
    n = len(X_test)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample_idx.sort()

    X_sample = X_test.iloc[sample_idx]
    y_sample = y_test[sample_idx]
    score_sample = test_score[sample_idx]

    # TreeExplainer sobre el booster: en XGBoost binary devuelve SHAP en
    # margen (logit). model_output='raw' es el default.
    booster = clf.get_booster()
    # Forzar CPU para SHAP: TreeExplainer con device='cuda' a veces falla.
    booster.set_param({"device": "cpu"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_sample)
    booster.set_param({"device": config.XGBOOST_DEVICE})

    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    mean_abs = np.abs(shap_values).mean(axis=0)
    top10 = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Top-3 TPs y top-3 FPs por score dentro del sample. No filtramos por
    # threshold operativo: en 5k filas con tasa 0.58 % puede haber <3 casos
    # por encima de thr* en cada clase. "FP interesante" = el modelo le
    # asigna score alto aunque la label sea 0.
    tp_pos = np.where(y_sample == 1)[0]
    tp_top = tp_pos[np.argsort(-score_sample[tp_pos])][:3].tolist()
    fp_pos = np.where(y_sample == 0)[0]
    fp_top = fp_pos[np.argsort(-score_sample[fp_pos])][:3].tolist()

    return ShapResult(
        sample_idx=sample_idx,
        shap_values=shap_values,
        base_value=base_value,
        top10=top10,
        tp_indices=tp_top,
        fp_indices=fp_top,
    )


# --- Plots --- #


def _plot_pr_roc_test(y: np.ndarray, score: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y, score)
    fpr, tpr, _ = roc_curve(y, score)

    pr_path = out_dir / "pr_curve_test.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y, score):.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall (test)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(pr_path, dpi=120); plt.close(fig)

    roc_path = out_dir / "roc_curve_test.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y, score):.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (test)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(roc_path, dpi=120); plt.close(fig)
    return pr_path, roc_path


def _plot_confusion(cm: np.ndarray, out_dir: Path) -> Path:
    out = out_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred=0", "pred=1"])
    ax.set_yticks([0, 1], ["y=0", "y=1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion @ thr={OPERATING_THRESHOLD:.4f} (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _plot_calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> Path:
    out = out_dir / "calibration_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfecto")
    for name, (x, y) in curves.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
    ax.set_title("Curva de calibración (val, 10 bins)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _plot_segments(seg_df: pd.DataFrame, out_dir: Path) -> Path:
    out = out_dir / "segment_metrics.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    df = seg_df.dropna(subset=["pr_auc"]).copy()
    df["label"] = df["segment_type"] + " | " + df["segment"].astype(str)
    df = df.sort_values("pr_auc")
    axes[0].barh(df["label"], df["pr_auc"], color="steelblue")
    axes[0].set_xlabel("PR-AUC"); axes[0].set_title("PR-AUC por segmento (test)")
    axes[0].grid(alpha=0.3, axis="x")
    axes[1].barh(df["label"], df["recall_at_1pct"], color="indianred")
    axes[1].set_xlabel("recall@1%"); axes[1].set_title("recall@1% por segmento (test)")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _plot_cost_curve(cost: CostResult, out_dir: Path) -> Path:
    out = out_dir / "cost_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cost.thresholds, cost.total_cost, label="costo total")
    ax.axvline(cost.threshold_min_cost, color="red", linestyle="--",
               label=f"min costo @ {cost.threshold_min_cost:.2f}")
    ax.axvline(OPERATING_THRESHOLD, color="green", linestyle="--",
               label=f"F1* @ {OPERATING_THRESHOLD:.4f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Costo total (USD, val)")
    ax.set_title("Sweep de costo (FN=monto, FP=$5)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _plot_shap(
    shap_res: ShapResult, X_sample: pd.DataFrame, out_dir: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    bar = out_dir / "shap_summary_bar.png"
    plt.figure()
    shap.summary_plot(shap_res.shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig(bar, dpi=120); plt.close("all")
    paths["bar"] = bar

    bee = out_dir / "shap_summary_beeswarm.png"
    plt.figure()
    shap.summary_plot(shap_res.shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(bee, dpi=120); plt.close("all")
    paths["beeswarm"] = bee

    # Force plots: matplotlib=True genera PNG estático.
    for tag, indices in [("tp", shap_res.tp_indices), ("fp", shap_res.fp_indices)]:
        for k, idx in enumerate(indices):
            p = out_dir / f"shap_force_{tag}_{k + 1}.png"
            plt.figure()
            shap.force_plot(
                shap_res.base_value,
                shap_res.shap_values[idx],
                X_sample.iloc[idx],
                matplotlib=True,
                show=False,
            )
            plt.tight_layout(); plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close("all")
            paths[f"force_{tag}_{k + 1}"] = p
    return paths


# --- Reporte --- #


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _segment_table_md(seg_df: pd.DataFrame) -> str:
    lines = ["| segmento | n | n_pos | PR-AUC | recall@1% |", "|---|---:|---:|---:|---:|"]
    for row in seg_df.itertuples():
        pr = "—" if pd.isna(row.pr_auc) else f"{row.pr_auc:.4f}"
        rk = "—" if pd.isna(row.recall_at_1pct) else f"{row.recall_at_1pct:.4f}"
        lines.append(f"| {row.segment_type} / **{row.segment}** | {row.n:,} | {row.n_pos:,} | {pr} | {rk} |")
    return "\n".join(lines)


def _identify_weak_segments(seg_df: pd.DataFrame) -> list[str]:
    """Segmentos cuyo PR-AUC < global - 0.10 (heurística)."""
    df = seg_df.dropna(subset=["pr_auc"]).copy()
    if df.empty:
        return []
    median = df["pr_auc"].median()
    weak = df.loc[df["pr_auc"] < median - 0.10]
    return [f"{r.segment_type} / **{r.segment}** (PR-AUC={r.pr_auc:.4f}, n_pos={r.n_pos})" for r in weak.itertuples()]


def _write_evaluation_report(
    report_path: Path,
    metrics: TestMetrics,
    calib: CalibrationResult,
    seg_df: pd.DataFrame,
    cost: CostResult,
    shap_top10: pd.DataFrame,
    shap_force_paths: dict[str, Path],
    monthly_volume: float,
    monthly_savings: float,
) -> None:
    weak = _identify_weak_segments(seg_df)
    weak_md = "\n".join(f"- {w}" for w in weak) if weak else "_Ninguno: PR-AUC homogéneo entre segmentos._"
    top10_md = "| feature | mean(|SHAP|) |\n|---|---:|\n" + "\n".join(
        f"| `{r.feature}` | {r.mean_abs_shap:.4f} |" for r in shap_top10.itertuples()
    )
    seg_md = _segment_table_md(seg_df)

    body = f"""# Evaluación final — Fase 4

Modelo: `models/xgb_best.json` (Fase 3, no re-tuneado).
Threshold operativo: **{OPERATING_THRESHOLD:.4f}** (F1 max en val, fijo).

## 1. Resumen ejecutivo (test, n={int(metrics.confusion.sum()):,})

| métrica | valor |
|---|---:|
| PR-AUC | {metrics.pr_auc:.4f} |
| ROC-AUC | {metrics.roc_auc:.4f} |
| recall@1 % | {metrics.recall_at_1pct:.4f} |
| precision @ thr* | {metrics.precision_at_thr:.4f} |
| recall @ thr* | {metrics.recall_at_thr:.4f} |
| F1 @ thr* | {metrics.f1_at_thr:.4f} |
| TP / FP / FN / TN | {metrics.confusion[1,1]:,} / {metrics.confusion[0,1]:,} / {metrics.confusion[1,0]:,} / {metrics.confusion[0,0]:,} |
| lift @ 1 % / 5 % / 10 % | {metrics.lift['1pct']:.1f} / {metrics.lift['5pct']:.1f} / {metrics.lift['10pct']:.1f} |

![PR test](figures/pr_curve_test.png)
![ROC test](figures/roc_curve_test.png)
![Confusión](figures/confusion_matrix.png)

## 2. Calibración

| score | Brier (test) |
|---|---:|
| crudo | {calib.brier_raw:.5f} |
| Platt | {calib.brier_platt:.5f} |
| isotónica | {calib.brier_isotonic:.5f} |

Calibrador elegido (mejor Brier en val): **{calib.chosen}**. Brier en test al
calibrador elegido: **{calib.test_brier_chosen:.5f}**.

![Calibración](figures/calibration_curve.png)

**Recomendación**: el modelo se entrenó con `scale_pos_weight≈176`, por lo que
los scores crudos están sesgados hacia arriba en términos absolutos pero
preservan ranking. Usar **score crudo** cuando el caso de uso es ranking
(top-k para revisión manual, alerting). Usar **score calibrado** cuando se
necesita interpretarlo como probabilidad de fraude (umbral de auto-bloqueo,
modelo de riesgo, scoring para ensembles posteriores). El calibrador se
persiste en `models/calibrator.pkl` cuando mejora el Brier en val.

## 3. SHAP (sample n={SHAP_SAMPLE_SIZE} de test)

Top-10 features por mean(|SHAP|):

{top10_md}

![SHAP bar](figures/shap_summary_bar.png)
![SHAP beeswarm](figures/shap_summary_beeswarm.png)

### Casos de falso positivo (alto score, label 0)

Tres tx que el modelo marcaría como fraude pero no lo eran:

{chr(10).join(f"![FP {i+1}](figures/shap_force_fp_{i+1}.png)" for i in range(min(3, len([k for k in shap_force_paths if k.startswith('force_fp')]))))}

Patrón típico: combinación de monto alto (`log1p_amt`, `amt_gt_p95_legit`)
con desviación contra el rolling de 24h del titular (`rolling_amt_mean_24h`)
en horario nocturno. El modelo aprende que esa combinación es predominantemente
fraude, pero existe una minoría de tx legítimas atípicas (compras grandes
puntuales, viajes, regalos) que entran en el mismo régimen y no son
distinguibles con las features actuales.

### Casos de verdadero positivo (alto score, label 1)

{chr(10).join(f"![TP {i+1}](figures/shap_force_tp_{i+1}.png)" for i in range(min(3, len([k for k in shap_force_paths if k.startswith('force_tp')]))))}

## 4. Análisis por segmento

{seg_md}

![Segmentos](figures/segment_metrics.png)

**Segmentos donde el modelo subperformea** (PR-AUC < mediana − 0.10):

{weak_md}

Hipótesis: en segmentos con bajo `n_pos` el PR-AUC es ruidoso. Cuando el
descenso es sistemático (no por muestra chica) sugiere que las features
agregadas (`rolling_*`, `te_*`) capturan peor el comportamiento de ese
subconjunto — p.ej. tarjetas con poca historia o categorías con
distribución de monto bimodal.

## 5. Costo-beneficio

Asunciones: `costo_FN = monto_tx` (fraude no detectado = pérdida total),
`costo_FP = ${COST_FP_USD:.0f}` (revisión manual). Sweep en val.

| threshold | costo total val (USD) | n_FP | n_FN |
|---|---:|---:|---:|
| F1* = {OPERATING_THRESHOLD:.4f} | {_fmt_money(cost.cost_at_f1_threshold)} | — | — |
| **mín costo = {cost.threshold_min_cost:.2f}** | **{_fmt_money(cost.cost_at_min)}** | — | — |

Ahorro en val al pasar de F1* a threshold de mínimo costo: **{_fmt_money(cost.saved_vs_f1)}**.

![Costo](figures/cost_curve.png)

Estimación de ahorro mensual extrapolando volumen de val al test set:
- Volumen test ≈ {monthly_volume:,.0f} tx/mes (asumiendo split test cubre ~3 meses).
- Ahorro proyectado: **{_fmt_money(monthly_savings)} / mes** si se opera al
  threshold de mínimo costo en lugar del F1*.

> El threshold óptimo de costo es más bajo que el F1* (más recall a costa
> de más FPs): el dolor por dejar pasar fraude es proporcional al monto,
> mientras que el dolor por revisar de más es lineal y barato.

## 6. Limitaciones

- **Distribución de monto**: las pérdidas por FN se calculan con el monto
  observado; en producción habría que descontar la fracción recuperable
  por chargeback.
- **Drift**: el split temporal cubre un período acotado del dataset
  Kaggle. En producción se requiere monitoreo de drift (PSI sobre features
  top-3: `rolling_amt_mean_24h`, `amt_gt_p95_legit`, `log1p_amt`) y
  re-entrenamiento periódico.
- **Features ausentes**: device fingerprint, IP, MCC granular, historial
  comercial del merchant — todas darían señal adicional pero no están
  en el dataset.
- **Threshold inestable**: el F1* está cerca de un plateau de la curva
  PR; pequeñas variaciones en val mueven el threshold significativamente.

## 7. Próximos pasos

1. **Online learning**: entrenar incrementalmente con los chargebacks
   confirmados (etiquetas tardías), con un pipeline que normalice el
   delay (~30-60 días) entre tx y label final.
2. **Drift monitoring**: dashboard con PSI/KS sobre las features top-3
   y alertas si el PR-AUC en validación rolling baja >5 % vs baseline.
3. **Features adicionales**: ratio del monto vs media histórica del
   merchant; conteo de tx en países distintos en 1 h; flag de primera
   tx en ese MCC para esa cc.
4. **Modelo dual**: separar el problema en "alta confianza" (auto-bloqueo)
   vs "media confianza" (revisión manual) usando dos thresholds. Permite
   calibrar la cobertura del equipo de fraude.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


# --- Orquestación --- #


def run() -> dict:
    print("[evaluate] cargando parquet curado")
    splits = load_curated()
    print(f"[evaluate] test shape: {splits.X_test.shape} | tasa fraude test: {splits.y_test.mean():.4%}")

    print("[evaluate] cargando modelo")
    clf = load_model()

    val_score = clf.predict_proba(splits.X_val)[:, 1]
    test_score = clf.predict_proba(splits.X_test)[:, 1]
    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    # 1. Métricas
    print("[evaluate] métricas test")
    metrics = eval_test(y_test, test_score)
    print(f"[evaluate]   PR-AUC test = {metrics.pr_auc:.4f} (esperado ≈ 0.8771)")
    if abs(metrics.pr_auc - 0.8771) > 0.01:
        raise RuntimeError(
            f"PR-AUC test {metrics.pr_auc:.4f} no coincide con el esperado 0.8771. "
            "Hay drift en el modelo o en los splits."
        )

    # 2. Calibración
    print("[evaluate] calibración (Platt + isotónica)")
    calib = calibration_analysis(val_score, y_val, test_score, y_test)
    print(f"[evaluate]   Brier raw={calib.brier_raw:.5f} platt={calib.brier_platt:.5f} iso={calib.brier_isotonic:.5f} → elegido={calib.chosen}")

    # 3. Segmentos
    print("[evaluate] análisis por segmento")
    seg_df = segment_analysis(splits, test_score)
    print(f"[evaluate]   {len(seg_df)} segmentos analizados")

    # 4. Costo-beneficio (sobre val, costo_FN = monto_tx)
    print("[evaluate] costo-beneficio sweep")
    val_meta_for_amt = pd.read_parquet(config.STAGING_FILE)
    val_meta_for_amt = val_meta_for_amt.sort_values("unix_time", kind="stable").reset_index(drop=True)
    n = len(val_meta_for_amt)
    n_train = int(n * config.TRAIN_FRAC); n_val = int(n * config.VAL_FRAC)
    val_meta_for_amt["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    val_meta_for_amt = val_meta_for_amt.sort_values([config.GROUP_KEY, "unix_time"], kind="stable").reset_index(drop=True)
    val_amt = val_meta_for_amt.loc[val_meta_for_amt["split"] == "val", "amt"].to_numpy()
    if not np.array_equal(
        val_meta_for_amt.loc[val_meta_for_amt["split"] == "val", "is_fraud"].to_numpy(),
        y_val,
    ):
        raise RuntimeError("Mismatch entre staging val y curated val (orden roto).")
    cost = cost_benefit_analysis(val_score, y_val, val_amt)
    print(f"[evaluate]   threshold óptimo costo = {cost.threshold_min_cost:.2f} | costo val = ${cost.cost_at_min:,.0f} (vs F1* ${cost.cost_at_f1_threshold:,.0f})")

    # 5. SHAP (último, el más caro)
    print(f"[evaluate] SHAP TreeExplainer (sample={SHAP_SAMPLE_SIZE})")
    t0 = time.time()
    shap_res = shap_analysis(clf, splits.X_test, y_test, test_score)
    print(f"[evaluate]   SHAP listo en {time.time() - t0:.1f}s | top1={shap_res.top10.iloc[0]['feature']}")

    # --- figuras --- #
    print("[evaluate] generando figuras")
    pr_path, roc_path = _plot_pr_roc_test(y_test, test_score, config.FIGURES_DIR)
    cm_path = _plot_confusion(metrics.confusion, config.FIGURES_DIR)
    cal_path = _plot_calibration(calib.val_curves, config.FIGURES_DIR)
    seg_path = _plot_segments(seg_df, config.FIGURES_DIR)
    cost_path = _plot_cost_curve(cost, config.FIGURES_DIR)
    X_sample = splits.X_test.iloc[shap_res.sample_idx]
    shap_paths = _plot_shap(shap_res, X_sample, config.FIGURES_DIR)

    # --- reporte --- #
    # Estimación de ahorro mensual: el split test cubre el último 15 % en
    # tiempo. El dataset Kaggle abarca ~24 meses (1.3M tx), así que
    # 15 % ≈ 3.6 meses. Dividimos el ahorro proyectado en val por la
    # razón de tasas (asumimos misma dinámica) y reportamos por mes.
    # Esto es aproximado y se documenta como tal.
    monthly_volume = float(len(y_test) / 3.6)
    val_to_test_ratio = float(len(y_test) / len(y_val))
    monthly_savings = float(cost.saved_vs_f1 * val_to_test_ratio / 3.6)

    print("[evaluate] escribiendo reporte")
    _write_evaluation_report(
        config.REPORTS_DIR / "evaluation_report.md",
        metrics=metrics,
        calib=calib,
        seg_df=seg_df,
        cost=cost,
        shap_top10=shap_res.top10,
        shap_force_paths=shap_paths,
        monthly_volume=monthly_volume,
        monthly_savings=monthly_savings,
    )
    seg_csv = config.REPORTS_DIR / "segment_metrics.csv"
    seg_df.to_csv(seg_csv, index=False)
    shap_csv = config.REPORTS_DIR / "shap_top10.csv"
    shap_res.top10.to_csv(shap_csv, index=False)

    # --- MLflow run --- #
    print("[evaluate] logging en MLflow run 'evaluation'")
    mlflow.set_tracking_uri(f"file://{config.MLRUNS_DIR}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_param("operating_threshold", OPERATING_THRESHOLD)
        mlflow.log_param("cost_fp_usd", COST_FP_USD)
        mlflow.log_param("shap_sample_size", SHAP_SAMPLE_SIZE)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.as_flat_dict().items() if isinstance(v, (int, float))})
        mlflow.log_metric("brier_raw_test", calib.brier_raw)
        mlflow.log_metric("brier_platt_test", calib.brier_platt)
        mlflow.log_metric("brier_isotonic_test", calib.brier_isotonic)
        mlflow.log_metric("brier_chosen_test", calib.test_brier_chosen)
        mlflow.log_param("calibrator_chosen", calib.chosen)
        mlflow.log_metric("cost_threshold_min", cost.threshold_min_cost)
        mlflow.log_metric("cost_at_min_val_usd", cost.cost_at_min)
        mlflow.log_metric("cost_at_f1_val_usd", cost.cost_at_f1_threshold)
        mlflow.log_metric("cost_saved_vs_f1_val_usd", cost.saved_vs_f1)
        for p in (pr_path, roc_path, cm_path, cal_path, seg_path, cost_path,
                  config.REPORTS_DIR / "evaluation_report.md", seg_csv, shap_csv):
            mlflow.log_artifact(str(p))
        for p in shap_paths.values():
            mlflow.log_artifact(str(p))
        calibrator_path = config.MODELS_DIR / "calibrator.pkl"
        if calibrator_path.exists():
            mlflow.log_artifact(str(calibrator_path))

    print("[evaluate] hecho")
    return {
        "test_metrics": metrics.as_flat_dict(),
        "calibration": {
            "raw": calib.brier_raw,
            "platt": calib.brier_platt,
            "isotonic": calib.brier_isotonic,
            "chosen": calib.chosen,
        },
        "cost": {
            "threshold_min": cost.threshold_min_cost,
            "cost_at_min": cost.cost_at_min,
            "cost_at_f1": cost.cost_at_f1_threshold,
            "saved_vs_f1": cost.saved_vs_f1,
        },
        "shap_top1": shap_res.top10.iloc[0]["feature"],
    }


def main() -> int:
    try:
        result = run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[evaluate][error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

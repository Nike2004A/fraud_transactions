# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the `fraud_transactions/` root.

```bash
make install        # pip install -e ".[dev]"
make test           # pytest -v (runs tests/)
make all            # full pipeline: ingest → features → train → evaluate
make clean          # remove generated artifacts (parquet, models, mlruns, caches)
```

**Individual pipeline phases:**
```bash
python -m src.ingest       # Phase 1: CSV → staging parquet
python -m src.features     # Phase 2: staging → curated (27 features)
python -m src.train        # Phase 3: XGBoost + Optuna + MLflow
python -m src.evaluate     # Phase 4: metrics, calibration, SHAP, reports
```

`src.train` accepts `--n-trials N` (default 50) and `--skip-optuna` flags.

**Linting:** `ruff check .` — configured in pyproject.toml (line-length=100, py311, rules: E/F/I/N/UP/B/SIM/PL, ignores PLR0913).

**Single test:** `python -m pytest tests/test_features.py -v`

## Architecture

Sequential 4-phase DAG pipeline — each phase reads from the previous phase's output and is idempotent:

```
data/raw/*.csv → [ingest] → data/staging/transactions.parquet
    → [features] → data/curated/transactions_features.parquet
    → [train] → models/xgb_best.json + mlruns/
    → [evaluate] → reports/ (markdown, CSV, figures)
```

**Modules:**
- `src/config.py` — single source of truth for all paths, constants, hyperparameters, and seeds. Every module imports from here; no magic constants elsewhere.
- `src/ingest.py` — reads Kaggle CSV pair via DuckDB, validates schema against EXPECTED_COLUMNS, sorts by unix_time, writes ZSTD parquet.
- `src/features.py` — builds 27 features (amount, temporal, demographic, geographic, rolling, target-encoded). Outputs curated parquet with a `split` column (train/val/test).
- `src/train.py` — loads curated splits, runs baseline smoke check, Optuna TPE search (50 trials, MedianPruner), final retrain, MLflow logging.
- `src/evaluate.py` — test metrics, Platt + isotonic calibration on val, segment analysis (category/hour/amount/age), cost-benefit threshold sweep, SHAP TreeExplainer on 5K sample.

**Tracking:** MLflow experiment `fraud_xgboost` (file-based backend in `mlruns/`).

## Anti-Leakage Invariants

These constraints prevent information from the future or from the test set from leaking into training. They are tested in `tests/test_features.py` and must never be relaxed:

1. **Temporal splits** — 70/15/15 train/val/test ordered by `unix_time`. No random shuffling. Split assignment happens before any feature computation.
2. **Rolling windows use `closed='left'`** — row i only sees rows strictly before i (1h, 24h, 7d windows).
3. **Target encoding is KFold out-of-fold on train only** — the mapping is fitted on training data; val/test use the frozen train mapping.
4. **`amt_gt_p95_legit`** — the p95 threshold is calculated from legitimate (non-fraud) transactions in the train split only.
5. **No `cc_num` as feature** — it's used as GROUP_KEY for rolling aggregations then dropped.

## Key Conventions

- **Primary metric:** PR-AUC (not accuracy or ROC-AUC) — the dataset is heavily imbalanced (0.564% fraud).
- **Operating threshold:** derived from F1-max on validation; also cross-checked via cost-benefit sweep (cost_FN = transaction amount, cost_FP = $5 manual review).
- **Reproducibility:** `SEED=42` pinned everywhere. All XGBoost training uses `device='cuda'`, `tree_method='hist'`.
- **scale_pos_weight ≈ 176** — used instead of SMOTE or undersampling (see `docs/architecture_decisions.md` ADR-3).
- **Python 3.11** required (>=3.11, <3.12).
- **Data format:** Parquet with ZSTD compression throughout. Raw CSVs go in `data/raw/` and are gitignored.
- **OUTPUT_FEATURES tuple** in `features.py` is the explicit contract for which 27 features enter the model.

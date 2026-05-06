# Documentación del proyecto

Carpeta orientada a soportar una presentación / defensa del capstone. Cada documento mapea a un bloque de slides distinto.

## Mapa de documentos

| documento | propósito | uso típico en PowerPoint |
|---|---|---|
| [`executive_summary.md`](executive_summary.md) | 1 página — el "qué hicimos y qué vendió" | slide 1-2 + leave-behind |
| [`methodology.md`](methodology.md) | pipeline end-to-end + diagrama de flujo | slides 3-5 (cómo se construyó) |
| [`architecture_decisions.md`](architecture_decisions.md) | ADRs cortos — por qué cada decisión técnica | apoyo en Q&A / apéndice |
| [`model_evaluation.md`](model_evaluation.md) | análisis profundo de resultados | slides 6-10 (resultados + caveats) |
| [`presentation_outline.md`](presentation_outline.md) | estructura de slides sugerida + speaker notes | guía mientras armás el deck |

## Reportes técnicos asociados

Los números crudos viven en [`../reports/`](../reports/):

- [`reports/evaluation_report.md`](../reports/evaluation_report.md) — reporte completo de Fase 4
- [`reports/training_report.md`](../reports/training_report.md) — Optuna + métricas de Fase 3
- [`reports/eda_report.md`](../reports/eda_report.md) — exploración inicial (Fase 1)
- [`reports/features_report.md`](../reports/features_report.md) — feature engineering (Fase 2)
- `reports/*.csv` — tablas (segment metrics, SHAP top-10, feature importance)

## Figuras

- [`figures/`](figures/) — copias de las figuras clave referenciadas por los documentos en esta carpeta
- `../reports/figures/` — set completo de figuras (no trackeado en git: regenerable con `make all`)

## Cómo regenerar todo

```bash
make ingest     # CSV crudo → parquet staging
make features   # staging → curated (~27 features)
make train      # XGBoost + Optuna + MLflow
make evaluate   # métricas + SHAP + calibración + reporte
make test       # pytest (14 tests)
```

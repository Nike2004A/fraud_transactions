"""CSV crudo -> DuckDB -> parquet en data/staging/.

Carga los CSV de Kaggle ("Credit Card Transactions Fraud Detection"),
valida el esquema, reporta nulos y balance de clases, ordena por
``unix_time`` y persiste un único parquet listo para feature engineering.

Uso:
    python -m src.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from src import config

EXPECTED_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
    "merch_zipcode",
)


def _resolve_input_files() -> list[Path]:
    """Devuelve los CSV presentes en ``data/raw/``.

    Prioriza el par canónico de Kaggle (fraudTrain.csv + fraudTest.csv).
    Si no aparecen, hace fallback a cualquier *.csv presente.

    Raises:
        FileNotFoundError: si no hay ningún CSV en ``data/raw/``.
    """
    canonical = [p for p in (config.RAW_TRAIN_FILE, config.RAW_TEST_FILE) if p.exists()]
    if canonical:
        return canonical
    fallback = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    if not fallback:
        raise FileNotFoundError(
            f"No se encontró ningún CSV en {config.DATA_RAW_DIR}. "
            "Colocá fraudTrain.csv / fraudTest.csv (Kaggle) y reintentá."
        )
    return fallback


def _validate_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Valida que la tabla tenga las columnas esperadas del dataset Kaggle."""
    cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    missing = set(EXPECTED_COLUMNS) - cols
    extra = cols - set(EXPECTED_COLUMNS)
    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {sorted(missing)}")
    if extra:
        print(f"[ingest][warn] columnas extra (se conservan): {sorted(extra)}")


def _print_quality_report(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Imprime balance de clases, rango temporal y nulos por columna."""
    n_rows: int = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    n_fraud: int = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {config.TARGET_COLUMN} = 1"
    ).fetchone()[0]
    fraud_rate = n_fraud / n_rows if n_rows else 0.0
    d_min, d_max = con.execute(
        f"SELECT MIN(trans_date_trans_time), MAX(trans_date_trans_time) FROM {table}"
    ).fetchone()
    n_cards: int = con.execute(
        f"SELECT COUNT(DISTINCT {config.GROUP_KEY}) FROM {table}"
    ).fetchone()[0]

    print("=" * 64)
    print(f"Filas totales:        {n_rows:,}")
    print(f"Fraudes:              {n_fraud:,} ({fraud_rate:.4%})")
    print(f"Rango temporal:       {d_min} -> {d_max}")
    print(f"Tarjetas únicas:      {n_cards:,}")
    print("-- nulos por columna (>0) --")
    any_nulls = False
    for col in EXPECTED_COLUMNS:
        n_null = con.execute(
            f'SELECT SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) FROM {table}'
        ).fetchone()[0]
        if n_null:
            any_nulls = True
            print(f"  {col:<28} {n_null:,}")
    if not any_nulls:
        print("  (sin nulos)")
    print("=" * 64)


def ingest(output_path: Path | None = None) -> Path:
    """Lee los CSV crudos, valida, ordena por ``unix_time`` y escribe parquet.

    Args:
        output_path: Destino del parquet. Default: ``config.STAGING_FILE``.

    Returns:
        Path al parquet escrito.
    """
    output_path = output_path or config.STAGING_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_input_files()
    print(f"[ingest] inputs: {[p.name for p in inputs]}")

    con = duckdb.connect()
    files_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in inputs) + "]"
    con.execute(
        f"CREATE TEMP TABLE raw AS "
        f"SELECT * FROM read_csv_auto({files_sql}, header=True, union_by_name=True)"
    )

    _validate_schema(con, "raw")
    _print_quality_report(con, "raw")

    con.execute(
        f"COPY (SELECT * FROM raw ORDER BY unix_time) "
        f"TO '{output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[ingest] wrote {output_path} ({size_mb:,.1f} MB)")
    return output_path


def main() -> int:
    try:
        ingest()
    except FileNotFoundError as exc:
        print(f"[ingest][error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test del bootstrap: imports y consistencia mínima de config."""
from __future__ import annotations


def test_config_paths_exist() -> None:
    from src import config

    assert config.ROOT_DIR.exists()
    assert config.DATA_RAW_DIR.exists()
    assert config.DATA_STAGING_DIR.exists()
    assert config.DATA_CURATED_DIR.exists()


def test_split_fractions_sum_to_one() -> None:
    from src import config

    total = config.TRAIN_FRAC + config.VAL_FRAC + config.TEST_FRAC
    assert abs(total - 1.0) < 1e-9


def test_target_not_in_drop_columns() -> None:
    from src import config

    assert config.TARGET_COLUMN not in config.DROP_COLUMNS
    assert config.GROUP_KEY not in config.DROP_COLUMNS


def test_ingest_module_importable() -> None:
    from src import ingest

    assert callable(ingest.ingest)
    assert "is_fraud" in ingest.EXPECTED_COLUMNS

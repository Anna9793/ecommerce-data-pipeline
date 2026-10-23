import os
from pathlib import Path
import pytest


def test_paths_definition_and_structure():
    """Verify that config.paths defines valid Path objects and required directories exist."""
    from config.paths import (
        BASE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
        REPORTS_DIR, MODELS_DIR, CONFIG_DIR,
        ONLINE_RETAIL_XLSX, ONLINE_RETAIL_CSV, CLEAN_RETAIL,
        FEATURE_RETAIL, RFM_CUSTOMERS, CLUSTER_PROFILE, EXPERIMENT_CONFIG_PATH
    )
    assert isinstance(BASE_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(RAW_DIR, Path)
    assert isinstance(PROCESSED_DIR, Path)
    assert isinstance(ONLINE_RETAIL_CSV, Path)
    assert isinstance(CLEAN_RETAIL, Path)
    assert isinstance(RFM_CUSTOMERS, Path)
    assert isinstance(CLUSTER_PROFILE, Path)
    assert isinstance(EXPERIMENT_CONFIG_PATH, Path)

    # Check directory existence
    assert DATA_DIR.exists()
    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert PREDICTIONS_DIR.exists()


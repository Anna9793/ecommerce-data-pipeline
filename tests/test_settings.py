import os
from pathlib import Path
import pytest
from config.settings import Settings, get_settings


def test_settings_default_values(monkeypatch):
    """Verify default configurations adhere to 12-Factor safe local defaults."""
    monkeypatch.delenv("USE_BIGQUERY", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.GCP_PROJECT == "anna-ml-pipeline"
    assert settings.GCP_LOCATION == "us-central1"
    assert settings.USE_BIGQUERY is False
    assert settings.RFM_FEATURE_VERSION == "rfm_v1"
    assert settings.CHURN_FEATURE_VERSION == "churn_v2"
    assert settings.TWO_TOWER_VERSION == "two_tower_v1"
    assert settings.API_PORT == 8000


def test_settings_environment_override(monkeypatch):
    """Verify settings dynamically load environment overrides."""
    monkeypatch.setenv("GCP_PROJECT", "override-gcp-project")
    monkeypatch.setenv("USE_BIGQUERY", "true")
    monkeypatch.setenv("CHURN_FEATURE_VERSION", "churn_v3_custom")
    
    settings = Settings()
    assert settings.GCP_PROJECT == "override-gcp-project"
    assert settings.USE_BIGQUERY is True
    assert settings.CHURN_FEATURE_VERSION == "churn_v3_custom"


def test_settings_derived_paths_exist():
    """Verify all canonical filesystem directory paths and dataset paths resolve to valid Paths."""
    settings = get_settings()
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.RAW_DIR, Path)
    assert isinstance(settings.PROCESSED_DIR, Path)
    assert isinstance(settings.PREDICTIONS_DIR, Path)
    assert isinstance(settings.RFM_CUSTOMERS, Path)
    assert isinstance(settings.CLUSTER_PROFILE, Path)
    assert isinstance(settings.EXPERIMENT_CONFIG_PATH, Path)
    assert settings.RAW_DIR.exists()
    assert settings.PROCESSED_DIR.exists()


def test_paths_backward_compatibility():
    """Verify that config.paths re-exports remain 100% backward compatible."""
    from config.paths import (
        BASE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
        RFM_CUSTOMERS, CLUSTER_PROFILE, EXPERIMENT_CONFIG_PATH
    )
    assert isinstance(BASE_DIR, Path)
    assert isinstance(RFM_CUSTOMERS, Path)
    assert isinstance(CLUSTER_PROFILE, Path)
    assert isinstance(EXPERIMENT_CONFIG_PATH, Path)

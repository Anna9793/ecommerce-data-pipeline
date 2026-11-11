import pytest
from pydantic import ValidationError
from config.loader import load_config
from config.schema import ExperimentConfig, ClusteringConfig, TrackingConfig, ChurnConfig
from config.paths import EXPERIMENT_CONFIG_PATH


def test_load_production_config():
    """Verify that the production experiment.yaml loads and passes Pydantic validation."""
    config = load_config(EXPERIMENT_CONFIG_PATH)

    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.clustering, ClusteringConfig)
    assert isinstance(config.tracking, TrackingConfig)
    assert isinstance(config.churn, ChurnConfig)

    assert config.clustering.features == ["recency", "frequency", "avg_order_value"]
    assert config.clustering.cluster_range == [4]
    assert config.clustering.random_state == 42
    assert config.tracking.experiment_name == "customer_segmentation"

    assert config.churn.cutoff_days == 90
    assert config.churn.test_size == 0.2
    assert config.churn.random_state == 42
    assert config.churn.experiment_name == "customer_churn"
    assert "recency" in config.churn.features


def test_empty_yaml_uses_default_factories(tmp_path):
    """Verify that an empty or minimal YAML correctly populates all default_factories."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("{}")

    config = load_config(empty_yaml)

    assert isinstance(config.clustering, ClusteringConfig)
    assert isinstance(config.tracking, TrackingConfig)
    assert isinstance(config.churn, ChurnConfig)
    assert config.clustering.cluster_range == [4]
    assert config.churn.cutoff_days == 90
    assert config.tracking.experiment_name == "customer_segmentation"


def test_config_validation_rejects_invalid_types(tmp_path):
    """Verify that Pydantic enforces strict types and rejects malformed configs."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("""
clustering:
  features: "not-a-list"
  cluster_range: [4]
  random_state: "not-an-int"
churn:
  cutoff_days: "noventa"
tracking:
  experiment_name: "test"
""")

    with pytest.raises(ValidationError):
        load_config(invalid_yaml)



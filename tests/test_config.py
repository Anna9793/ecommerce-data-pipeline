import pytest
from pydantic import ValidationError
from config.loader import load_config
from config.schema import ExperimentConfig, ClusteringConfig, MLflowConfig
from config.paths import EXPERIMENT_CONFIG_PATH


def test_load_production_config():
    """Verify that the production experiment.yaml loads and passes Pydantic validation."""
    config = load_config(EXPERIMENT_CONFIG_PATH)

    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.clustering, ClusteringConfig)
    assert isinstance(config.mlflow, MLflowConfig)

    assert config.clustering.features == ["recency", "frequency", "avg_order_value"]
    assert config.clustering.cluster_range == [4]
    assert config.clustering.random_state == 42
    assert config.mlflow.experiment_name == "customer_segmentation"


def test_config_validation_rejects_invalid_types(tmp_path):
    """Verify that Pydantic enforces strict types and rejects malformed configs."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("""
clustering:
  features: "not-a-list"
  cluster_range: [4]
  random_state: "not-an-int"
mlflow:
  experiment_name: "test"
""")

    with pytest.raises(ValidationError):
        load_config(invalid_yaml)

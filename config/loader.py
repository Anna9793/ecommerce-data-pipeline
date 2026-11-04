import yaml
from pathlib import Path
from config.schema import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    """Loads and validates YAML configuration using Pydantic schema."""
    path = Path(path)
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return ExperimentConfig(**raw_config)

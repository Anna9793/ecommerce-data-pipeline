import yaml
from pathlib import Path
from src.utils.config_schema import ExperimentConfig

def load_config(path):

    path = Path(path)

    print("Loading config from:", path.resolve())

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    config = ExperimentConfig(**raw_config)

    return config
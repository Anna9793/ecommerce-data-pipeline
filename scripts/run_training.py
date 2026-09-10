import logging
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.config_loader import load_config
from config.paths import EXPERIMENT_CONFIG_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def main():
    config = load_config(EXPERIMENT_CONFIG_PATH)

    run_training_pipeline(config)

if __name__ == "__main__":
    main()

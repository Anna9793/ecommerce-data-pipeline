import os
import sys
import logging

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import load_config
from config.paths import EXPERIMENT_CONFIG_PATH
from src.pipelines.churn_pipeline import run_churn_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    config = load_config(EXPERIMENT_CONFIG_PATH)
    run_churn_pipeline(config.churn)


if __name__ == "__main__":
    main()

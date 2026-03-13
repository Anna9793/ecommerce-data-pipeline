import logging
import argparse
from pathlib import Path
from src.utils.config_loader import load_config
from src.cleaning import run_cleaning
from src.transformation import run_transformation
from src.rfm_features import run_rfm_features
from src.clustering import run_clustering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Data pipeline controller"
    )

    parser.add_argument(
        "command",
        choices=["clean", "transform", "rfm", "cluster", "experiments", "all"],
        help="Pipeline stage to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration file"
    )

    return parser.parse_args()

def run_clustering_experiments():

    experiments_dir = Path("config/experiments")

    for config_file in experiments_dir.glob("*.yaml"):

        logging.info(
            "Running clustering experiment: %s",
            config_file.name
        )

        config = load_config(config_file)

        run_clustering(config, config_file.name)

def main():

    args = parse_args()

    logging.info("Starting pipeline stage: %s", args.command)

    if args.command == "clean":
        run_cleaning()

    elif args.command == "transform":
        run_transformation()

    elif args.command == "rfm":
        run_rfm_features()

    elif args.command == "cluster":

        config_path = args.config or "config/experiment.yaml"

        logging.info("Loading config from: %s", config_path)

        config = load_config(config_path)

        run_clustering(config, config_path)

    elif args.command == "experiments":
        run_clustering_experiments()
        

    elif args.command == "all":
        run_cleaning()
        run_transformation()
        run_rfm_features()
        run_clustering_experiments

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

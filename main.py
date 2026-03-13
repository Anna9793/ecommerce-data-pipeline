from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.config_loader import load_config

config = load_config("config/experiment.yaml")
def main():

    run_training_pipeline(config)

if __name__ == "__main__":
    main()

import logging
import argparse

from src.pipelines.prediction_pipeline import run_prediction_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run prediction pipeline"
    )

    parser.add_argument("--input", required=True)

    parser.add_argument(
        "--output",
        default="data/predictions/customer_clusters.csv"
    )

    args = parser.parse_args()

    predictions = run_prediction_pipeline(args.input)

    logging.info("Saving predictions to %s", args.output)

    predictions.to_csv(args.output, index = False)



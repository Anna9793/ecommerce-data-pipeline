import os
import logging
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.bronze import run_bronze
from src.silver import run_silver
from src.gold import run_gold, load_gold
from src.clustering import run_clustering


def run_training_pipeline(config):
    """
    Executes the end-to-end customer segmentation training pipeline:
    1. Medallion ETL:
       - Bronze: Ingests raw source into immutable local landing storage.
       - Silver: Cleans, types, and enriches transactions in a single pass.
       - Gold: Computes customer RFM and velocity features (0 NaNs guaranteed).
    2. Runs clustering model training and evaluation.
    3. Logs metrics, parameters, artifacts, and registers model in MLflow.
    4. Persists artifacts locally and exports to BigQuery (if USE_BIGQUERY=true).
    """
    logging.info("Starting Medallion training pipeline")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(config.mlflow.experiment_name)

    # 1. Medallion Feature Engineering (Local deterministic execution)
    run_bronze()
    run_silver()
    run_gold()

    df = load_gold(use_bigquery=False)
    feature_columns = config.clustering.features

    with mlflow.start_run():
        logging.info("Executing clustering training run in MLflow")

        model, best_k, metrics, fig, pca_fig = run_clustering(
            config,
            df=df,
            feature_columns=feature_columns,
            cluster_range=config.clustering.cluster_range
        )

        input_example = df[feature_columns].head(5)
        predictions = model.predict(input_example)
        signature = infer_signature(input_example, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        mlflow.log_metric("silhouette_score", metrics["silhouette"])
        mlflow.log_figure(fig, "silhouette_plot.png")
        if pca_fig is not None:
            mlflow.log_figure(pca_fig, "pca_clusters.png")

        mlflow.log_param("feature_set", ",".join(feature_columns))
        mlflow.log_param("n_clusters", best_k)

        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="customer_segmentation_model"
        )

    logging.info("Training pipeline completed successfully")
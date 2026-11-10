import os
import logging

from src.bronze import run_bronze
from src.silver import run_silver
from src.gold import run_gold, load_gold
from src.clustering import run_clustering
from src.utils.tracker import VertexExperimentTracker
from config.paths import SEGMENTATION_MODEL_PKL


def run_training_pipeline(config):
    """
    Executes the end-to-end customer segmentation training pipeline:
    1. Medallion ETL:
       - Bronze: Ingests raw source into immutable local landing storage.
       - Silver: Cleans, types, and enriches transactions in a single pass.
       - Gold: Computes customer RFM and velocity features (0 NaNs guaranteed).
    2. Runs clustering model training and evaluation.
    3. Logs metrics, parameters, artifacts, and registers model in Vertex AI Experiments & Model Registry.
    4. Persists artifacts locally and exports to BigQuery (if USE_BIGQUERY=true).
    """
    logging.info("Starting Medallion training pipeline")

    tracking_cfg = config.get_tracking() if hasattr(config, "get_tracking") else (config.tracking or config.mlflow)
    experiment_name = tracking_cfg.experiment_name

    tracker = VertexExperimentTracker(experiment_name=experiment_name)

    # 1. Medallion Feature Engineering (Local deterministic execution)
    run_bronze()
    run_silver()
    run_gold()

    df = load_gold(use_bigquery=False)
    feature_columns = config.clustering.features

    with tracker.start_run(run_name="customer_segmentation_kmeans"):
        logging.info("Executing clustering training run in Vertex AI Experiments")

        model, best_k, metrics, fig, pca_fig = run_clustering(
            config,
            df=df,
            feature_columns=feature_columns,
            cluster_range=config.clustering.cluster_range
        )

        tracker.log_metrics({"silhouette_score": float(metrics["silhouette"])})
        tracker.log_params({
            "feature_set": ",".join(feature_columns),
            "n_clusters": int(best_k)
        })

        tracker.log_figure(fig, "silhouette_plot.png")
        if pca_fig is not None:
            tracker.log_figure(pca_fig, "pca_clusters.png")

        tracker.save_and_register_model(
            model=model,
            model_name="customer_segmentation_model",
            local_path=SEGMENTATION_MODEL_PKL
        )

    logging.info("Training pipeline completed successfully")
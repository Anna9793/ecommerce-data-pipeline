import os
import logging
from typing import Optional, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from config.paths import (
    SEGMENTATION_MODEL_PKL,
    EXPERIMENT_CONFIG_PATH,
    CUSTOMER_CLUSTERS,
    CLUSTER_PROFILE,
)
from config.loader import load_config
from config.schema import ClusteringConfig
from src.bronze import run_bronze
from src.silver import run_silver
from src.gold import run_gold, load_gold
from src.clustering import run_clustering
from src.utils.tracker import VertexExperimentTracker


def save_clustering_outputs(profile: pd.DataFrame, scored_df: Optional[pd.DataFrame] = None):
    """Persists cluster profile and scored customers to local CSV and BigQuery."""
    logging.info("Saving cluster profile to %s", CLUSTER_PROFILE)
    profile.to_csv(CLUSTER_PROFILE)

    if scored_df is not None:
        logging.info("Saving batch customer segmentation to %s", CUSTOMER_CLUSTERS)
        scored_df.to_csv(CUSTOMER_CLUSTERS, index=False)

        if os.getenv("USE_BIGQUERY", "false").lower() == "true":
            project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
            destination_table = "retail_data.customer_segments"
            logging.info("Exporting %d scored customers to BigQuery: %s", len(scored_df), destination_table)
            try:
                import pandas_gbq
                bq_df = scored_df.copy()
                if "customer_id" in bq_df.columns:
                    bq_df["customer_id"] = bq_df["customer_id"].astype(str)
                pandas_gbq.to_gbq(
                    bq_df,
                    destination_table=destination_table,
                    project_id=project_id,
                    if_exists="replace"
                )
                logging.info("Successfully exported batch customer segments to BigQuery table %s", destination_table)
            except Exception as e:
                logging.error("Failed to export customer segments to BigQuery: %s", e)
                raise


save_outputs = save_clustering_outputs  # Backward compatibility alias


def run_clustering_pipeline(
    config=None,
    df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[list] = None,
    cluster_range: Optional[list] = None,
    random_state: Optional[int] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Executes the end-to-end Medallion customer segmentation training and batch scoring pipeline."""
    logging.info("Starting Medallion clustering training pipeline")

    if config is None:
        try:
            full_config = load_config(EXPERIMENT_CONFIG_PATH)
            clustering_cfg = full_config.clustering
            experiment_name = full_config.tracking.experiment_name
        except Exception as e:
            logging.warning("Could not load experiment config (%s), falling back to defaults.", e)
            clustering_cfg = ClusteringConfig()
            experiment_name = "customer_segmentation"
    elif hasattr(config, "clustering"):
        clustering_cfg = config.clustering
        experiment_name = getattr(config.tracking, "experiment_name", "customer_segmentation")
    elif isinstance(config, ClusteringConfig):
        clustering_cfg = config
        experiment_name = "customer_segmentation"
    else:
        clustering_cfg = ClusteringConfig()
        experiment_name = "customer_segmentation"

    effective_features = feature_cols if feature_cols is not None else clustering_cfg.features
    effective_range = cluster_range if cluster_range is not None else clustering_cfg.cluster_range
    effective_random_state = random_state if random_state is not None else clustering_cfg.random_state

    tracker = VertexExperimentTracker(experiment_name=experiment_name)

    # 1. Medallion Feature Engineering (Local deterministic execution)
    if df is None:
        run_bronze()
        run_silver()
        run_gold()
        df = load_gold(use_bigquery=False)

    df_clean = df.dropna(subset=effective_features).copy()

    # 2. Train / Test Split (for unbiased hyperparameter selection)
    train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=effective_random_state)
    train_df = train_df.copy()

    # 3. Model Training, Evaluation, and Vertex AI Tracking
    with tracker.start_run(run_name="customer_segmentation_kmeans"):
        logging.info("Executing clustering training run in Vertex AI Experiments")

        model, best_k, metrics, profile, figures = run_clustering(
            train_df=train_df,
            feature_columns=effective_features,
            cluster_range=effective_range,
            random_state=effective_random_state
        )

        tracker.log_metrics(metrics)
        tracker.log_params({
            "feature_set": ",".join(effective_features),
            "n_clusters": int(best_k),
            "random_state": effective_random_state,
            "test_size": test_size
        })

        for fig_name, fig in figures.items():
            tracker.log_figure(fig, fig_name)

        tracker.save_and_register_model(
            model=model,
            model_name="customer_segmentation_model",
            local_path=SEGMENTATION_MODEL_PKL
        )

    # 4. Batch Scoring & Persistence (Local CSV + BigQuery)
    scored_customers = df_clean.copy()
    scored_customers["cluster"] = model.predict(scored_customers[effective_features])
    scored_customers["segment"] = scored_customers["cluster"].map(profile["segment"])
    save_clustering_outputs(profile, scored_df=scored_customers)

    logging.info("Clustering pipeline completed successfully")
    return {
        "status": "registered",
        "model_name": "customer_segmentation_model",
        "best_k": int(best_k),
        "silhouette_score": float(metrics["silhouette_score"])
    }


# Backward compatibility alias
run_training_pipeline = run_clustering_pipeline
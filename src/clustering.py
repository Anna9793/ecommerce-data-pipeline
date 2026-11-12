import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from dotenv import load_dotenv
from src.gold import load_gold
from src.utils.plotting import plot_silhouette_scores, plot_pca_clusters

from config.paths import (
    GOLD_RFM,
    CUSTOMER_CLUSTERS,
    CLUSTER_PROFILE,
)

load_dotenv()
load_rfm = load_gold  # Backward compatibility alias


def build_pipeline(k, random_state, feature_columns):
    """
    Builds a native Scikit-Learn pipeline with ColumnTransformer:
    - Applies np.log1p to heavily skewed features (frequency, avg_order_value).
    - Passes through remaining features (recency).
    - Applies StandardScaler globally.
    - Fits KMeans with k clusters.
    """
    logging.info("Building KMeans pipeline (k=%d)", k)
    log_features = [col for col in ["frequency", "avg_order_value"] if col in feature_columns]
    other_features = [col for col in feature_columns if col not in log_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("log", FunctionTransformer(np.log1p), log_features),
            ("pass", "passthrough", other_features)
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=k, random_state=random_state))
    ])


def find_best_kmeans(train_df, feature_columns, cluster_range, random_state):
    """
    Evaluates KMeans over a range of k using Silhouette Score to find the optimal model.

    Returns:
        tuple: (best_pipeline, best_k, best_score, scores)
    """
    logging.info("Searching optimal k in range %s", cluster_range)
    best_score = -1.0
    best_k = None
    best_pipeline = None
    scores = []

    for k in cluster_range:
        pipeline = build_pipeline(k, random_state, feature_columns)
        clusters = pipeline.fit_predict(train_df)

        # Silhouette is evaluated on transformed/scaled features
        X_transformed = pipeline[:-1].transform(train_df)
        score = silhouette_score(X_transformed, clusters)

        scores.append((k, score))
        logging.info("k=%d → silhouette score=%.4f", k, score)

        if score > best_score:
            best_score = score
            best_k = k
            best_pipeline = pipeline

    logging.info("Best model selected: k=%d (silhouette=%.4f)", best_k, best_score)
    return best_pipeline, best_k, best_score, scores


def profile_clusters(df):
    """
    Computes RFM summary statistics per cluster and assigns business segment labels.

    Returns:
        tuple: (df_with_segment, profile_summary_df)
    """
    logging.info("Profiling clusters and assigning business segment labels")
    profile = (
        df.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_order_value=("avg_order_value", "mean"),
        )
        .round(2)
    )

    def _assign_segment(row):
        if row["avg_recency"] > 150:
            return "Inactive Customers"
        if row["avg_frequency"] > 10:
            return "Frequent Buyers"
        if row["avg_order_value"] > 500:
            return "Occasional High Value Buyers"
        return "Medium Customers"

    profile["segment"] = profile.apply(_assign_segment, axis=1)

    df_labeled = df.copy()
    df_labeled["segment"] = df_labeled["cluster"].map(profile["segment"])

    return df_labeled, profile


def save_outputs(profile, scored_df=None):
    """
    Saves clustering artifacts:
    - Local: CLUSTER_PROFILE and CUSTOMER_CLUSTERS (CSV)
    - Cloud (if USE_BIGQUERY=true): Uploads batch customer segmentation to BigQuery table
    """
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


def run_clustering(
    train_df: pd.DataFrame,
    feature_columns: list,
    cluster_range: list,
    random_state: int
):
    """
    Core clustering modeling workflow:
    1. Evaluates candidate k values on train_df using Silhouette Score.
    2. Profiles clusters and assigns business segment labels.
    3. Generates Silhouette curve and PCA 2D projection figures.

    Returns:
        tuple: (best_pipeline, best_k, metrics_dict, profile_df, figures_dict)
    """
    logging.info("Starting KMeans clustering model training")

    # 1. Hyperparameter optimization
    best_pipeline, best_k, best_score, scores = find_best_kmeans(
        train_df, feature_columns, cluster_range, random_state
    )

    # 2. Profile & Business segments (derived from train set)
    train_df_scored = train_df.copy()
    train_df_scored["cluster"] = best_pipeline.predict(train_df_scored[feature_columns])
    train_df_scored, profile = profile_clusters(train_df_scored)
    logging.info("Cluster profile:\n%s", profile)

    # 3. Visualizations
    fig = plot_silhouette_scores(scores)
    pca_fig = plot_pca_clusters(train_df_scored, best_pipeline, feature_columns)

    figures = {
        "silhouette_plot.png": fig
    }
    if pca_fig is not None:
        figures["pca_clusters.png"] = pca_fig

    metrics = {
        "silhouette_score": float(best_score)
    }

    logging.info("KMeans clustering model training completed successfully")
    return best_pipeline, best_k, metrics, profile, figures

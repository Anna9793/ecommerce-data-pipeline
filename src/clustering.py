import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from src.utils.plotting import plot_silhouette_scores, plot_pca_clusters


def build_pipeline(k, random_state, feature_columns):
    """Builds a Scikit-Learn KMeans pipeline with log transform and scaling."""
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
    """Evaluates KMeans over cluster_range using Silhouette score to select best model."""
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
    """Computes RFM summary statistics per cluster and assigns business segment labels."""
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


def run_clustering(
    train_df: pd.DataFrame,
    feature_columns: list,
    cluster_range: list,
    random_state: int
):
    """Trains KMeans clustering across candidate k, profiles clusters, and generates plots."""
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

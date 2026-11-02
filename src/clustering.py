import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from src.rfm_features import load_rfm
from src.utils.plotting import plot_silhouette_scores, plot_pca_clusters

from config.paths import (
    RFM_CUSTOMERS,
    TRAIN_CLUSTERS,
    CUSTOMER_CLUSTERS,
    CLUSTER_PROFILE,
)


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


def save_outputs(train_df, test_df, profile):
    """Saves training clusters, predicted test clusters, and cluster profile to CSV."""
    logging.info("Saving cluster artifacts to processed directory")
    train_df.to_csv(TRAIN_CLUSTERS, index=False)
    test_df.to_csv(CUSTOMER_CLUSTERS, index=False)
    profile.to_csv(CLUSTER_PROFILE)


def run_clustering(config, df=None, feature_columns=None, cluster_range=None):
    """
    End-to-end clustering workflow:
    1. Splits RFM data into train/test sets.
    2. Identifies optimal k using Silhouette Score.
    3. Assigns clusters and generates business profiles.
    4. Generates Silhouette and PCA visualization figures.
    5. Saves outputs to disk.
    """
    logging.info("Starting clustering pipeline")

    if cluster_range is None:
        cluster_range = config.clustering.cluster_range
    random_state = config.clustering.random_state

    if df is None:
        df = load_rfm(RFM_CUSTOMERS)

    # 1. Train / Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 2. Hyperparameter optimization
    best_pipeline, best_k, best_score, scores = find_best_kmeans(
        train_df, feature_columns, cluster_range, random_state
    )

    # 3. Predict clusters
    train_df["cluster"] = best_pipeline.predict(train_df[feature_columns])
    test_df["cluster"] = best_pipeline.predict(test_df[feature_columns])

    # 4. Profile & Business segments
    train_df, profile = profile_clusters(train_df)
    test_df["segment"] = test_df["cluster"].map(profile["segment"])
    logging.info("Cluster profile:\n%s", profile)

    # 5. Visualizations
    fig = plot_silhouette_scores(scores)
    pca_fig = plot_pca_clusters(train_df, best_pipeline, feature_columns)

    # 6. Save artifacts
    save_outputs(train_df, test_df, profile)

    logging.info("Clustering pipeline completed successfully")
    return best_pipeline, best_k, {"silhouette": best_score}, fig, pca_fig


# Aliases for backward compatibility
build_cluster_profile = profile_clusters
visualize_clusters = plot_pca_clusters

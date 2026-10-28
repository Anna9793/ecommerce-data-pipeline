import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.clustering import (
    build_pipeline,
    split_dataset,
    find_best_kmeans,
    assign_clusters,
    profile_clusters,
    label_clusters,
    build_cluster_profile,
    visualize_clusters
)


@pytest.fixture
def sample_rfm_df():
    """Synthetic customer RFM dataset for testing clustering logic."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(n)],
        "recency": np.concatenate([
            np.random.uniform(5, 30, size=20),    # Recent
            np.random.uniform(40, 90, size=20),   # Moderate
            np.random.uniform(180, 300, size=20)  # Inactive
        ]),
        "frequency": np.concatenate([
            np.random.uniform(15, 30, size=20),   # Frequent
            np.random.uniform(2, 6, size=20),     # Moderate
            np.random.uniform(1, 2, size=20)      # Rare
        ]),
        "avg_order_value": np.concatenate([
            np.random.uniform(200, 600, size=20),
            np.random.uniform(50, 150, size=20),
            np.random.uniform(20, 80, size=20)
        ])
    })


def test_split_dataset(sample_rfm_df):
    train_df, test_df = split_dataset(sample_rfm_df, random_state=42)
    assert len(train_df) == 48  # 80% of 60
    assert len(test_df) == 12   # 20% of 60
    assert set(train_df["customer_id"]).isdisjoint(set(test_df["customer_id"]))


def test_build_pipeline_structure():
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps

    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], KMeans)
    assert pipeline.named_steps["model"].n_clusters == 3


def test_build_pipeline_fit_predict(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)

    predictions = pipeline.fit_predict(sample_rfm_df)
    assert len(predictions) == len(sample_rfm_df)
    assert set(predictions).issubset({0, 1, 2})

    # Verify preprocessing outputs correct shape and finite values
    X_transformed = pipeline[:-1].transform(sample_rfm_df)
    assert X_transformed.shape == (len(sample_rfm_df), len(features))
    assert np.all(np.isfinite(X_transformed))


def test_find_best_kmeans(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    best_pipeline, best_k, best_score, scores = find_best_kmeans(
        sample_rfm_df,
        feature_columns=features,
        cluster_range=[2, 3],
        random_state=42
    )

    assert best_k in [2, 3]
    assert -1.0 <= best_score <= 1.0
    assert len(scores) == 2
    assert isinstance(best_pipeline, Pipeline)


def test_assign_clusters(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)
    pipeline.fit(sample_rfm_df)

    clustered_df = assign_clusters(sample_rfm_df, pipeline, features)
    assert "cluster" in clustered_df.columns
    assert clustered_df["cluster"].nunique() <= 3
    assert len(clustered_df) == len(sample_rfm_df)


def test_profile_and_label_clusters():
    # Synthetic clustered data with distinct cluster characteristics
    df = pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(4)],
        "cluster": [0, 1, 2, 3],
        "recency": [200.0, 10.0, 30.0, 50.0],
        "frequency": [1.0, 15.0, 3.0, 4.0],
        "avg_order_value": [30.0, 100.0, 600.0, 100.0]
    })

    profile = profile_clusters(df)
    assert "customers" in profile.columns
    assert "avg_recency" in profile.columns
    assert len(profile) == 4

    labeled_profile = label_clusters(profile)
    assert "segment" in labeled_profile.columns

    # Check rule-based mapping:
    # cluster 0: recency 200 > 150 -> Inactive Customers
    assert labeled_profile.loc[0, "segment"] == "Inactive Customers"
    # cluster 1: frequency 15 > 10 -> Frequent Buyers
    assert labeled_profile.loc[1, "segment"] == "Frequent Buyers"
    # cluster 2: avg_order_value 600 > 500 -> Occasional High Value Buyers
    assert labeled_profile.loc[2, "segment"] == "Occasional High Value Buyers"
    # cluster 3: default -> Medium Customers
    assert labeled_profile.loc[3, "segment"] == "Medium Customers"


def test_build_cluster_profile_end_to_end(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)
    sample_rfm_df["cluster"] = pipeline.fit_predict(sample_rfm_df)

    enriched_df, profile = build_cluster_profile(sample_rfm_df)
    assert "segment" in enriched_df.columns
    assert "segment" in profile.columns
    assert len(enriched_df) == len(sample_rfm_df)
    assert not enriched_df["segment"].isna().any()


def test_visualize_clusters(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)
    sample_rfm_df["cluster"] = pipeline.fit_predict(sample_rfm_df)

    fig = visualize_clusters(sample_rfm_df, pipeline, features)
    assert fig is not None

    # Test with < 2 features (skips PCA)
    fig_single = visualize_clusters(sample_rfm_df, pipeline, ["recency"])
    assert fig_single is None

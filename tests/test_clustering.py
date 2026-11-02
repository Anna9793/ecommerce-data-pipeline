import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.clustering import (
    build_pipeline,
    find_best_kmeans,
    profile_clusters,
    save_outputs,
)
from src.utils.plotting import (
    plot_silhouette_scores,
    plot_pca_clusters,
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


def test_profile_clusters_and_business_labels():
    # Synthetic clustered data with distinct cluster characteristics
    df = pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(4)],
        "cluster": [0, 1, 2, 3],
        "recency": [200.0, 10.0, 30.0, 50.0],
        "frequency": [1.0, 15.0, 3.0, 4.0],
        "avg_order_value": [30.0, 100.0, 600.0, 100.0]
    })

    df_labeled, profile = profile_clusters(df)

    assert "customers" in profile.columns
    assert "avg_recency" in profile.columns
    assert "segment" in profile.columns
    assert len(profile) == 4

    # Check rule-based segment mappings:
    # cluster 0: recency 200 > 150 -> Inactive Customers
    assert profile.loc[0, "segment"] == "Inactive Customers"
    # cluster 1: frequency 15 > 10 -> Frequent Buyers
    assert profile.loc[1, "segment"] == "Frequent Buyers"
    # cluster 2: avg_order_value 600 > 500 -> Occasional High Value Buyers
    assert profile.loc[2, "segment"] == "Occasional High Value Buyers"
    # cluster 3: default -> Medium Customers
    assert profile.loc[3, "segment"] == "Medium Customers"

    # Verify customer dataframe contains segment column
    assert "segment" in df_labeled.columns
    assert not df_labeled["segment"].isna().any()
    assert df_labeled.loc[0, "segment"] == "Inactive Customers"


def test_plot_silhouette_scores():
    scores = [(2, 0.25), (3, 0.35), (4, 0.30)]
    fig = plot_silhouette_scores(scores)
    assert fig is not None


def test_plot_pca_clusters(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    pipeline = build_pipeline(k=3, random_state=42, feature_columns=features)
    sample_rfm_df["cluster"] = pipeline.fit_predict(sample_rfm_df)

    # Valid with >= 2 features
    fig = plot_pca_clusters(sample_rfm_df, pipeline, features)
    assert fig is not None

    # Skips when fewer than 2 features
    fig_single = plot_pca_clusters(sample_rfm_df, pipeline, ["recency"])
    assert fig_single is None


def test_save_outputs(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    profile_path = tmp_path / "profile.csv"

    monkeypatch.setattr("src.clustering.TRAIN_CLUSTERS", train_path)
    monkeypatch.setattr("src.clustering.CUSTOMER_CLUSTERS", test_path)
    monkeypatch.setattr("src.clustering.CLUSTER_PROFILE", profile_path)

    df_train = pd.DataFrame({"customer_id": ["C1"], "cluster": [0]})
    df_test = pd.DataFrame({"customer_id": ["C2"], "cluster": [0]})
    df_profile = pd.DataFrame({"customers": [1]}, index=[0])

    save_outputs(df_train, df_test, df_profile)

    assert train_path.exists()
    assert test_path.exists()
    assert profile_path.exists()

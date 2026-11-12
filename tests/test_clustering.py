import pytest
from unittest.mock import patch
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
    run_clustering,
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


def test_save_outputs_local_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_BIGQUERY", "false")
    profile_path = tmp_path / "profile.csv"
    customers_path = tmp_path / "customers.csv"

    monkeypatch.setattr("src.clustering.CLUSTER_PROFILE", profile_path)
    monkeypatch.setattr("src.clustering.CUSTOMER_CLUSTERS", customers_path)

    df_profile = pd.DataFrame({"customers": [1]}, index=[0])
    df_scored = pd.DataFrame({"customer_id": ["C1"], "cluster": [0], "segment": ["Medium Customers"]})

    with patch("pandas_gbq.to_gbq") as mock_to_gbq:
        save_outputs(df_profile, scored_df=df_scored)
        assert not mock_to_gbq.called

    assert profile_path.exists()
    assert customers_path.exists()


def test_save_outputs_bigquery_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_BIGQUERY", "true")
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    profile_path = tmp_path / "profile.csv"
    customers_path = tmp_path / "customers.csv"

    monkeypatch.setattr("src.clustering.CLUSTER_PROFILE", profile_path)
    monkeypatch.setattr("src.clustering.CUSTOMER_CLUSTERS", customers_path)

    df_profile = pd.DataFrame({"customers": [1]}, index=[0])
    df_scored = pd.DataFrame({"customer_id": ["C1"], "cluster": [0], "segment": ["Medium Customers"]})

    with patch("pandas_gbq.to_gbq") as mock_to_gbq:
        save_outputs(df_profile, scored_df=df_scored)
        assert mock_to_gbq.called
        call_kwargs = mock_to_gbq.call_args[1]
        assert call_kwargs["destination_table"] == "retail_data.customer_segments"
        assert call_kwargs["project_id"] == "test-project"
        assert call_kwargs["if_exists"] == "replace"

    assert profile_path.exists()
    assert customers_path.exists()


@patch("src.pipelines.clustering_pipeline.save_outputs")
@patch("src.pipelines.clustering_pipeline.VertexExperimentTracker")
@patch("src.pipelines.clustering_pipeline.run_clustering")
@patch("src.pipelines.clustering_pipeline.load_gold")
@patch("src.pipelines.clustering_pipeline.run_gold")
@patch("src.pipelines.clustering_pipeline.run_silver")
@patch("src.pipelines.clustering_pipeline.run_bronze")
def test_run_clustering_pipeline_orchestration(
    mock_bronze,
    mock_silver,
    mock_gold,
    mock_load_gold,
    mock_run_clustering,
    mock_tracker_cls,
    mock_save_outputs,
    sample_rfm_df
):
    from unittest.mock import MagicMock
    from config.schema import ExperimentConfig
    from src.pipelines.clustering_pipeline import run_clustering_pipeline

    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker
    mock_load_gold.return_value = sample_rfm_df

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(len(sample_rfm_df), dtype=int)
    mock_profile = pd.DataFrame({"segment": ["Medium Customers"]}, index=[0])
    mock_run_clustering.return_value = (mock_model, 4, {"silhouette_score": 0.45}, mock_profile, {})

    cfg = ExperimentConfig()
    result = run_clustering_pipeline(cfg)

    assert result["status"] == "registered"
    assert result["model_name"] == "customer_segmentation_model"
    assert result["best_k"] == 4
    assert result["silhouette_score"] == 0.45
    assert mock_bronze.called
    assert mock_silver.called
    assert mock_gold.called
    assert mock_run_clustering.called
    assert mock_tracker.save_and_register_model.called
    assert mock_save_outputs.called


def test_run_clustering_direct(sample_rfm_df):
    features = ["recency", "frequency", "avg_order_value"]
    cluster_range = [3, 4]
    random_state = 42

    pipeline, best_k, metrics, profile, figures = run_clustering(
        train_df=sample_rfm_df,
        feature_columns=features,
        cluster_range=cluster_range,
        random_state=random_state
    )

    assert pipeline is not None
    assert best_k in cluster_range
    assert "silhouette_score" in metrics
    assert isinstance(metrics["silhouette_score"], float)
    assert "segment" in profile.columns
    assert "silhouette_plot.png" in figures
    assert "pca_clusters.png" in figures


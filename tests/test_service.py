import pandas as pd 
from unittest.mock import patch

from app.service import predict_clustering_service

def test_predict_cluster_returns_cluster_and_label():

    features = {
        "recency": 30,
        "frequency": 5,
        "avg_order_value": 100
    }

    cluster, label = predict_clustering_service(features)

    assert isinstance(cluster, int)
    assert isinstance(label, str)
    assert len(label) > 0

@patch("app.service.prod_clustering_pipeline")
def test_predict_cluster_returns_unknown_for_unmapped_cluster(
    mock_pipeline
):

    mock_pipeline.predict.return_value = [999]
    mock_pipeline.cluster_labels_ = {}

    features = {
        "recency": 30,
        "frequency": 5,
        "avg_order_value": 100
    }

    cluster, label = predict_clustering_service(features)

    assert cluster == 999
    assert label == "Unknown"


@patch("app.service.prod_clustering_pipeline")
def test_predict_cluster_uses_pipeline_cluster_labels(mock_pipeline):
    mock_pipeline.predict.return_value = [1]
    mock_pipeline.cluster_labels_ = {1: "VIP Champions"}

    features = {
        "recency": 10,
        "frequency": 20,
        "avg_order_value": 500
    }
    cluster, label = predict_clustering_service(features)

    assert cluster == 1
    assert label == "VIP Champions"


@patch("joblib.load")
def test_load_all_models_enforces_cluster_labels_contract(mock_joblib_load):
    import pytest
    from unittest.mock import MagicMock
    from app.service import _load_all_models

    mock_bad_model = MagicMock(spec=["predict"])
    mock_joblib_load.return_value = mock_bad_model

    with pytest.raises(ValueError, match="Model Contract Violation"):
        _load_all_models()
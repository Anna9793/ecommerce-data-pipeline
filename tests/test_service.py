import pandas as pd 
from unittest.mock import patch

from app.service import predict_cluster

def test_predict_cluster_returns_cluster_and_label():

    features = {
        "recency": 30,
        "frequency": 5,
        "avg_order_value": 100
    }

    cluster, label = predict_cluster(features)

    assert isinstance(cluster,int)

    assert isinstance(label, str)

    assert len(label) > 0

@patch("app.service.prod_pipeline")
def test_predict_cluster_returns_unknown_for_unmapped_cluster(
    mock_pipeline
):

    mock_pipeline.predict.return_value = [999]

    features = {
        "recency": 30,
        "frequency": 5,
        "avg_order_value": 100
    }

    cluster, label = predict_cluster(features)

    assert cluster == 999

    assert label == "Unknown"
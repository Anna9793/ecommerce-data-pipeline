from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

@patch("app.main.insert_churn_prediction")
@patch("app.main.predict_churn_service")
def test_predict_churn_endpoint_returns_prediction(
    mock_predict,
    mock_insert):
    
    mock_predict.return_value = (1, 0.85)

    response = client.post(
        "/predict/churn",
        json={
            "recency": 90,
            "frequency": 2,
            "avg_order_value": 50,
            "spending_velocity": 1.0,
            "cancellation_rate": 0.0,
            "preferred_shopping_hour": 12
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["is_churn"] == 1
    assert data["churn_probability"] == 0.85
    assert mock_insert.called

def test_predict_churn_reject_invalid_input():

    response = client.post(
        "/predict/churn",
        json={
            "recency": "not-a-number",
            "frequency": 2,
            "avg_order_value": 50
        }
    )

    assert response.status_code == 422

@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_trigger_churn_retraining_endpoint(mock_submit):
    response = client.post("/train/churn")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@patch("src.churn_training.VertexExperimentTracker")
@patch("src.churn_training.create_churn_target")
@patch("src.churn_training.compute_gold_features")
@patch("src.churn_training.train_test_split")
@patch("src.churn_training.Pipeline")
@patch("src.churn_training.classification_report")
def test_run_churn_training_respects_custom_config(
    mock_report,
    mock_pipe_cls,
    mock_split,
    mock_rfm,
    mock_target,
    mock_tracker_cls
):
    import pandas as pd
    import numpy as np
    from unittest.mock import MagicMock
    from config.schema import ChurnConfig
    from src.churn_training import run_churn_training

    mock_tracker = MagicMock()
    mock_tracker.start_run.return_value.__enter__.return_value = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    custom_cfg = ChurnConfig(
        cutoff_days=45,
        test_size=0.3,
        random_state=123,
        experiment_name="custom-churn-test"
    )

    dates = pd.date_range("2011-01-01", "2011-12-01", freq="D")
    df_dummy = pd.DataFrame({
        "invoice_date": dates,
        "customer_id": [1] * len(dates),
        "price": [10.0] * len(dates),
        "quantity": [1] * len(dates)
    })

    rfm_dummy = pd.DataFrame([{
        "customer_id": 1,
        "recency": 10,
        "frequency": 5,
        "avg_order_value": 50.0,
        "spending_velocity": 1.0,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 14
    }])
    mock_rfm.return_value = rfm_dummy
    mock_target.return_value = pd.DataFrame([{"customer_id": 1, "churn": 0}])

    X_dummy = rfm_dummy[custom_cfg.features]
    y_dummy = pd.Series([0])
    mock_split.return_value = (X_dummy, X_dummy, y_dummy, y_dummy)

    mock_pipe_inst = MagicMock()
    mock_pipe_inst.predict.return_value = np.array([0])
    mock_pipe_inst.named_steps = {"model": MagicMock()}
    mock_pipe_cls.return_value = mock_pipe_inst

    mock_report.return_value = {
        "accuracy": 1.0,
        "1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}
    }

    result = run_churn_training(config=custom_cfg, df=df_dummy)

    assert result["status"] == "registered"
    assert result["model_name"] == "customer_churn_model"
    mock_tracker_cls.assert_called_with(experiment_name="custom-churn-test")


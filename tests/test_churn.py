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


@patch("src.pipelines.churn_pipeline.VertexExperimentTracker")
@patch("src.pipelines.churn_pipeline.prepare_churn_data")
@patch("src.pipelines.churn_pipeline.train_and_evaluate_churn_models")
def test_run_churn_training_respects_custom_config(
    mock_train_models,
    mock_prepare_data,
    mock_tracker_cls
):
    import pandas as pd
    from unittest.mock import MagicMock
    from config.schema import ChurnConfig
    from src.churn_training import run_churn_training

    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    custom_cfg = ChurnConfig(
        cutoff_days=45,
        test_size=0.3,
        random_state=123,
        experiment_name="custom-churn-test"
    )

    X_dummy = pd.DataFrame({"recency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y_dummy = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    training_dummy = pd.DataFrame({"churn": y_dummy})
    mock_prepare_data.return_value = (X_dummy, y_dummy, training_dummy)

    mock_pipeline = MagicMock()
    mock_train_models.return_value = (mock_pipeline, 0.85, "customer_churn_model")

    result = run_churn_training(config=custom_cfg)

    assert result["status"] == "registered"
    assert result["model_name"] == "customer_churn_model"
    assert result["f1_score"] == 0.85
    mock_tracker_cls.assert_called_with(experiment_name="custom-churn-test")
    assert mock_tracker.save_and_register_model.called


@patch("src.pipelines.churn_pipeline.VertexExperimentTracker")
@patch("src.pipelines.churn_pipeline.prepare_churn_data")
@patch("src.pipelines.churn_pipeline.train_and_evaluate_churn_models")
def test_run_churn_pipeline_direct(
    mock_train_models,
    mock_prepare_data,
    mock_tracker_cls
):
    import pandas as pd
    from unittest.mock import MagicMock
    from src.pipelines.churn_pipeline import run_churn_pipeline

    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    X_dummy = pd.DataFrame({"recency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y_dummy = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    training_dummy = pd.DataFrame({"churn": y_dummy})
    mock_prepare_data.return_value = (X_dummy, y_dummy, training_dummy)

    mock_pipeline = MagicMock()
    mock_train_models.return_value = (mock_pipeline, 0.90, "customer_churn_model")

    result = run_churn_pipeline()

    assert result["status"] == "registered"
    assert result["f1_score"] == 0.90
    assert mock_tracker.save_and_register_model.called



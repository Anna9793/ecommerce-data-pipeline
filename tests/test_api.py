import os
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "status":"healthy"
    }

@patch("app.main.insert_prediction")
@patch("app.main.predict_cluster")
def test_predict_endpoint_returns_prediction(
    mock_predict,
    mock_insert):
    
    mock_predict.return_value = (123, "Test Segment")

    response = client.post(
        "/predict",
        json={
            "recency": 30,
            "frequency": 40,
            "avg_order_value": 100
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["cluster"] == 123

    assert data["label"] == "Test Segment"

    assert mock_insert.called

def test_predict_reject_invalid_input():

    response = client.post(
        "/predict",
        json={
            "recency":"banana",
            "frequency": 5,
            "avg_order_value": 100
        }
    )

    assert response.status_code == 422

    assert "detail" in response.json()

@patch("app.service.reload_production_models")
def test_reload_models_endpoint(mock_reload):
    response = client.post("/reload-models")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert mock_reload.called

def test_simulate_endpoint_local_mode():
    response = client.post("/simulate?mode=standard&num_records=10")
    assert response.status_code == 200
    assert "mocked" in response.json()["message"]

def test_monitoring_drift_endpoint():
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "drift_detected" in response.json()

@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_monitoring_check_and_retrain_healthy(mock_submit):
    response = client.post("/monitoring/check-and-retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert not mock_submit.called

@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_monitoring_check_and_retrain_drifted(mock_submit):
    mock_submit.return_value = "mock-vertex-job-name"
    os.environ["TEST_DRIFT_ACTIVE"] = "true"
    try:
        response = client.post("/monitoring/check-and-retrain")
        assert response.status_code == 200
        assert response.json()["status"] == "drift_detected"
        assert "console_url" in response.json()
        assert mock_submit.called
    finally:
        os.environ["TEST_DRIFT_ACTIVE"] = "false"
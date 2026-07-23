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
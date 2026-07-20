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

@patch("app.main.submit_vertex_training_job")
def test_trigger_churn_retraining_endpoint(mock_submit):
    response = client.post("/train/churn")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

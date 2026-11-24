import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "status":"healthy"
    }

@patch("app.main.insert_clustering_prediction")
@patch("app.main.predict_clustering_service")
def test_predict_endpoint_returns_prediction(
    mock_predict,
    mock_insert):
    
    mock_predict.return_value = (123, "Test Segment")

    # Test symmetrical endpoint /predict/clustering
    response = client.post(
        "/predict/clustering",
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
        "/predict/clustering",
        json={
            "recency":"banana",
            "frequency": 5,
            "avg_order_value": 100
        }
    )

    assert response.status_code == 422

    assert "detail" in response.json()


def test_predict_missing_features_rejects_with_422():
    response = client.post(
        "/predict/clustering",
        json={"recency": 10.0}
    )
    assert response.status_code == 422
    assert "Incomplete Cluster feature set" in response.json()["detail"]


@patch("app.main.insert_clustering_prediction")
@patch("app.main.predict_clustering_service")
@patch("app.db_postgres.get_online_features")
def test_predict_by_customer_id_via_feature_store_clustering(mock_get_features, mock_predict, mock_insert):
    mock_get_features.return_value = {
        "customer_id": "C_101",
        "recency": 15.0,
        "frequency": 8.0,
        "avg_order_value": 250.0
    }
    mock_predict.return_value = (0, "Frequent Buyers")

    response = client.post(
        "/predict/clustering",
        json={"customer_id": "C_101"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "C_101"
    assert data["cluster"] == 0
    assert data["label"] == "Frequent Buyers"
    mock_get_features.assert_called_once_with("C_101")
    assert mock_insert.called


@patch("app.db_postgres.get_online_features")
def test_predict_by_customer_id_incomplete_cluster_features(mock_get_features):
    mock_get_features.return_value = {
        "customer_id": "C_102",
        "recency": 15.0
        # Missing frequency and avg_order_value
    }
    response = client.post(
        "/predict/clustering",
        json={"customer_id": "C_102"}
    )
    assert response.status_code == 422
    assert "Incomplete Cluster feature set" in response.json()["detail"]

@patch("app.service.reload_production_models")
def test_reload_models_endpoint(mock_reload):
    response = client.post("/reload-models")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert mock_reload.called

def test_simulate_endpoint_local_mode(monkeypatch):
    monkeypatch.setenv("USE_BIGQUERY", "false")
    response = client.post("/simulate?mode=standard&num_records=10")
    assert response.status_code == 200
    assert "mocked" in response.json()["message"]

@patch("src.monitoring.calculate_feature_drift")
def test_monitoring_drift_endpoint(mock_drift):
    mock_drift.return_value = {"status": "success", "drift_detected": False, "features": {}}
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "drift_detected" in response.json()

@patch("src.monitoring.calculate_feature_drift")
@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_monitoring_check_and_retrain_healthy(mock_submit, mock_drift):
    mock_drift.return_value = {"drift_detected": False, "p_value": 0.5}
    response = client.post("/monitoring/check-and-retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert not mock_submit.called

@patch("src.monitoring.calculate_feature_drift")
@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_monitoring_check_and_retrain_drifted(mock_submit, mock_drift):
    mock_drift.return_value = {"drift_detected": True, "p_value": 0.01}
    mock_submit.return_value = "mock-vertex-job-name"
    response = client.post("/monitoring/check-and-retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "drift_detected"
    assert "console_url" in response.json()
    assert mock_submit.called

@patch("app.main.insert_churn_prediction")
@patch("app.main.predict_churn_service")
@patch("app.db_postgres.get_online_features")
def test_predict_by_customer_id_via_feature_store(mock_get_features, mock_predict, mock_insert):
    mock_get_features.return_value = {
        "customer_id": "12345",
        "recency": 12.0,
        "frequency": 5,
        "avg_order_value": 150.0,
        "spending_velocity": 1.1,
        "cancellation_rate": 0.05,
        "preferred_shopping_hour": 14
    }
    mock_predict.return_value = (1, 0.85, 0.31)
    
    response = client.post(
        "/predict/churn",
        json={"customer_id": "12345"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "12345"
    assert data["is_churn"] == 1
    assert data["churn_probability"] == 0.85
    assert data["threshold_applied"] == 0.31
    mock_get_features.assert_called_once_with("12345")


@patch("app.db_postgres.get_online_features")
def test_predict_by_customer_id_incomplete_features(mock_get_features):
    mock_get_features.return_value = {
        "customer_id": "999",
        "recency": 10.0,
        "frequency": 2,
        "avg_order_value": 50.0
        # Missing spending_velocity, cancellation_rate, preferred_shopping_hour
    }
    response = client.post(
        "/predict/churn",
        json={"customer_id": "999"}
    )
    assert response.status_code == 422
    assert "Incomplete feature set" in response.json()["detail"]


@patch("app.main.close_pool")
def test_lifespan_graceful_shutdown(mock_close_pool):
    """Verifies that FastAPI Lifespan context manager gracefully teardowns connection pools upon shutdown (Factor IX)."""
    with TestClient(app) as c:
        resp = c.get("/")
        assert resp.status_code == 200
    assert mock_close_pool.called


def test_trigger_job_endpoint_invalid_job():
    resp = client.post("/events/trigger-job?job_name=invalid-job-name")
    assert resp.status_code == 400
    assert "Invalid job name" in resp.json()["detail"]


@patch.dict("os.environ", {"USE_BIGQUERY": "false"})
def test_trigger_job_endpoint_local_mode():
    resp = client.post("/events/trigger-job?job_name=sync-vectors-job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert "Local simulation mode" in resp.json()["message"]



@patch.dict("os.environ", {"USE_BIGQUERY": "true", "GCP_PROJECT": "anna-ml-pipeline", "GCP_LOCATION": "us-central1"})
@patch("subprocess.run")
def test_trigger_job_endpoint_cloud_mode(mock_subproc):
    mock_subproc.return_value = MagicMock(stdout="Execution started\n")
    resp = client.post("/events/trigger-job?job_name=sync-features-job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert "dispatched successfully" in resp.json()["message"]
    assert mock_subproc.called
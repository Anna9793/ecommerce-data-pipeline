import pytest
from unittest.mock import patch, MagicMock
from scripts.sync_all import sync_all
from scripts.run_drift_monitor import evaluate_drift_and_retrain


# ==============================================================================
# 1. Tests for scripts/sync_all.py
# ==============================================================================

@patch("scripts.sync_all.sync_feature_store")
@patch("scripts.sync_all.sync_product_vectors")
def test_sync_all_both_success(mock_sync_vectors, mock_sync_features):
    results = sync_all(features=True, vectors=True)
    assert mock_sync_features.called
    assert mock_sync_vectors.called
    assert results["features"]["status"] == "success"
    assert results["vectors"]["status"] == "success"
    assert "total_duration_seconds" in results


@patch("scripts.sync_all.sync_feature_store")
@patch("scripts.sync_all.sync_product_vectors")
def test_sync_all_features_only(mock_sync_vectors, mock_sync_features):
    results = sync_all(features=True, vectors=False)
    assert mock_sync_features.called
    assert not mock_sync_vectors.called
    assert "features" in results
    assert "vectors" not in results


@patch("scripts.sync_all.sync_feature_store")
@patch("scripts.sync_all.sync_product_vectors")
def test_sync_all_vectors_only(mock_sync_vectors, mock_sync_features):
    results = sync_all(features=False, vectors=True)
    assert not mock_sync_features.called
    assert mock_sync_vectors.called
    assert "features" not in results
    assert "vectors" in results


@patch("scripts.sync_all.sync_feature_store")
def test_sync_all_feature_failure_fail_fast(mock_sync_features):
    mock_sync_features.side_effect = RuntimeError("Firestore down")
    with pytest.raises(RuntimeError, match="Feature store synchronization failed"):
        sync_all(features=True, vectors=True)


@patch("scripts.sync_all.sync_feature_store")
@patch("scripts.sync_all.sync_product_vectors")
def test_sync_all_vector_failure_fail_fast(mock_sync_vectors, mock_sync_features):
    mock_sync_vectors.side_effect = RuntimeError("PostgreSQL pgvector down")
    with pytest.raises(RuntimeError, match="Product vector synchronization failed"):
        sync_all(features=True, vectors=True)


# ==============================================================================
# 2. Tests for scripts/run_drift_monitor.py
# ==============================================================================

@patch("src.monitoring.calculate_feature_drift")
@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_run_drift_monitor_healthy(mock_submit, mock_drift):
    mock_drift.return_value = {"drift_detected": False, "p_value": 0.45}
    result = evaluate_drift_and_retrain()
    assert result["status"] == "healthy"
    assert not mock_submit.called


@patch("src.monitoring.calculate_feature_drift")
@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_run_drift_monitor_drift_triggers_retrain(mock_submit, mock_drift):
    mock_drift.return_value = {"drift_detected": True, "p_value": 0.005}
    mock_submit.return_value = "mock-vertex-job-xyz"
    result = evaluate_drift_and_retrain()
    assert result["status"] == "drift_detected"
    assert result["job_name"] == "mock-vertex-job-xyz"
    assert "console_url" in result
    assert mock_submit.called

import pytest
airflow = pytest.importorskip("airflow")
from airflow.models import DagBag
from dags.ecommerce_master_pipeline_dag import dag, evaluate_drift_and_branch_func

def test_dag_loaded_without_errors():
    """Verifies that the Master DAG loads cleanly without any syntax errors or import exceptions."""
    dagbag = DagBag(dag_folder="dags", include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    assert "ecommerce_daily_master_pipeline" in dagbag.dags

def test_dag_structure_and_task_count():
    """Verifies that the Master DAG contains all 8 expected pipeline tasks."""
    assert dag is not None
    assert len(dag.tasks) == 8

    expected_task_ids = {
        "check_raw_transactions",
        "validate_data_quality",
        "refresh_rfm_features",
        "sync_online_feature_store",
        "sync_product_vectors",
        "evaluate_statistical_drift_branch",
        "trigger_vertex_ml_pipeline",
        "log_pipeline_healthy",
    }
    actual_task_ids = {t.task_id for t in dag.tasks}
    assert actual_task_ids == expected_task_ids

def test_dag_dependencies_and_ordering():
    """Verifies upstream and downstream dependencies across the orchestration graph."""
    check_task = dag.get_task("check_raw_transactions")
    dq_task = dag.get_task("validate_data_quality")
    rfm_task = dag.get_task("refresh_rfm_features")
    sync_fs_task = dag.get_task("sync_online_feature_store")
    sync_vec_task = dag.get_task("sync_product_vectors")
    branch_task = dag.get_task("evaluate_statistical_drift_branch")
    trigger_vertex_task = dag.get_task("trigger_vertex_ml_pipeline")
    healthy_task = dag.get_task("log_pipeline_healthy")

    # Dependency assertions
    assert dq_task in check_task.downstream_list
    assert rfm_task in dq_task.downstream_list
    assert sync_fs_task in rfm_task.downstream_list
    assert sync_vec_task in rfm_task.downstream_list
    assert branch_task in sync_fs_task.downstream_list
    assert branch_task in sync_vec_task.downstream_list
    assert trigger_vertex_task in branch_task.downstream_list
    assert healthy_task in branch_task.downstream_list

def test_drift_branching_logic(monkeypatch):
    """Verifies that the BranchPythonOperator returns the correct task ID based on drift detection."""
    # Test Branch 1: Drift Detected -> trigger_vertex_ml_pipeline
    monkeypatch.setattr("src.monitoring.calculate_feature_drift", lambda: {"drift_detected": True})
    assert evaluate_drift_and_branch_func() == "trigger_vertex_ml_pipeline"

    # Test Branch 2: No Drift -> log_pipeline_healthy
    monkeypatch.setattr("src.monitoring.calculate_feature_drift", lambda: {"drift_detected": False})
    assert evaluate_drift_and_branch_func() == "log_pipeline_healthy"

import pytest
from unittest.mock import patch, MagicMock
from scripts.setup_cloud_jobs import (
    run_cmd,
    get_gcp_context,
    provision_pubsub_topic,
    provision_cloud_run_job,
    provision_eventarc_trigger,
    provision_scheduler_job,
    publish_event,
    setup_all,
)


def test_run_cmd_dry_run():
    res = run_cmd(["echo", "hello"], dry_run=True)
    assert res == "DRY_RUN_SUCCESS"


@patch("subprocess.run")
def test_run_cmd_success(mock_subproc):
    mock_subproc.return_value = MagicMock(stdout="ok\n")
    res = run_cmd(["echo", "hello"], dry_run=False)
    assert res == "ok"
    assert mock_subproc.called


@patch("subprocess.run")
def test_run_cmd_failure(mock_subproc):
    import subprocess
    mock_subproc.side_effect = subprocess.CalledProcessError(1, ["cmd"], stderr="Error")
    res = run_cmd(["cmd"], dry_run=False)
    assert res is None


@patch.dict("os.environ", {"GCP_PROJECT": "test-project", "GCP_LOCATION": "us-central1"})
@patch("scripts.setup_cloud_jobs.run_cmd")
def test_get_gcp_context(mock_run_cmd):
    mock_run_cmd.return_value = "557448715890"
    ctx = get_gcp_context()
    assert ctx["project_id"] == "test-project"
    assert ctx["location"] == "us-central1"
    assert "test-project" in ctx["image_tag"]
    assert "557448715890-compute@developer.gserviceaccount.com" in ctx["service_account"]


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_pubsub_topic_exists(mock_run_cmd):
    mock_run_cmd.return_value = "topic-info"
    ctx = {"project_id": "test-project"}
    success = provision_pubsub_topic("catalog-updates", ctx)
    assert success is True
    # Should only run describe, not create
    assert mock_run_cmd.call_count == 1
    assert "describe" in mock_run_cmd.call_args[0][0]


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_pubsub_topic_creates(mock_run_cmd):
    # First describe returns None, second create returns success
    mock_run_cmd.side_effect = [None, "created"]
    ctx = {"project_id": "test-project"}
    success = provision_pubsub_topic("catalog-updates", ctx)
    assert success is True
    assert mock_run_cmd.call_count == 2
    assert "create" in mock_run_cmd.call_args[0][0]


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_cloud_run_job_create(mock_run_cmd):
    # describe returns None -> action is create
    mock_run_cmd.side_effect = [None, "created"]
    ctx = {
        "project_id": "test-project",
        "location": "us-central1",
        "image_tag": "gcr.io/test-project/img:latest",
    }
    success = provision_cloud_run_job("sync-vectors-job", "scripts/sync_product_vectors.py", ctx)
    assert success is True
    create_args = mock_run_cmd.call_args[0][0]
    assert "create" in create_args
    assert "sync-vectors-job" in create_args
    assert "--command" in create_args


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_cloud_run_job_update(mock_run_cmd):
    # describe returns existing info -> action is update
    mock_run_cmd.side_effect = ["existing_job_info", "updated"]
    ctx = {
        "project_id": "test-project",
        "location": "us-central1",
        "image_tag": "gcr.io/test-project/img:latest",
    }
    success = provision_cloud_run_job("sync-vectors-job", "scripts/sync_product_vectors.py", ctx)
    assert success is True
    update_args = mock_run_cmd.call_args[0][0]
    assert "update" in update_args


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_eventarc_trigger_create(mock_run_cmd):
    # describe returns None -> action is create
    mock_run_cmd.side_effect = [None, "created"]
    ctx = {
        "project_id": "test-project",
        "location": "us-central1",
        "service_account": "sa@test.iam.gserviceaccount.com",
    }
    success = provision_eventarc_trigger(
        "trigger-sync-vectors", "sync-vectors-job", "catalog-updates", ctx
    )
    assert success is True
    create_args = mock_run_cmd.call_args[0][0]
    assert "create" in create_args
    assert "trigger-sync-vectors" in create_args
    assert "--destination-run-job" in create_args


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_provision_scheduler_job(mock_run_cmd):
    mock_run_cmd.side_effect = [None, "created"]
    ctx = {"project_id": "test-project", "location": "us-central1"}
    success = provision_scheduler_job(
        "trigger-drift-monitor", "drift-monitor-job", "0 0 * * 0", ctx
    )
    assert success is True
    sched_args = mock_run_cmd.call_args[0][0]
    assert "run-job" in sched_args
    assert "--schedule" in sched_args


@patch("scripts.setup_cloud_jobs.run_cmd")
def test_publish_event(mock_run_cmd):
    mock_run_cmd.return_value = "published"
    ctx = {"project_id": "test-project"}
    success = publish_event("catalog-updates", '{"action": "test"}', ctx)
    assert success is True
    pub_args = mock_run_cmd.call_args[0][0]
    assert "publish" in pub_args
    assert "catalog-updates" in pub_args


@patch("scripts.setup_cloud_jobs.provision_pubsub_topic")
@patch("scripts.setup_cloud_jobs.provision_cloud_run_job")
@patch("scripts.setup_cloud_jobs.provision_eventarc_trigger")
@patch("scripts.setup_cloud_jobs.provision_scheduler_job")
def test_setup_all(mock_sched, mock_eventarc, mock_job, mock_topic):
    ctx = {
        "project_id": "test-project",
        "location": "us-central1",
        "image_tag": "test-img",
        "service_account": "test-sa",
    }
    setup_all(ctx, dry_run=True)
    assert mock_topic.call_count == 2
    assert mock_job.call_count == 3
    assert mock_eventarc.call_count == 2
    assert mock_sched.call_count == 1

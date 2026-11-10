"""
Vertex AI Experiment Tracker & Model Registry adapter.
Provides unified experiment tracking with automatic offline/local fallback.
"""
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

logger = logging.getLogger(__name__)


class VertexExperimentTracker:
    """
    Manages experiment tracking, metrics, parameters, and model artifacts using
    Google Cloud Vertex AI Experiments with an automatic local fallback for offline/test environments.
    """

    def __init__(
        self,
        experiment_name: str,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        # Vertex AI resource IDs must match regex: [a-z0-9][a-z0-9-]{0,127} (hyphens, no underscores)
        self.experiment_name = experiment_name.lower().replace("_", "-")
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")
        self.active_run = None
        self._is_cloud_active = False

        # Only attempt remote GCP initialization if cloud mode is active
        if os.getenv("USE_BIGQUERY", "false").lower() == "true":
            try:
                from google.cloud import aiplatform

                aiplatform.init(
                    project=self.project_id,
                    location=self.location,
                    experiment=self.experiment_name,
                )
                self._is_cloud_active = True
                logger.info(
                    "Vertex AI Tracker initialized: project=%s, experiment=%s",
                    self.project_id,
                    self.experiment_name,
                )
            except Exception as e:
                logger.warning(
                    "Could not connect to Vertex AI (%s). Operating in local fallback mode.",
                    e,
                )
                self._is_cloud_active = False
        else:
            logger.info("Local/Offline tracking active for experiment: %s", self.experiment_name)

    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """Context manager for an experiment run."""
        if run_name:
            run_name = run_name.lower().replace("_", "-")

        if self._is_cloud_active:
            from google.cloud import aiplatform

            try:
                with aiplatform.start_run(run=run_name) as run:
                    self.active_run = run
                    logger.info("Started Vertex AI run: %s", run_name)
                    yield self
            finally:
                self.active_run = None
        else:
            self.active_run = run_name or "local_run"
            logger.info("Started local experiment run: %s", self.active_run)
            try:
                yield self
            finally:
                self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to Vertex AI or logs locally."""
        logger.info("Experiment Params: %s", params)
        if self._is_cloud_active and self.active_run:
            from google.cloud import aiplatform

            try:
                aiplatform.log_params(params)
            except Exception as e:
                logger.warning("Failed to log params to Vertex AI: %s", e)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs evaluation metrics to Vertex AI or logs locally."""
        logger.info("Experiment Metrics: %s", metrics)
        if self._is_cloud_active and self.active_run:
            from google.cloud import aiplatform

            try:
                aiplatform.log_metrics(metrics)
            except Exception as e:
                logger.warning("Failed to log metrics to Vertex AI: %s", e)

    def log_figure(self, fig, filename: str) -> None:
        """Saves a matplotlib figure locally and logs as artifact."""
        from config.paths import MODELS_DIR

        out_path = MODELS_DIR / filename
        try:
            fig.savefig(out_path, bbox_inches="tight")
            logger.info("Figure saved locally to %s", out_path)
        except Exception as e:
            logger.warning("Could not save figure %s: %s", filename, e)

    def save_and_register_model(self, model: Any, model_name: str, local_path: Path) -> None:
        """Persists model artifact locally and syncs to GCS / Vertex AI Model Registry."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, local_path)
        logger.info("Saved model artifact to %s", local_path)

        if self._is_cloud_active:
            bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
            try:
                from google.cloud import storage

                client = storage.Client(project=self.project_id)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"models/{model_name}.pkl")
                blob.upload_from_filename(str(local_path))
                logger.info(
                    "Uploaded %s to gs://%s/models/%s.pkl",
                    model_name,
                    bucket_name,
                    model_name,
                )
            except Exception as e:
                logger.warning("Failed to upload model to GCS: %s", e)

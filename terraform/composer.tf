# ============================================================
# Google Cloud Composer 2 (Managed Apache Airflow Environment)
# ============================================================

resource "google_composer_environment" "airflow_environment" {
  name   = "ecommerce-airflow-composer"
  region = var.region

  config {
    software_config {
      image_version = "composer-2-airflow-2"

      env_variables = {
        GCP_PROJECT     = var.project_id
        GCS_BUCKET_NAME = var.gcs_bucket_name
        ENVIRONMENT     = var.environment
      }

      pypi_packages = {
        apache-airflow-providers-google = ">=10.0.0"
        scipy                           = ">=1.10.0"
        pgvector                        = ">=0.2.0"
      }
    }

    workloads_config {
      scheduler {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        count      = 1
      }
      web_server {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
      }
      worker {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        min_count  = 1
        max_count  = 3
      }
    }

    environment_size = "ENVIRONMENT_SIZE_SMALL"
  }

  labels = {
    orchestrator = "apache-airflow"
    environment  = var.environment
    managed_by   = "terraform"
  }
}

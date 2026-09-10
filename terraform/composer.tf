# ============================================================
# Google Cloud Composer Service Account & IAM Roles
# ============================================================

resource "google_service_account" "composer_sa" {
  account_id   = "ecommerce-composer-sa"
  display_name = "Cloud Composer Airflow Service Account"
}

resource "google_project_iam_member" "composer_worker" {
  project = var.project_id
  role    = "roles/composer.worker"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

resource "google_project_iam_member" "composer_datalineage" {
  project = var.project_id
  role    = "roles/datalineage.eventsProducer"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

resource "google_project_iam_member" "composer_bigquery_admin" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

# ============================================================
# Google Cloud Composer 2 (Managed Apache Airflow Environment)
# ============================================================

resource "google_composer_environment" "airflow_environment" {
  provider = google-beta
  name     = "ecommerce-airflow-composer"
  region   = var.region

  config {
    node_config {
      service_account = google_service_account.composer_sa.email
    }

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

# ============================================================
# Google Cloud Run Serverless Microservices
# ============================================================

# 1. FastAPI Backend Service (api-service)
resource "google_cloud_run_v2_service" "api_service" {
  name     = "api-service"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      ports {
        container_port = 8000
      }

      env {
        name  = "PORT"
        value = "8000"
      }
      env {
        name  = "USE_BIGQUERY"
        value = "true"
      }
      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = var.gcs_bucket_name
      }
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
    }
  }

  labels = {
    service     = "fastapi-backend"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Allow public unauthenticated access to api-service
resource "google_cloud_run_v2_service_iam_member" "api_public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 2. Streamlit Dashboard Frontend (dashboard-service)
resource "google_cloud_run_v2_service" "dashboard_service" {
  name     = "dashboard-service"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image   = var.container_image
      command = ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api_service.uri
      }
      env {
        name  = "USE_BIGQUERY"
        value = "true"
      }
      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = var.gcs_bucket_name
      }
    }
  }

  labels = {
    service     = "streamlit-dashboard"
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [google_cloud_run_v2_service.api_service]
}

# Allow public unauthenticated access to dashboard-service
resource "google_cloud_run_v2_service_iam_member" "dashboard_public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.dashboard_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

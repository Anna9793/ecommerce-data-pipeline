# ============================================================
# Google Cloud Storage Bucket for ML Models & Data Artifacts
# ============================================================

resource "google_storage_bucket" "mlops_artifacts" {
  name          = var.gcs_bucket_name
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "ecommerce-data-pipeline"
  }
}

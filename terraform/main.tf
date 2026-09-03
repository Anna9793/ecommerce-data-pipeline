provider "google" {
  project = var.project_id
  region  = var.region
}

# ============================================================
# Enable Essential GCP APIs
# ============================================================

locals {
  gcp_services = [
    "bigquery.googleapis.com",
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudscheduler.googleapis.com",
    "firestore.googleapis.com",
    "pubsub.googleapis.com",
    "dataflow.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each                   = toset(locals.gcp_services)
  project                    = var.project_id
  service                    = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}

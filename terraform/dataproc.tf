# ============================================================
# Google Cloud Dataproc (Apache Spark) Infrastructure
# ============================================================

# 1. Autoscaling Policy for Ephemeral & Dynamic Spark Workloads
resource "google_dataproc_autoscaling_policy" "spark_autoscaler" {
  policy_id = "ecommerce-spark-autoscaler"
  location  = var.region

  worker_config {
    min_instances = 2
    max_instances = 4
  }

  secondary_worker_config {
    min_instances = 0
    max_instances = 8 # Spot / Preemptible VMs for 60-80% cost savings
  }

  basic_algorithm {
    yarn_config {
      scale_up_factor              = 0.5
      scale_down_factor            = 0.5
      scale_up_min_worker_fraction = 0.0
      scale_down_min_worker_fraction = 0.0
      graceful_decommission_timeout = "300s"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# 2. Dataproc Cluster for Distributed PySpark Feature Engineering
resource "google_dataproc_cluster" "pyspark_cluster" {
  name     = "ecommerce-pyspark-cluster"
  region   = var.region
  project  = var.project_id

  cluster_config {
    staging_bucket = google_storage_bucket.mlops_artifacts.name

    master_config {
      num_instances = 1
      machine_type  = "n1-standard-4"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 50
      }
    }

    worker_config {
      num_instances = 2
      machine_type  = "n1-standard-4"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 50
      }
    }

    # FinOps: Preemptible / Spot VMs for elastic scale-out
    preemptible_worker_config {
      num_instances = 2
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 50
      }
    }

    autoscaling_config {
      policy_uri = google_dataproc_autoscaling_policy.spark_autoscaler.name
    }

    software_config {
      image_version = "2.1-debian11"
      override_properties = {
        "spark:spark.serializer"                     = "org.apache.spark.serializer.KryoSerializer"
        "spark:spark.sql.execution.arrow.pyspark.enabled" = "true"
        "spark:spark.dynamicAllocation.enabled"      = "true"
      }
    }
  }

  labels = {
    environment = var.environment
    workload    = "pyspark-feature-engineering"
    managed-by  = "terraform"
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_storage_bucket.mlops_artifacts
  ]
}

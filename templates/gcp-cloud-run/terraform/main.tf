terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "run" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container_registry" {
  service = "containerregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry repository for container images
resource "google_artifact_registry_repository" "ml_models" {
  location      = var.region
  repository_id = "${var.project_name}-models"
  description   = "Docker repository for ML model containers"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.artifact_registry]
}

# Cloud Storage bucket for models (optional)
resource "google_storage_bucket" "model_bucket" {
  name          = "${var.project_id}-${var.project_name}-models-${var.environment}"
  location      = var.region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Service account for Cloud Run
resource "google_service_account" "cloud_run_sa" {
  account_id   = "${var.project_name}-cloudrun-${var.environment}"
  display_name = "Cloud Run service account for ${var.project_name}"
  description  = "Service account for Cloud Run ML model deployment"
}

# Grant service account access to storage bucket
resource "google_storage_bucket_iam_member" "cloud_run_sa_storage" {
  bucket = google_storage_bucket.model_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant service account logging permissions
resource "google_project_iam_member" "cloud_run_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "ml_model" {
  name     = "${var.project_name}-${var.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        cpu_idle = true
        startup_cpu_boost = true
      }

      env {
        name  = "MODEL_PATH"
        value = var.model_path
      }

      env {
        name  = "MODEL_VERSION"
        value = var.model_version
      }

      env {
        name  = "MODEL_TYPE"
        value = var.model_type
      }

      env {
        name  = "LOG_LEVEL"
        value = var.log_level
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      dynamic "env" {
        for_each = var.additional_env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 3
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    timeout = "${var.timeout}s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.run,
    google_service_account.cloud_run_sa
  ]

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# IAM policy to allow unauthenticated access (optional)
resource "google_cloud_run_service_iam_member" "public_access" {
  count    = var.allow_unauthenticated ? 1 : 0
  location = google_cloud_run_v2_service.ml_model.location
  service  = google_cloud_run_v2_service.ml_model.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# IAM policy for authenticated access
resource "google_cloud_run_service_iam_member" "authenticated_access" {
  count    = var.allow_unauthenticated ? 0 : 1
  location = google_cloud_run_v2_service.ml_model.location
  service  = google_cloud_run_v2_service.ml_model.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Cloud Monitoring alert policy for errors
resource "google_monitoring_alert_policy" "error_rate" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "${var.project_name} Error Rate Alert"
  combiner     = "OR"

  conditions {
    display_name = "Error rate is above threshold"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.ml_model.name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.error_threshold

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }
}

# Cloud Monitoring alert policy for latency
resource "google_monitoring_alert_policy" "latency" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "${var.project_name} Latency Alert"
  combiner     = "OR"

  conditions {
    display_name = "Request latency is above threshold"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.ml_model.name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.latency_threshold_ms

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }
}

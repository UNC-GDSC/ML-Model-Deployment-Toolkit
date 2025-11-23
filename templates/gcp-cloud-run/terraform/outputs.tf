output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.ml_model.name
}

output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.ml_model.uri
}

output "service_account_email" {
  description = "Email of the service account"
  value       = google_service_account.cloud_run_sa.email
}

output "storage_bucket_name" {
  description = "Name of the storage bucket"
  value       = google_storage_bucket.model_bucket.name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.ml_models.id
}

output "container_image_url" {
  description = "Container image URL format"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_models.repository_id}"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-model"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
}

variable "cpu" {
  description = "CPU allocation"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory allocation"
  type        = string
  default     = "2Gi"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 100
}

variable "timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}

variable "model_path" {
  description = "Path to model file"
  type        = string
  default     = "/app/models/model.pkl"
}

variable "model_version" {
  description = "Model version"
  type        = string
  default     = "1.0.0"
}

variable "model_type" {
  description = "Model type (sklearn, tensorflow, pytorch)"
  type        = string
  default     = "sklearn"
}

variable "log_level" {
  description = "Logging level"
  type        = string
  default     = "INFO"
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated access"
  type        = bool
  default     = true
}

variable "additional_env_vars" {
  description = "Additional environment variables"
  type        = map(string)
  default     = {}
}

variable "enable_monitoring" {
  description = "Enable Cloud Monitoring alerts"
  type        = bool
  default     = true
}

variable "error_threshold" {
  description = "Error rate threshold for alerts"
  type        = number
  default     = 10
}

variable "latency_threshold_ms" {
  description = "Latency threshold in milliseconds for alerts"
  type        = number
  default     = 1000
}

variable "notification_channels" {
  description = "Notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "aws_region" {
  description = "AWS region for SageMaker deployment"
  type        = string
  default     = "us-east-1"
}

variable "model_name" {
  description = "Name of the ML model (lowercase, no spaces)"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.model_name))
    error_message = "Model name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "model_data_url" {
  description = "S3 URL to model artifacts (must be a tar.gz file)"
  type        = string

  validation {
    condition     = can(regex("^s3://", var.model_data_url))
    error_message = "Model data URL must be a valid S3 URL starting with s3://."
  }
}

variable "instance_type" {
  description = "SageMaker instance type for inference"
  type        = string
  default     = "ml.t2.medium"

  validation {
    condition     = can(regex("^ml\\.", var.instance_type))
    error_message = "Instance type must be a valid SageMaker instance type starting with 'ml.'."
  }
}

variable "instance_count" {
  description = "Initial number of instances"
  type        = number
  default     = 1

  validation {
    condition     = var.instance_count >= 1 && var.instance_count <= 10
    error_message = "Instance count must be between 1 and 10."
  }
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "auto_scaling_enabled" {
  description = "Enable auto-scaling for the endpoint"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "Minimum instance count for auto-scaling"
  type        = number
  default     = 1

  validation {
    condition     = var.min_capacity >= 1
    error_message = "Minimum capacity must be at least 1."
  }
}

variable "max_capacity" {
  description = "Maximum instance count for auto-scaling"
  type        = number
  default     = 3

  validation {
    condition     = var.max_capacity >= 1 && var.max_capacity <= 10
    error_message = "Maximum capacity must be between 1 and 10."
  }
}

variable "target_invocations_per_instance" {
  description = "Target number of invocations per instance for auto-scaling"
  type        = number
  default     = 1000

  validation {
    condition     = var.target_invocations_per_instance >= 100
    error_message = "Target invocations must be at least 100."
  }
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
    Project   = "ML-Model-Deployment"
  }
}

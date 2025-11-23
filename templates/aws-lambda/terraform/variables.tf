variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
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

variable "lambda_runtime" {
  description = "Lambda runtime"
  type        = string
  default     = "python3.11"
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 30
}

variable "lambda_memory" {
  description = "Lambda memory in MB"
  type        = number
  default     = 1024
}

variable "lambda_zip_path" {
  description = "Path to Lambda deployment package"
  type        = string
  default     = "../deployment.zip"
}

variable "create_lambda_layer" {
  description = "Whether to create a Lambda layer for dependencies"
  type        = bool
  default     = false
}

variable "lambda_layer_zip_path" {
  description = "Path to Lambda layer zip file"
  type        = string
  default     = "../layer.zip"
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

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "create_function_url" {
  description = "Create Lambda Function URL"
  type        = bool
  default     = true
}

variable "function_url_auth_type" {
  description = "Function URL authorization type (NONE or AWS_IAM)"
  type        = string
  default     = "NONE"
}

variable "create_api_gateway" {
  description = "Create API Gateway"
  type        = bool
  default     = false
}

variable "cors_allow_origins" {
  description = "CORS allowed origins"
  type        = list(string)
  default     = ["*"]
}

variable "additional_env_vars" {
  description = "Additional environment variables"
  type        = map(string)
  default     = {}
}

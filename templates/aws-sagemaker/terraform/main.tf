/**
 * AWS SageMaker Endpoint Terraform Configuration
 *
 * This configuration creates:
 * - IAM role for SageMaker
 * - SageMaker model
 * - SageMaker endpoint configuration
 * - SageMaker endpoint
 * - CloudWatch log group
 * - Auto-scaling configuration
 */

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "model_name" {
  description = "Name of the ML model"
  type        = string
}

variable "model_data_url" {
  description = "S3 URL to model artifacts (tar.gz)"
  type        = string
}

variable "instance_type" {
  description = "SageMaker instance type"
  type        = string
  default     = "ml.t2.medium"
}

variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "auto_scaling_enabled" {
  description = "Enable auto-scaling"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "Minimum instance count for auto-scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum instance count for auto-scaling"
  type        = number
  default     = 3
}

variable "target_invocations_per_instance" {
  description = "Target invocations per instance for scaling"
  type        = number
  default     = 1000
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Data sources
data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

# IAM role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.model_name}-sagemaker-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    var.tags,
    {
      Name        = "${var.model_name}-sagemaker-role"
      Environment = var.environment
    }
  )
}

# IAM policy for SageMaker role
resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "${var.model_name}-sagemaker-policy"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch log group
resource "aws_cloudwatch_log_group" "sagemaker_logs" {
  name              = "/aws/sagemaker/Endpoints/${var.model_name}-${var.environment}"
  retention_in_days = 7

  tags = merge(
    var.tags,
    {
      Name        = "${var.model_name}-logs"
      Environment = var.environment
    }
  )
}

# SageMaker model
resource "aws_sagemaker_model" "model" {
  name               = "${var.model_name}-model-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    # Use SageMaker's built-in scikit-learn container
    # For custom containers, replace with your ECR image URI
    image          = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-sklearn:latest"
    model_data_url = var.model_data_url

    environment = {
      SAGEMAKER_PROGRAM         = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = var.model_data_url
      SAGEMAKER_REGION          = var.aws_region
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.model_name}-model"
      Environment = var.environment
    }
  )
}

# SageMaker endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "config" {
  name = "${var.model_name}-config-${var.environment}-${formatdate("YYYYMMDDhhmmss", timestamp())}"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.model.name
    instance_type          = var.instance_type
    initial_instance_count = var.instance_count
    initial_variant_weight = 1.0
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://sagemaker-${var.aws_region}-${data.aws_caller_identity.current.account_id}/${var.model_name}/data-capture"

    capture_options {
      capture_mode = "InputAndOutput"
    }

    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.model_name}-config"
      Environment = var.environment
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# SageMaker endpoint
resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = "${var.model_name}-endpoint-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.config.name

  tags = merge(
    var.tags,
    {
      Name        = "${var.model_name}-endpoint"
      Environment = var.environment
    }
  )
}

# Auto-scaling target
resource "aws_appautoscaling_target" "sagemaker_target" {
  count = var.auto_scaling_enabled ? 1 : 0

  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "endpoint/${aws_sagemaker_endpoint.endpoint.name}/variant/AllTraffic"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

# Auto-scaling policy
resource "aws_appautoscaling_policy" "sagemaker_policy" {
  count = var.auto_scaling_enabled ? 1 : 0

  name               = "${var.model_name}-scaling-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }

    target_value       = var.target_invocations_per_instance
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "model_latency" {
  alarm_name          = "${var.model_name}-high-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = "60"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "Alert when model latency is high"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.endpoint.name
    VariantName  = "AllTraffic"
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "model_errors" {
  alarm_name          = "${var.model_name}-high-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "ModelInvocationErrors"
  namespace           = "AWS/SageMaker"
  period              = "60"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "Alert when model has errors"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.endpoint.name
    VariantName  = "AllTraffic"
  }

  tags = var.tags
}

# Outputs
output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.endpoint.name
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.endpoint.arn
}

output "model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.model.name
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker IAM role"
  value       = aws_iam_role.sagemaker_role.arn
}

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.sagemaker_logs.name
}

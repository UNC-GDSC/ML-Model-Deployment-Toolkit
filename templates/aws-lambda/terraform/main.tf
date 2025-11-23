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

# S3 bucket for model storage
resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-models-${var.environment}"

  tags = {
    Name        = "${var.project_name}-models"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_bucket_encryption" {
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM role for Lambda function
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-lambda-role"
    Environment = var.environment
  }
}

# IAM policy for Lambda function
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_bucket.arn,
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Lambda layer for dependencies
resource "aws_lambda_layer_version" "dependencies" {
  count               = var.create_lambda_layer ? 1 : 0
  filename            = var.lambda_layer_zip_path
  layer_name          = "${var.project_name}-dependencies"
  compatible_runtimes = ["python3.11", "python3.10", "python3.9"]
  source_code_hash    = filebase64sha256(var.lambda_layer_zip_path)

  description = "ML model dependencies"
}

# Lambda function
resource "aws_lambda_function" "ml_model" {
  filename      = var.lambda_zip_path
  function_name = "${var.project_name}-${var.environment}"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.lambda_handler"
  runtime       = var.lambda_runtime
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory

  source_code_hash = filebase64sha256(var.lambda_zip_path)

  layers = var.create_lambda_layer ? [aws_lambda_layer_version.dependencies[0].arn] : []

  environment {
    variables = merge({
      MODEL_PATH    = "/opt/model/model.pkl"
      MODEL_VERSION = var.model_version
      MODEL_TYPE    = var.model_type
      LOG_LEVEL     = var.log_level
      ENVIRONMENT   = var.environment
    }, var.additional_env_vars)
  }

  tags = {
    Name        = "${var.project_name}-function"
    Environment = var.environment
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.ml_model.function_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.project_name}-logs"
    Environment = var.environment
  }
}

# Lambda Function URL (simpler alternative to API Gateway)
resource "aws_lambda_function_url" "ml_model_url" {
  count              = var.create_function_url ? 1 : 0
  function_name      = aws_lambda_function.ml_model.function_name
  authorization_type = var.function_url_auth_type

  cors {
    allow_origins     = var.cors_allow_origins
    allow_methods     = ["GET", "POST"]
    allow_headers     = ["content-type", "authorization"]
    max_age           = 86400
  }
}

# API Gateway (optional, more feature-rich)
resource "aws_apigatewayv2_api" "ml_api" {
  count         = var.create_api_gateway ? 1 : 0
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = var.cors_allow_origins
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
    max_age       = 86400
  }

  tags = {
    Name        = "${var.project_name}-api"
    Environment = var.environment
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  count              = var.create_api_gateway ? 1 : 0
  api_id             = aws_apigatewayv2_api.ml_api[0].id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.ml_model.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "predict" {
  count     = var.create_api_gateway ? 1 : 0
  api_id    = aws_apigatewayv2_api.ml_api[0].id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration[0].id}"
}

resource "aws_apigatewayv2_route" "health" {
  count     = var.create_api_gateway ? 1 : 0
  api_id    = aws_apigatewayv2_api.ml_api[0].id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration[0].id}"
}

resource "aws_apigatewayv2_stage" "default" {
  count       = var.create_api_gateway ? 1 : 0
  api_id      = aws_apigatewayv2_api.ml_api[0].id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs[0].arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  tags = {
    Name        = "${var.project_name}-stage"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "api_logs" {
  count             = var.create_api_gateway ? 1 : 0
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.project_name}-api-logs"
    Environment = var.environment
  }
}

resource "aws_lambda_permission" "api_gateway" {
  count         = var.create_api_gateway ? 1 : 0
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ml_model.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.ml_api[0].execution_arn}/*/*"
}

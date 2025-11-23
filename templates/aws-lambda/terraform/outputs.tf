output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.ml_model.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.ml_model.arn
}

output "lambda_function_url" {
  description = "Function URL endpoint"
  value       = var.create_function_url ? aws_lambda_function_url.ml_model_url[0].function_url : null
}

output "api_gateway_endpoint" {
  description = "API Gateway endpoint URL"
  value       = var.create_api_gateway ? aws_apigatewayv2_api.ml_api[0].api_endpoint : null
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for models"
  value       = aws_s3_bucket.model_bucket.id
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.lambda_logs.name
}

output "endpoint_url" {
  description = "Primary endpoint URL for the API"
  value = var.create_api_gateway ? aws_apigatewayv2_api.ml_api[0].api_endpoint : (
    var.create_function_url ? aws_lambda_function_url.ml_model_url[0].function_url : "No endpoint created"
  )
}

# AWS Lambda Deployment Template

Deploy your ML models to AWS Lambda with this production-ready template.

## Features

- ⚡ Serverless deployment with auto-scaling
- 🔒 IAM roles and policies configured
- 📊 CloudWatch logging and monitoring
- 🌐 API Gateway or Function URL support
- 💾 S3 bucket for model storage
- 🏗️ Infrastructure as Code with Terraform

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- Python 3.8+
- Your trained model file (`.pkl`, `.h5`, `.pt`, etc.)

## Quick Start

### 1. Prepare Your Model

Place your trained model in the `models/` directory:

```bash
cp /path/to/your/model.pkl models/model.pkl
```

### 2. Configure Environment

Edit `terraform/variables.tf` or set environment variables:

```bash
export PROJECT_NAME="my-ml-model"
export ENVIRONMENT="prod"
export AWS_REGION="us-east-1"
export MODEL_PATH="models/model.pkl"
```

### 3. Deploy

Run the deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

Or deploy manually:

```bash
# Package the application
pip install -r requirements.txt -t package/
cp handler.py package/
cp -r ../../src package/
cd package && zip -r ../deployment.zip . && cd ..

# Deploy with Terraform
cd terraform
terraform init
terraform apply
```

### 4. Test Your Deployment

```bash
# Get the endpoint URL
ENDPOINT_URL=$(cd terraform && terraform output -raw endpoint_url)

# Health check
curl $ENDPOINT_URL/health

# Make a prediction
curl -X POST $ENDPOINT_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Configuration

### Lambda Function Settings

Edit `terraform/variables.tf`:

```hcl
lambda_timeout = 30      # Timeout in seconds (max 900)
lambda_memory  = 1024    # Memory in MB (128-10240)
lambda_runtime = "python3.11"
```

### Model Configuration

Set environment variables in `terraform/main.tf`:

```hcl
environment {
  variables = {
    MODEL_PATH    = "/opt/model/model.pkl"
    MODEL_VERSION = "1.0.0"
    MODEL_TYPE    = "sklearn"  # or "tensorflow", "pytorch"
    LOG_LEVEL     = "INFO"
  }
}
```

### API Endpoint Options

#### Option 1: Lambda Function URL (Simpler)

```hcl
create_function_url = true
function_url_auth_type = "NONE"  # or "AWS_IAM"
```

#### Option 2: API Gateway (More Features)

```hcl
create_api_gateway = true
create_function_url = false
```

## Project Structure

```
aws-lambda/
├── handler.py              # Lambda function handler
├── requirements.txt        # Python dependencies
├── deploy.sh              # Deployment script
├── README.md              # This file
└── terraform/
    ├── main.tf            # Main Terraform configuration
    ├── variables.tf       # Variable definitions
    └── outputs.tf         # Output definitions
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### Prediction

```bash
POST /predict
Content-Type: application/json

{
  "features": [1.0, 2.0, 3.0, 4.0],
  "return_probabilities": false
}
```

Response:
```json
{
  "prediction": 1,
  "model_version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "latency_ms": 45.2
}
```

### Model Info

```bash
GET /info
```

## Monitoring

### CloudWatch Logs

View logs in AWS Console:
```
CloudWatch > Log Groups > /aws/lambda/<function-name>
```

Or using CLI:
```bash
aws logs tail /aws/lambda/ml-model-prod --follow
```

### Metrics

Monitor Lambda metrics:
- Invocations
- Duration
- Errors
- Throttles

## Cost Optimization

1. **Right-size memory**: Start with 1024 MB and adjust based on performance
2. **Use Lambda layers**: Share dependencies across functions
3. **Set appropriate timeout**: Avoid paying for long-running failed requests
4. **Use reserved concurrency**: Control costs for high-traffic scenarios

Estimated costs (us-east-1):
- Lambda: $0.20 per 1M requests + $0.0000166667 per GB-second
- API Gateway: $1.00 per million requests
- CloudWatch: Logs storage ~$0.50/GB/month

## Troubleshooting

### Issue: Lambda timeout

**Solution**: Increase timeout in `variables.tf`:
```hcl
lambda_timeout = 60
```

### Issue: Out of memory

**Solution**: Increase memory allocation:
```hcl
lambda_memory = 2048
```

### Issue: Model file too large

**Solution**: Use Lambda layers or load from S3:
```python
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket-name', 'model.pkl', '/tmp/model.pkl')
```

### Issue: Cold start latency

**Solution**:
1. Use provisioned concurrency
2. Optimize model loading
3. Use lighter model formats (ONNX)

## Security Best Practices

1. **Enable IAM authentication** for production:
   ```hcl
   function_url_auth_type = "AWS_IAM"
   ```

2. **Use VPC** for sensitive workloads
3. **Enable encryption** for S3 buckets (already configured)
4. **Rotate credentials** regularly
5. **Use AWS Secrets Manager** for API keys

## Clean Up

To delete all resources:

```bash
cd terraform
terraform destroy
```

## Next Steps

- Add API authentication
- Implement rate limiting
- Set up CI/CD pipeline
- Add model versioning
- Configure auto-scaling policies

## Support

For issues and questions:
- GitHub Issues: https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues
- Documentation: ../../docs/aws-lambda.md

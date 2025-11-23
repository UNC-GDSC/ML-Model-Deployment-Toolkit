# AWS SageMaker Deployment Template

Deploy ML models to AWS SageMaker with managed infrastructure, auto-scaling, and comprehensive monitoring.

## Overview

AWS SageMaker is a fully managed machine learning service that provides:

- **Managed Hosting**: Automatic infrastructure management and scaling
- **Multi-Model Endpoints**: Host multiple models on a single endpoint
- **A/B Testing**: Built-in support for traffic splitting
- **Auto-Scaling**: Automatically scale based on traffic
- **Model Monitoring**: Data capture and model quality monitoring
- **High Availability**: Multi-AZ deployment with automatic failover

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- Python 3.11+
- AWS account with SageMaker permissions

## Quick Start

### 1. Package Your Model

Place your trained model in the `models/` directory:

```bash
# Example structure:
models/
└── my-model/
    ├── model.pkl          # Your trained model
    └── metadata.json      # Model metadata
```

### 2. Deploy to SageMaker

```bash
# Deploy with default settings
./deploy.sh my-model

# Deploy to specific environment
./deploy.sh my-model prod us-west-2 ml.m5.large

# Arguments:
# 1. MODEL_NAME (required)
# 2. ENVIRONMENT (dev|staging|prod, default: dev)
# 3. AWS_REGION (default: us-east-1)
# 4. INSTANCE_TYPE (default: ml.t2.medium)
```

The deployment script will:
1. Package your model and inference code
2. Upload to S3
3. Create SageMaker endpoint using Terraform
4. Configure auto-scaling and monitoring

### 3. Test Your Endpoint

```bash
# Test the deployed endpoint
./test.sh my-model us-east-1 dev
```

## Instance Types

Choose instance type based on your requirements:

### CPU Instances

| Instance Type | vCPUs | Memory | Use Case | Cost/hour* |
|--------------|-------|--------|----------|-----------|
| ml.t2.medium | 2 | 4 GB | Development/Testing | $0.065 |
| ml.m5.large | 2 | 8 GB | Light production | $0.115 |
| ml.m5.xlarge | 4 | 16 GB | Medium production | $0.230 |
| ml.c5.2xlarge | 8 | 16 GB | CPU-intensive | $0.408 |

### GPU Instances

| Instance Type | GPUs | Memory | GPU Memory | Use Case | Cost/hour* |
|--------------|------|--------|------------|----------|-----------|
| ml.g4dn.xlarge | 1 | 16 GB | 16 GB | Deep learning inference | $0.736 |
| ml.g4dn.2xlarge | 1 | 32 GB | 16 GB | Large DL models | $1.046 |
| ml.p3.2xlarge | 1 | 61 GB | 16 GB | High-performance DL | $3.825 |

*Approximate costs in us-east-1 region (check AWS pricing for current rates)

## Configuration

### Terraform Variables

Edit `terraform/terraform.tfvars` or pass variables:

```hcl
model_name                      = "my-model"
model_data_url                  = "s3://bucket/model.tar.gz"
instance_type                   = "ml.t2.medium"
instance_count                  = 1
environment                     = "prod"
aws_region                      = "us-east-1"

# Auto-scaling
auto_scaling_enabled            = true
min_capacity                    = 1
max_capacity                    = 5
target_invocations_per_instance = 1000

# Tags
tags = {
  Project     = "ML-Deployment"
  Team        = "Data-Science"
  Environment = "Production"
}
```

### Inference Code

The `inference.py` script implements SageMaker's inference interface:

```python
def model_fn(model_dir):
    """Load model from S3"""
    pass

def input_fn(request_body, content_type):
    """Deserialize input"""
    pass

def predict_fn(input_data, model):
    """Make predictions"""
    pass

def output_fn(prediction, accept):
    """Serialize output"""
    pass
```

## Auto-Scaling

SageMaker automatically scales based on invocations per instance:

```bash
# The deployment configures:
- Min instances: 1
- Max instances: 3
- Target: 1000 invocations per instance
- Scale-out cooldown: 60 seconds
- Scale-in cooldown: 300 seconds
```

Monitor scaling in CloudWatch or with:

```bash
aws application-autoscaling describe-scaling-activities \
  --service-namespace sagemaker \
  --resource-id endpoint/my-model-endpoint-prod/variant/AllTraffic
```

## Making Predictions

### Python

```python
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = client.invoke_endpoint(
    EndpointName='my-model-endpoint-prod',
    ContentType='application/json',
    Body=json.dumps({'features': [5.1, 3.5, 1.4, 0.2]})
)

result = json.loads(response['Body'].read())
print(result)
```

### AWS CLI

```bash
# Single prediction
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-model-endpoint-prod \
  --content-type application/json \
  --body '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  response.json

# Batch prediction
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-model-endpoint-prod \
  --content-type application/json \
  --body '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}' \
  response.json
```

### cURL

```bash
# Using AWS Signature V4 (requires aws-sigv4-proxy or similar)
curl -X POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-model-endpoint-prod/invocations \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Monitoring

### CloudWatch Metrics

Key metrics automatically collected:

- **Invocations**: Number of requests
- **ModelLatency**: Time taken for model inference (ms)
- **OverheadLatency**: SageMaker overhead time (ms)
- **Invocation4XXErrors**: Client errors
- **Invocation5XXErrors**: Server errors
- **CPUUtilization**: Instance CPU usage
- **MemoryUtilization**: Instance memory usage
- **DiskUtilization**: Instance disk usage

### CloudWatch Alarms

Automatic alarms configured for:
- High latency (> 1000ms)
- High error rate (> 5 errors/minute)

### View Metrics

```bash
# Get invocation count
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=my-model-endpoint-prod Name=VariantName,Value=AllTraffic \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 3600 \
  --statistics Sum

# Get average latency
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=my-model-endpoint-prod Name=VariantName,Value=AllTraffic \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 3600 \
  --statistics Average,Maximum
```

## Data Capture

The deployment automatically enables data capture for model monitoring:

```bash
# Data is saved to:
s3://sagemaker-{region}-{account}/my-model/data-capture/

# Access captured data
aws s3 ls s3://sagemaker-us-east-1-123456789/my-model/data-capture/ --recursive
```

Captured data includes:
- Input payloads
- Output predictions
- Timestamps
- Inference metadata

## Multi-Model Endpoints

Deploy multiple models to a single endpoint for cost optimization:

```python
# Not yet implemented in this template
# Coming soon: Multi-model endpoint support
```

## A/B Testing

Deploy multiple model variants with traffic splitting:

```hcl
# In main.tf, add additional production variants:
production_variants {
  variant_name           = "VariantA"
  model_name             = aws_sagemaker_model.model_a.name
  instance_type          = "ml.t2.medium"
  initial_instance_count = 1
  initial_variant_weight = 0.8  # 80% traffic
}

production_variants {
  variant_name           = "VariantB"
  model_name             = aws_sagemaker_model.model_b.name
  instance_type          = "ml.t2.medium"
  initial_instance_count = 1
  initial_variant_weight = 0.2  # 20% traffic
}
```

## Custom Docker Container

For custom dependencies or frameworks:

```bash
# Build custom container
docker build -t sagemaker-custom:latest .

# Push to ECR
aws ecr create-repository --repository-name sagemaker-custom
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin {account}.dkr.ecr.us-east-1.amazonaws.com
docker tag sagemaker-custom:latest {account}.dkr.ecr.us-east-1.amazonaws.com/sagemaker-custom:latest
docker push {account}.dkr.ecr.us-east-1.amazonaws.com/sagemaker-custom:latest

# Update Terraform to use custom image
# Edit main.tf and update the image parameter in primary_container
```

## Cost Optimization

### Strategies

1. **Instance Selection**: Use smallest instance that meets latency requirements
2. **Auto-Scaling**: Scale down during low-traffic periods
3. **Multi-Model Endpoints**: Share infrastructure across models
4. **Serverless Inference**: For intermittent traffic (coming soon)
5. **Instance Savings Plans**: Up to 64% savings for steady-state workloads

### Cost Estimation

```bash
# Example costs for ml.t2.medium (us-east-1):
# - Instance: $0.065/hour = $46.80/month
# - Data transfer: ~$0.09/GB
# - Storage: $0.023/GB-month

# For 1 instance running 24/7:
# Monthly cost: ~$50/month (excluding data transfer)

# With auto-scaling (avg 1.5 instances):
# Monthly cost: ~$75/month
```

## Troubleshooting

### Endpoint Fails to Create

```bash
# Check CloudWatch logs
aws logs tail /aws/sagemaker/Endpoints/my-model-endpoint-prod --follow

# Common issues:
# 1. Model artifacts not in tar.gz format
# 2. Insufficient IAM permissions
# 3. Invalid instance type for region
# 4. Container fails health checks
```

### High Latency

```bash
# Check instance utilization
aws cloudwatch get-metric-statistics \
  --namespace /aws/sagemaker/Endpoints \
  --metric-name CPUUtilization \
  --dimensions Name=EndpointName,Value=my-model-endpoint-prod

# Solutions:
# 1. Scale up instances
# 2. Use larger instance type
# 3. Optimize model (quantization, pruning)
# 4. Enable model compilation
```

### Prediction Errors

```bash
# Check error logs
aws logs filter-pattern "ERROR" --log-group-name /aws/sagemaker/Endpoints/my-model-endpoint-prod

# Validate input format
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-model-endpoint-prod \
  --content-type application/json \
  --body '{"features": [1, 2, 3]}' \
  --debug \
  response.json
```

## Cleanup

```bash
# Destroy all resources
cd terraform
terraform destroy

# Or use AWS CLI
aws sagemaker delete-endpoint --endpoint-name my-model-endpoint-prod
aws sagemaker delete-endpoint-config --endpoint-config-name my-model-config-prod
aws sagemaker delete-model --model-name my-model-model-prod
```

## Advanced Features

### Model Compilation

Optimize models for specific hardware:

```python
# Using SageMaker Neo
import boto3

client = boto3.client('sagemaker')

response = client.create_compilation_job(
    CompilationJobName='my-model-compilation',
    RoleArn='arn:aws:iam::123456789:role/SageMakerRole',
    InputConfig={
        'S3Uri': 's3://bucket/model.tar.gz',
        'DataInputConfig': '{"input": [1,3,224,224]}',
        'Framework': 'SKLEARN'
    },
    OutputConfig={
        'S3OutputLocation': 's3://bucket/compiled/',
        'TargetDevice': 'ml_c5'  # or 'ml_p3', 'jetson_nano', etc.
    }
)
```

### Batch Transform

For offline batch predictions:

```bash
# Create batch transform job
aws sagemaker create-transform-job \
  --transform-job-name my-batch-job \
  --model-name my-model-model-prod \
  --transform-input DataSource={S3DataSource={S3Uri=s3://bucket/input/}},ContentType=text/csv \
  --transform-output S3OutputPath=s3://bucket/output/ \
  --transform-resources InstanceType=ml.m5.large,InstanceCount=1
```

## Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)

## Support

For issues or questions:
- Check CloudWatch Logs
- Review [SageMaker FAQs](https://aws.amazon.com/sagemaker/faqs/)
- Open an issue in this repository

# GCP Cloud Run Deployment Template

Deploy your ML models to GCP Cloud Run with this production-ready containerized template.

## Features

- 🐳 Fully containerized deployment
- 🔄 Auto-scaling from 0 to N instances
- 📊 Cloud Logging and Monitoring integrated
- 🔒 IAM-based access control
- 💾 Cloud Storage for model artifacts
- 🏗️ Infrastructure as Code with Terraform
- ⚡ Fast cold start times
- 🌍 Global deployment options

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
- Terraform >= 1.0 (optional, can use `gcloud` directly)
- Docker installed
- GCP project with billing enabled
- Your trained model file

## Quick Start

### 1. Set Up GCP Project

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

### 2. Prepare Your Model

```bash
# Place your model in the models directory
cp /path/to/your/model.pkl models/model.pkl
```

### 3. Deploy

```bash
# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export PROJECT_NAME="my-ml-model"
export ENVIRONMENT="prod"
export GCP_REGION="us-central1"

# Run deployment script
chmod +x deploy.sh
./deploy.sh
```

### 4. Test Your Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ${PROJECT_NAME}-${ENVIRONMENT} \
    --region ${GCP_REGION} \
    --format 'value(status.url)')

# Health check
curl $SERVICE_URL/health

# Make a prediction
curl -X POST $SERVICE_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Configuration

### Container Resources

Edit `terraform/variables.tf`:

```hcl
cpu    = "1"      # Number of CPUs (0.5, 1, 2, 4)
memory = "2Gi"    # Memory (512Mi, 1Gi, 2Gi, 4Gi, 8Gi)
timeout = 300     # Request timeout in seconds
```

### Auto-scaling

```hcl
min_instances = 0    # Minimum instances (0 for scale-to-zero)
max_instances = 100  # Maximum instances
```

### Model Configuration

Set environment variables in `terraform/main.tf` or pass via deploy script:

```bash
export MODEL_TYPE="sklearn"     # sklearn, tensorflow, or pytorch
export MODEL_VERSION="1.0.0"
export LOG_LEVEL="INFO"
```

## Manual Deployment

### Using Docker + gcloud

```bash
# Build Docker image
docker build -t gcr.io/${GCP_PROJECT_ID}/ml-model:latest .

# Push to Container Registry
docker push gcr.io/${GCP_PROJECT_ID}/ml-model:latest

# Deploy to Cloud Run
gcloud run deploy ml-model \
    --image gcr.io/${GCP_PROJECT_ID}/ml-model:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1
```

### Using Cloud Build

```bash
# Build with Cloud Build
gcloud builds submit \
    --tag gcr.io/${GCP_PROJECT_ID}/ml-model:latest

# Deploy
gcloud run deploy ml-model \
    --image gcr.io/${GCP_PROJECT_ID}/ml-model:latest \
    --platform managed \
    --region us-central1
```

## Project Structure

```
gcp-cloud-run/
├── Dockerfile              # Container definition
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── deploy.sh             # Deployment script
├── README.md             # This file
└── terraform/
    ├── main.tf           # Main Terraform configuration
    ├── variables.tf      # Variable definitions
    └── outputs.tf        # Output definitions
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

### Model Info

```bash
GET /info
```

### Metrics

```bash
GET /metrics
```

## Monitoring

### Cloud Logging

View logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
    resource.labels.service_name=ml-model-prod" \
    --limit 50 \
    --format json
```

Stream logs:
```bash
gcloud run services logs tail ml-model-prod --region us-central1
```

### Cloud Monitoring

Monitor metrics in GCP Console:
- Request count
- Request latency
- Instance count
- Memory usage
- CPU utilization

### Custom Alerts

The Terraform template creates alerts for:
- Error rate > threshold
- Latency > threshold

Configure notification channels:
```hcl
notification_channels = ["projects/PROJECT/notificationChannels/CHANNEL_ID"]
```

## Performance Optimization

### Reduce Cold Start Time

1. **Use smaller base image**:
   ```dockerfile
   FROM python:3.11-slim
   ```

2. **Minimize dependencies**:
   - Only include necessary packages
   - Use pre-built wheels when possible

3. **Enable CPU boost**:
   ```hcl
   startup_cpu_boost = true
   ```

4. **Keep minimum instances > 0**:
   ```hcl
   min_instances = 1  # Eliminates cold starts
   ```

### Optimize Memory Usage

1. **Right-size memory allocation**
2. **Use model compression** (quantization, pruning)
3. **Load model lazily** if possible

## Cost Optimization

Cloud Run pricing (us-central1):
- CPU: $0.00002400 per vCPU-second
- Memory: $0.00000250 per GiB-second
- Requests: $0.40 per million requests
- Free tier: 2 million requests/month

Tips:
1. Use scale-to-zero (`min_instances = 0`)
2. Right-size CPU and memory
3. Optimize request handling time
4. Use request batching for bulk predictions

Estimated monthly cost for moderate traffic:
- 1M requests/month: ~$10-20
- 10M requests/month: ~$50-100

## Security Best Practices

### 1. Authentication

Require authentication:
```hcl
allow_unauthenticated = false
```

Access with service account:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    $SERVICE_URL/predict
```

### 2. VPC Integration

Deploy to VPC:
```hcl
vpc_connector = google_vpc_access_connector.connector.id
```

### 3. Secret Management

Use Secret Manager:
```hcl
env {
  name = "API_KEY"
  value_source {
    secret_key_ref {
      secret  = "api-key"
      version = "latest"
    }
  }
}
```

### 4. Binary Authorization

Enforce signed images:
```bash
gcloud run services update SERVICE_NAME \
    --binary-authorization=default
```

## Troubleshooting

### Issue: Container fails to start

**Check logs**:
```bash
gcloud run services logs read SERVICE_NAME --region REGION
```

**Common causes**:
- Missing dependencies in requirements.txt
- Model file not found
- Port not set to 8080

### Issue: Out of memory

**Solution**: Increase memory allocation:
```hcl
memory = "4Gi"
```

### Issue: Request timeout

**Solution**: Increase timeout:
```hcl
timeout = 600  # seconds
```

### Issue: High cold start latency

**Solutions**:
1. Set `min_instances = 1`
2. Optimize Docker image size
3. Enable CPU boost
4. Use lighter model format

### Issue: Authentication errors

**Solution**: Check IAM permissions:
```bash
gcloud run services get-iam-policy SERVICE_NAME --region REGION
```

## Advanced Features

### Traffic Splitting

Deploy new version with traffic split:
```bash
gcloud run services update-traffic SERVICE_NAME \
    --to-revisions=REVISION1=50,REVISION2=50
```

### GPU Support

Cloud Run doesn't support GPUs. For GPU inference:
- Use GKE (Google Kubernetes Engine)
- Use Vertex AI Prediction

### Model Versioning

Deploy multiple versions:
```bash
# Deploy v1
gcloud run deploy ml-model-v1 --image IMAGE:v1

# Deploy v2
gcloud run deploy ml-model-v2 --image IMAGE:v2
```

### Load from Cloud Storage

Modify `main.py` to download model:
```python
from google.cloud import storage

def download_model():
    client = storage.Client()
    bucket = client.bucket('bucket-name')
    blob = bucket.blob('models/model.pkl')
    blob.download_to_filename('/tmp/model.pkl')
```

## Clean Up

Delete all resources:

```bash
# With Terraform
cd terraform
terraform destroy

# With gcloud
gcloud run services delete SERVICE_NAME --region REGION
```

## Next Steps

- Set up CI/CD pipeline
- Implement model versioning
- Add request caching
- Configure custom domain
- Set up monitoring dashboards
- Implement A/B testing

## Support

For issues and questions:
- GitHub Issues: https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues
- Documentation: ../../docs/gcp-cloud-run.md
- GCP Cloud Run Docs: https://cloud.google.com/run/docs

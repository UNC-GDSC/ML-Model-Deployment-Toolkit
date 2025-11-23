# Getting Started with ML Model Deployment Toolkit

This guide will help you deploy your first ML model using the toolkit.

## Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- A trained ML model (scikit-learn, TensorFlow, or PyTorch)
- An account on your chosen platform (AWS, GCP, or Vercel)
- Basic familiarity with command line tools

## Installation

### Using pip (Recommended)

```bash
pip install ml-deployment-toolkit
```

### From Source

```bash
git clone https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit.git
cd ML-Model-Deployment-Toolkit
pip install -e .
```

## Quick Start

### 1. Train a Sample Model

Let's start by training a simple model:

```bash
cd examples/sklearn
python train_model.py
```

This creates a Random Forest classifier and saves it to `models/sklearn_model.pkl`.

### 2. Test Locally

Verify the model works:

```bash
python test_prediction.py
```

You should see successful predictions and health check results.

### 3. Choose Your Platform

Each platform has different strengths:

| Platform | Best For | Cost | Setup Difficulty |
|----------|----------|------|------------------|
| AWS Lambda | Variable workloads, cost-effective | Low | Medium |
| GCP Cloud Run | Auto-scaling, containers | Medium | Medium |
| Vercel | Lightweight models, global edge | Very Low | Easy |

### 4. Deploy Your Model

#### Option A: Using CLI (Recommended)

```bash
# AWS Lambda
ml-deploy deploy aws-lambda \
    --model-path examples/sklearn/models/sklearn_model.pkl \
    --name my-ml-model \
    --model-type sklearn

# GCP Cloud Run
ml-deploy deploy gcp-cloud-run \
    --model-path examples/sklearn/models/sklearn_model.pkl \
    --name my-ml-model \
    --model-type sklearn

# Vercel
ml-deploy deploy vercel \
    --model-path examples/sklearn/models/sklearn_model.pkl \
    --name my-ml-model \
    --model-type sklearn
```

#### Option B: Using Deployment Scripts

```bash
# AWS Lambda
cd templates/aws-lambda
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
./deploy.sh

# GCP Cloud Run
cd templates/gcp-cloud-run
export GCP_PROJECT_ID="your-project-id"
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
./deploy.sh

# Vercel
cd templates/vercel
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
./deploy.sh
```

### 5. Test Your Deployment

Once deployed, test your endpoint:

```bash
# Replace with your actual endpoint URL
ENDPOINT_URL="https://your-deployment-url"

# Health check
curl $ENDPOINT_URL/health

# Make a prediction
curl -X POST $ENDPOINT_URL/predict \
    -H "Content-Type: application/json" \
    -d '{
        "features": [0.5, -1.2, 0.8, -0.3, 1.1, 0.2, -0.9, 0.4, -0.6, 0.7,
                     0.1, -0.4, 0.9, -1.0, 0.3, -0.2, 0.6, -0.8, 0.5, -0.1],
        "return_probabilities": true
    }'
```

Expected response:

```json
{
    "prediction": 1,
    "probabilities": [0.23, 0.77],
    "model_version": "1.0.0",
    "timestamp": "2024-01-01T00:00:00.000Z",
    "latency_ms": 45.2
}
```

## Deploying Your Own Model

### Step 1: Prepare Your Model

Train and save your model:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'my_model.pkl')
```

### Step 2: Validate Your Model

```bash
ml-deploy validate my_model.pkl
```

This checks:
- File exists and is readable
- File size is compatible with platforms
- Model can be loaded

### Step 3: Choose Configuration

Create a configuration file `config.yaml`:

```yaml
model:
  path: my_model.pkl
  version: "1.0.0"
  type: sklearn

deployment:
  platform: aws-lambda  # or gcp-cloud-run, vercel
  name: my-model
  region: us-east-1

resources:
  memory: 1024  # MB
  timeout: 30   # seconds
```

### Step 4: Deploy

```bash
ml-deploy deploy aws-lambda \
    --model-path my_model.pkl \
    --name my-model \
    --model-type sklearn \
    --memory 1024 \
    --timeout 30
```

## Platform-Specific Guides

- [AWS Lambda Deployment](aws-lambda.md)
- [GCP Cloud Run Deployment](gcp-cloud-run.md)
- [Vercel Deployment](vercel.md)

## Common Issues

### Model Too Large

**Problem**: Model file exceeds platform limits

**Solutions**:
1. Use model compression
2. Switch to a platform with higher limits (GCP Cloud Run)
3. Load model from external storage (S3, GCS)
4. Use model quantization or pruning

### Slow Cold Starts

**Problem**: First request takes too long

**Solutions**:
1. Keep minimum instances > 0 (costs more)
2. Optimize model loading code
3. Use lighter model formats (ONNX)
4. Enable provisioned concurrency

### Prediction Timeouts

**Problem**: Predictions exceed timeout limit

**Solutions**:
1. Increase timeout setting
2. Optimize model inference
3. Use batch processing
4. Consider using a smaller model

## Next Steps

1. **Add Monitoring**: Set up logging and metrics
2. **Implement Security**: Add API authentication
3. **Enable CI/CD**: Automate deployments
4. **Optimize Performance**: Profile and optimize
5. **Scale Up**: Configure auto-scaling policies

## Getting Help

- 📖 [Full Documentation](../README.md)
- 💬 [Discord Community](https://discord.gg/ml-deploy)
- 🐛 [Report Issues](https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues)
- 📧 Email: support@ml-deployment-toolkit.com

## Additional Resources

- [Model Preparation Guide](model-preparation.md)
- [Security Best Practices](security.md)
- [Monitoring & Logging](monitoring.md)
- [API Reference](api-reference.md)
- [Troubleshooting](troubleshooting.md)

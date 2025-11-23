# Vercel Deployment Template

Deploy lightweight ML models to Vercel's edge network with this serverless template.

## Features

- ⚡ Edge network deployment for low latency
- 🌍 Global CDN distribution
- 💰 Generous free tier
- 🚀 Instant deployments
- 📊 Built-in analytics
- 🔄 Automatic HTTPS and SSL

## Important Limitations

Vercel has specific constraints that make it suitable for **lightweight models only**:

- **Size limit**: 50MB per serverless function
- **Execution time**: 10 seconds max (Hobby), 60 seconds (Pro)
- **Memory**: 1024MB max
- **Best for**: Small sklearn models, lightweight ONNX models

**Not recommended for**: TensorFlow, PyTorch, or large models (>50MB)

## Prerequisites

- Node.js and npm installed
- Vercel CLI: `npm install -g vercel`
- Vercel account (free tier available)
- **Small** trained model file (<50MB)

## Quick Start

### 1. Install Vercel CLI

```bash
npm install -g vercel
vercel login
```

### 2. Prepare Your Model

**Important**: Your model must be <50MB

```bash
# Check model size
ls -lh /path/to/model.pkl

# If model is small enough, copy it
cp /path/to/model.pkl models/model.pkl
```

### 3. Configure Project

Edit `vercel.json`:

```json
{
  "env": {
    "MODEL_VERSION": "1.0.0",
    "MODEL_TYPE": "sklearn"
  }
}
```

### 4. Deploy

```bash
# Deploy to preview
chmod +x deploy.sh
./deploy.sh

# Deploy to production
./deploy.sh production
```

Or manually:

```bash
# Copy source code
cp -r ../../src .

# Preview deployment
vercel

# Production deployment
vercel --prod
```

### 5. Test Your Deployment

```bash
# Get your deployment URL from Vercel output
VERCEL_URL="https://your-deployment.vercel.app"

# Health check
curl $VERCEL_URL/health

# Make a prediction
curl -X POST $VERCEL_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Project Structure

```
vercel/
├── api/
│   ├── predict.py         # Prediction endpoint
│   ├── health.py          # Health check endpoint
│   └── info.py            # Model info endpoint
├── models/
│   └── model.pkl          # Your trained model (<50MB)
├── vercel.json            # Vercel configuration
├── requirements.txt       # Python dependencies
├── .vercelignore         # Files to ignore
├── deploy.sh             # Deployment script
└── README.md             # This file
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
  "timestamp": "2024-01-01T00:00:00.000Z",
  "service": "ml-model-vercel"
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

## Configuration

### Environment Variables

Set in `vercel.json` or via Vercel dashboard:

```json
{
  "env": {
    "MODEL_VERSION": "1.0.0",
    "MODEL_TYPE": "sklearn",
    "LOG_LEVEL": "INFO",
    "API_KEY": "@api-key-secret"
  }
}
```

Add secrets via CLI:
```bash
vercel secrets add api-key-secret "your-secret-value"
```

### Function Configuration

Edit `vercel.json`:

```json
{
  "functions": {
    "api/*.py": {
      "memory": 1024,
      "maxDuration": 10
    }
  }
}
```

### Regions

Specify deployment regions:

```json
{
  "regions": ["iad1", "sfo1", "lhr1"]
}
```

Available regions:
- `iad1` - Washington, D.C., USA
- `sfo1` - San Francisco, USA
- `lhr1` - London, UK
- And more...

## Handling Large Models

If your model exceeds 50MB, you have several options:

### Option 1: Model Compression

```python
# Use sklearn's compression
import joblib
joblib.dump(model, 'model.pkl', compress=3)

# Or use pickle with optimization
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### Option 2: External Storage

Load model from S3/GCS at runtime:

```python
# In api/predict.py
import boto3
import tempfile

def download_model():
    s3 = boto3.client('s3')
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_fileobj('bucket-name', 'model.pkl', tmp)
        return tmp.name

model_path = download_model()
MODEL = SklearnModel(model_path, '1.0.0')
```

### Option 3: Use ONNX

Convert to ONNX for smaller size:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, n_features]))]
onx = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

## Monitoring

### Vercel Dashboard

Monitor your deployment:
- Real-time logs
- Performance metrics
- Error tracking
- Deployment history

### View Logs

```bash
vercel logs <deployment-url>
```

### Analytics

Vercel provides built-in analytics:
- Request count
- Response times
- Error rates
- Geographic distribution

## Performance Optimization

### 1. Minimize Dependencies

Only include necessary packages in `requirements.txt`:
```txt
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
```

### 2. Cold Start Optimization

- Keep model loading code efficient
- Use global variables for model caching
- Minimize import statements

### 3. Response Time

- Use smaller models when possible
- Optimize preprocessing steps
- Consider model quantization

## Cost

### Free (Hobby) Tier

- 100GB bandwidth/month
- 100 hours serverless execution/month
- Unlimited deployments
- Automatic SSL

### Pro Tier ($20/month)

- 1TB bandwidth
- 1000 hours execution
- Advanced analytics
- Team collaboration

Most ML inference workloads fit within the free tier!

## Security

### API Authentication

Add authentication to `api/predict.py`:

```python
def handler(request):
    # Check API key
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
    if api_key != os.getenv('API_KEY'):
        return {
            'statusCode': 401,
            'body': json.dumps({'error': 'Unauthorized'})
        }
    # ... rest of handler
```

### CORS Configuration

Already configured in responses:
```python
'Access-Control-Allow-Origin': '*'
```

For production, specify allowed origins:
```python
'Access-Control-Allow-Origin': 'https://yourdomain.com'
```

### Environment Variables

Never commit secrets:
```bash
# Add as Vercel secrets
vercel secrets add api-key "your-secret-key"

# Reference in vercel.json
{
  "env": {
    "API_KEY": "@api-key"
  }
}
```

## Troubleshooting

### Issue: Function size exceeds limit

**Solution**:
1. Remove unused dependencies
2. Use external model storage
3. Compress model file
4. Consider GCP Cloud Run instead

### Issue: Timeout after 10 seconds

**Solution**:
1. Upgrade to Pro for 60s timeout
2. Optimize model inference
3. Use lighter model
4. Consider AWS Lambda instead

### Issue: Out of memory

**Solution**:
1. Use smaller model
2. Reduce batch size
3. Optimize memory usage
4. Consider platform with more memory

### Issue: Cold start latency

**Expected behavior** on Vercel. Mitigate by:
1. Keep functions warm with periodic pings
2. Optimize model loading
3. Minimize dependencies

## Alternatives for Large Models

If Vercel doesn't fit your needs:

- **AWS Lambda**: Up to 10GB with container images
- **GCP Cloud Run**: Up to 32GB memory, no size limit
- **Azure Functions**: Up to 1.5GB
- **Self-hosted**: Unlimited resources

## Custom Domains

Add custom domain via Vercel dashboard or CLI:

```bash
vercel domains add yourdomain.com
```

Configure DNS records as instructed by Vercel.

## CI/CD Integration

### GitHub Integration

1. Push code to GitHub
2. Link repository in Vercel dashboard
3. Automatic deployments on push

### Manual CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Vercel
        run: |
          npm install -g vercel
          vercel --token ${{ secrets.VERCEL_TOKEN }} --prod
```

## Best Practices

1. ✅ Use lightweight models (<50MB)
2. ✅ Minimize dependencies
3. ✅ Cache model in global scope
4. ✅ Use external storage for large models
5. ✅ Monitor performance metrics
6. ✅ Set up proper error handling
7. ✅ Use environment variables for config
8. ✅ Enable CORS appropriately
9. ✅ Implement API authentication
10. ✅ Test locally before deploying

## Local Development

Test functions locally:

```bash
vercel dev
```

This starts a local server at `http://localhost:3000`

## Clean Up

Delete deployment:

```bash
vercel remove <deployment-name>
```

## Next Steps

- Add request rate limiting
- Implement caching
- Set up monitoring alerts
- Add model versioning
- Create custom domain
- Implement A/B testing

## Support

For issues and questions:
- GitHub Issues: https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues
- Documentation: ../../docs/vercel.md
- Vercel Docs: https://vercel.com/docs

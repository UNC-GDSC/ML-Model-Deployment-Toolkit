# ML Model Deployment Examples

This directory contains comprehensive examples demonstrating how to deploy various types of ML models using the ML Model Deployment Toolkit.

## Directory Structure

```
examples/
├── computer_vision/
│   ├── image_classification.py      # Image classification with ResNet
│   ├── object_detection.py          # Object detection with YOLO
│   └── image_segmentation.py        # Semantic segmentation
├── nlp/
│   ├── text_classification.py       # Sentiment analysis and topic classification
│   ├── named_entity_recognition.py  # NER for entity extraction
│   └── text_generation.py           # Text generation models
├── tabular/
│   ├── regression.py                # Regression models
│   └── classification.py            # Classification models
└── README.md                        # This file
```

## Computer Vision Examples

### Image Classification

Demonstrates deploying image classification models using pre-trained networks like ResNet, EfficientNet, or custom models.

**Use Cases:**
- Product categorization
- Quality control inspection
- Medical image diagnosis
- Content moderation

**Run Example:**
```bash
python examples/computer_vision/image_classification.py
```

**Key Features:**
- Pre-trained ImageNet models
- Support for TensorFlow and PyTorch
- Batch prediction support
- Easy deployment to all platforms

### Object Detection

Shows how to deploy object detection models for detecting and localizing multiple objects in images.

**Use Cases:**
- Autonomous vehicles
- Security surveillance
- Retail analytics
- Wildlife monitoring

**Run Example:**
```bash
python examples/computer_vision/object_detection.py
```

**Key Features:**
- YOLO model integration
- Bounding box visualization
- Real-time inference
- Multi-object detection

## NLP Examples

### Text Classification

Demonstrates sentiment analysis and topic classification using transformer models.

**Use Cases:**
- Customer feedback analysis
- Social media monitoring
- Content categorization
- Spam detection

**Run Example:**
```bash
python examples/nlp/text_classification.py
```

**Key Features:**
- Pre-trained transformers (BERT, DistilBERT)
- Multi-class classification
- Batch processing
- Fine-tuning support

### Named Entity Recognition

Shows how to extract entities like persons, organizations, and locations from text.

**Use Cases:**
- Information extraction
- Document analysis
- Privacy compliance (PII detection)
- Knowledge graph construction

**Run Example:**
```bash
python examples/nlp/named_entity_recognition.py
```

**Key Features:**
- Entity extraction (PER, ORG, LOC, MISC)
- Confidence scores
- Entity grouping
- Real-time processing

## Prerequisites

### Base Requirements
```bash
pip install -r requirements.txt
```

### Computer Vision Requirements
```bash
# For TensorFlow models
pip install tensorflow>=2.13.0

# For PyTorch models
pip install torch>=2.0.0 torchvision>=0.15.0

# For YOLO
pip install ultralytics
```

### NLP Requirements
```bash
# For transformer models
pip install transformers>=4.30.0

# For tokenization
pip install tokenizers>=0.13.0
```

## Running Examples

### Local Testing

All examples can be run locally for testing:

```bash
# Image classification
python examples/computer_vision/image_classification.py

# Object detection
python examples/computer_vision/object_detection.py

# Text classification
python examples/nlp/text_classification.py

# NER
python examples/nlp/named_entity_recognition.py
```

### Deployment

Each example includes deployment instructions for all supported platforms:

#### AWS Lambda
```bash
ml-deploy aws-lambda \
  --model-path models/my_model.pkl \
  --name my-model \
  --memory 3008 \
  --timeout 60
```

#### AWS SageMaker
```bash
cd templates/aws-sagemaker
./deploy.sh my-model prod us-east-1 ml.m5.large
```

#### GCP Cloud Run
```bash
cd templates/gcp-cloud-run
gcloud run deploy my-model --source . --region us-central1
```

#### Kubernetes
```bash
cd templates/kubernetes
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

#### Azure Functions
```bash
cd templates/azure-functions
func azure functionapp publish my-model-app
```

#### Vercel
```bash
cd templates/vercel
vercel deploy --prod
```

## Performance Benchmarks

### Image Classification (ResNet50)

| Platform | Cold Start | Warm Latency | Cost/1M requests |
|----------|-----------|--------------|------------------|
| AWS Lambda | ~3s | 200-300ms | $20 |
| SageMaker | N/A | 50-100ms | $50 |
| Cloud Run | ~2s | 150-250ms | $25 |
| Kubernetes | N/A | 80-150ms | $40 |

### Text Classification (DistilBERT)

| Platform | Cold Start | Warm Latency | Cost/1M requests |
|----------|-----------|--------------|------------------|
| AWS Lambda | ~5s | 100-200ms | $30 |
| SageMaker | N/A | 30-80ms | $60 |
| Cloud Run | ~3s | 80-150ms | $35 |
| Kubernetes | N/A | 50-100ms | $45 |

*Note: Benchmarks are approximate and vary based on model size and configuration.*

## Model Optimization

All examples support model optimization techniques:

### Quantization
```python
from src.optimization.model_optimization import ModelQuantizer

quantizer = ModelQuantizer(model, 'tensorflow')
quantizer.quantize_tensorflow('model_quantized.tflite', 'OPTIMIZE_FOR_SIZE')
```

### Pruning
```python
from src.optimization.model_optimization import ModelPruner

pruner = ModelPruner(model, 'pytorch')
pruner.prune_pytorch(amount=0.5, output_path='model_pruned.pth')
```

### ONNX Conversion
```python
from src.optimization.model_optimization import ModelConverter

converter = ModelConverter()
converter.pytorch_to_onnx(
    pytorch_model,
    dummy_input,
    'model.onnx'
)
```

## Monitoring and Logging

All examples include built-in monitoring:

### Metrics Tracked
- Request rate
- Latency (p50, p95, p99)
- Error rate
- Model performance metrics
- Resource utilization

### Logging
```python
# Structured logging included in all examples
{
    "timestamp": "2024-01-15T10:30:00Z",
    "model_version": "1.0.0",
    "latency_ms": 145.2,
    "prediction": "...",
    "confidence": 0.95
}
```

## Testing

Run tests for all examples:

```bash
# Unit tests
pytest tests/examples/

# Integration tests
pytest tests/integration/

# Load tests
locust -f tests/load/locustfile.py
```

## Best Practices

1. **Model Selection**
   - Start with pre-trained models
   - Fine-tune on your specific dataset
   - Consider model size vs. accuracy tradeoffs

2. **Optimization**
   - Quantize models for production
   - Use ONNX for cross-framework compatibility
   - Enable batch processing when possible

3. **Deployment**
   - Test locally first
   - Use staging environment
   - Monitor performance metrics
   - Set up alerting for errors

4. **Security**
   - Validate all inputs
   - Use API authentication
   - Rate limit endpoints
   - Sanitize outputs

## Troubleshooting

### Common Issues

**Model Loading Errors**
```python
# Ensure dependencies are installed
pip install -r requirements.txt

# Check model path
assert os.path.exists(model_path)
```

**Memory Issues**
```python
# Increase Lambda memory
--memory 3008

# Use smaller model variant
model = 'distilbert-base-uncased'  # instead of bert-large
```

**Slow Inference**
```python
# Enable GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use batch processing
model.predict_batch(texts)

# Optimize model
quantizer.quantize_pytorch(model, 'model_optimized.pth')
```

## Additional Resources

- [Model Optimization Guide](../docs/optimization.md)
- [Deployment Best Practices](../docs/deployment.md)
- [Monitoring and Observability](../docs/monitoring.md)
- [Security Guidelines](../docs/security.md)

## Contributing

We welcome contributions! To add a new example:

1. Create a new file in the appropriate directory
2. Follow the existing example structure
3. Include comprehensive documentation
4. Add unit tests
5. Update this README
6. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing examples
- Review documentation
- Contact maintainers

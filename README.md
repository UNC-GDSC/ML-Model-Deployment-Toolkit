# ML Model Deployment Toolkit

A comprehensive, production-ready toolkit for deploying machine learning models to AWS Lambda, AWS SageMaker, GCP Cloud Run, Azure Functions, Kubernetes, and Vercel. This toolkit provides everything you need to deploy ML models at scale with best practices built-in.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🚀 **Multi-Platform Support**: Deploy to AWS Lambda, AWS SageMaker, GCP Cloud Run, Azure Functions, Kubernetes, or Vercel
- 🔧 **Framework Agnostic**: Works with TensorFlow, PyTorch, scikit-learn, and more
- 📦 **Production Ready**: Includes monitoring, logging, error handling, and security
- 🐳 **Containerized**: Docker support for consistent deployments
- 🏗️ **Infrastructure as Code**: Terraform templates for AWS and GCP
- 🧪 **Testing**: Comprehensive test suites and examples
- 📊 **Monitoring**: Built-in health checks and performance monitoring
- 🔐 **Security**: API authentication, rate limiting, and input validation
- 📖 **Well Documented**: Extensive documentation and examples

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit.git
cd ML-Model-Deployment-Toolkit

# Install dependencies
pip install -r requirements.txt

# Install the CLI tool
pip install -e .
```

### Deploy Your First Model

```bash
# Deploy to AWS Lambda
ml-deploy aws-lambda --model-path ./models/my_model.pkl --name my-model

# Deploy to GCP Cloud Run
ml-deploy gcp-cloud-run --model-path ./models/my_model.pkl --name my-model

# Deploy to Vercel
ml-deploy vercel --model-path ./models/my_model.pkl --name my-model
```

## Architecture

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │
         ├─────────────┬──────────────┬──────────────┐
         │             │              │              │
    ┌────▼─────┐  ┌───▼────┐    ┌────▼──────┐  ┌───▼────┐
    │   AWS    │  │  GCP   │    │  Vercel   │  │ Custom │
    │  Lambda  │  │ Cloud  │    │ Serverless│  │Backend │
    │          │  │  Run   │    │           │  │        │
    └────┬─────┘  └───┬────┘    └────┬──────┘  └───┬────┘
         │            │              │             │
         └────────────┴──────────────┴─────────────┘
                           │
                    ┌──────▼──────┐
                    │  ML Model   │
                    │  (TF/PyTorch│
                    │  /sklearn)  │
                    └─────────────┘
```

## Supported Platforms

### AWS Lambda
- Serverless deployment with auto-scaling
- API Gateway integration
- CloudWatch logging and monitoring
- Terraform infrastructure templates
- Cost-effective for variable workloads

### GCP Cloud Run
- Fully managed container platform
- Auto-scaling from zero to N
- Cloud Logging integration
- Terraform infrastructure templates
- Great for containerized models

### Vercel
- Edge network deployment
- Serverless functions
- Built-in CDN
- Perfect for lightweight models
- Simple deployment workflow

### AWS SageMaker
- Fully managed ML infrastructure
- Auto-scaling with target tracking
- Built-in A/B testing support
- Model monitoring and data capture
- Multi-model endpoints

### Kubernetes
- Self-hosted or cloud-managed clusters
- Horizontal pod auto-scaling
- Rolling updates and canary deployments
- Service mesh integration (Istio)
- Maximum control and flexibility

### Azure Functions
- Serverless compute on Azure
- Multiple trigger types (HTTP, queue, blob)
- Azure Monitor integration
- Easy integration with Azure ecosystem
- Cost-effective for event-driven workloads

## Project Structure

```
ML-Model-Deployment-Toolkit/
├── src/
│   ├── core/              # Core utilities and base classes
│   ├── handlers/          # Platform-specific request handlers
│   ├── models/            # Model wrapper classes
│   └── utils/             # Utility functions
├── templates/
│   ├── aws-lambda/        # AWS Lambda deployment templates
│   ├── gcp-cloud-run/     # GCP Cloud Run deployment templates
│   └── vercel/            # Vercel deployment templates
├── examples/
│   ├── sklearn/           # scikit-learn model examples
│   ├── tensorflow/        # TensorFlow model examples
│   └── pytorch/           # PyTorch model examples
├── tests/                 # Comprehensive test suites
├── docs/                  # Detailed documentation
└── cli/                   # Command-line deployment tool
```

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [AWS Lambda Deployment](docs/aws-lambda.md)
- [GCP Cloud Run Deployment](docs/gcp-cloud-run.md)
- [Vercel Deployment](docs/vercel.md)
- [Model Preparation](docs/model-preparation.md)
- [Security Best Practices](docs/security.md)
- [Monitoring & Logging](docs/monitoring.md)
- [API Reference](docs/api-reference.md)

## Examples

### Deploying a scikit-learn Model

```python
from ml_deploy import ModelDeployer
import joblib

# Load your trained model
model = joblib.load('my_model.pkl')

# Deploy to AWS Lambda
deployer = ModelDeployer(platform='aws-lambda')
endpoint = deployer.deploy(
    model=model,
    name='my-sklearn-model',
    requirements=['scikit-learn==1.3.0', 'numpy==1.24.3']
)

print(f"Model deployed at: {endpoint}")
```

### Making Predictions

```python
import requests

response = requests.post(
    'https://your-endpoint.com/predict',
    json={'features': [1.0, 2.0, 3.0, 4.0]},
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

prediction = response.json()
print(prediction)
```

## Requirements

- Python 3.8+
- Docker (for containerized deployments)
- Terraform (for infrastructure provisioning)
- AWS CLI (for AWS deployments)
- gcloud CLI (for GCP deployments)
- Vercel CLI (for Vercel deployments)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@ml-deployment-toolkit.com
- 💬 Discord: [Join our community](https://discord.gg/ml-deploy)
- 🐛 Issues: [GitHub Issues](https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues)

## Acknowledgments

- Built with ❤️ by the UNC Google Developer Student Club
- Inspired by the amazing ML and DevOps communities
- Special thanks to all contributors

---

**Star this repo if you find it helpful!** ⭐

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-01

### Added

#### Core Features
- Initial release of ML Model Deployment Toolkit
- Support for scikit-learn, TensorFlow, and PyTorch models
- Base model abstraction with preprocessing and postprocessing
- Comprehensive configuration management
- Structured logging with JSON support
- Metrics collection with Prometheus support

#### Deployment Platforms
- **AWS Lambda**: Complete deployment template with Terraform
  - Lambda function with configurable memory and timeout
  - API Gateway or Function URL support
  - S3 bucket for model storage
  - CloudWatch logging and monitoring
  - IAM roles and policies

- **GCP Cloud Run**: Containerized deployment template
  - Fully managed auto-scaling
  - Cloud Logging integration
  - Artifact Registry support
  - Terraform infrastructure templates
  - Health checks and probes

- **Vercel**: Serverless function deployment
  - Edge network deployment
  - Automatic HTTPS
  - Simple configuration
  - Global CDN distribution

#### CLI Tool
- `ml-deploy` command-line interface
- Deploy command for all platforms
- Model validation
- Deployment info and management
- Platform listing

#### Examples
- scikit-learn Random Forest classifier
- Training and testing scripts
- Platform-specific deployment examples

#### Testing
- Comprehensive test suite
- Unit tests for models and handlers
- Integration tests
- pytest configuration
- Code coverage reporting

#### CI/CD
- GitHub Actions workflows
- Automated testing on push
- Multi-platform deployment
- PyPI publishing

#### Documentation
- Comprehensive README
- Platform-specific deployment guides
- Getting started guide
- Contributing guidelines
- API documentation
- Security best practices

#### Development Tools
- Docker and Docker Compose configuration
- Makefile for common tasks
- Pre-commit hooks
- Development environment setup
- Example .env file

### Security
- API key authentication support
- Rate limiting
- CORS configuration
- Input validation
- Secure environment variable handling

### Performance
- Model caching for reduced latency
- Optimized cold start times
- Efficient preprocessing pipelines
- Batch prediction support

## [Unreleased]

### Planned Features
- Support for ONNX models
- Model versioning and A/B testing
- Request batching
- Model monitoring dashboards
- Automated model retraining
- Multi-model deployments
- Custom preprocessing pipelines
- Model explainability integration

---

For older versions and detailed changes, see the [releases page](https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/releases).

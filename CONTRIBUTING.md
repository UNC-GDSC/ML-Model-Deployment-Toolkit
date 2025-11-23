# Contributing to ML Model Deployment Toolkit

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Code snippets or error messages

### Suggesting Features

1. Check existing feature requests in Issues
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Update documentation as needed
7. Commit with clear messages
8. Push to your fork
9. Create a Pull Request

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Docker (for testing containers)

### Installation

```bash
# Clone the repository
git clone https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit.git
cd ML-Model-Deployment-Toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Maximum line length: 100 characters
- Use type hints where possible
- Write docstrings for all public functions and classes

### Formatting

We use:
- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

Run formatters:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

Example test:

```python
import pytest
from src.models.sklearn_model import SklearnModel

def test_model_loading(sample_model_path):
    """Test that model loads correctly."""
    model = SklearnModel(sample_model_path, "1.0.0")
    model.load()

    assert model.loaded
    assert model.model is not None
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def deploy_model(platform: str, model_path: str) -> str:
    """
    Deploy a model to the specified platform.

    Args:
        platform: Target deployment platform (aws-lambda, gcp-cloud-run, vercel)
        model_path: Path to the model file

    Returns:
        Deployment endpoint URL

    Raises:
        ValueError: If platform is not supported
        FileNotFoundError: If model file doesn't exist

    Example:
        >>> endpoint = deploy_model('aws-lambda', 'models/model.pkl')
        >>> print(endpoint)
        'https://abc123.lambda-url.us-east-1.on.aws/'
    """
    pass
```

### Updating Documentation

- Update README.md for user-facing changes
- Update docs/ for detailed documentation
- Update CHANGELOG.md for notable changes
- Add examples for new features

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for PyTorch models
fix: resolve memory leak in model loading
docs: update AWS Lambda deployment guide
test: add tests for GCP Cloud Run handler
refactor: simplify config validation logic
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create and push tag: `git tag v1.x.x && git push --tags`
4. GitHub Actions will automatically create release and publish to PyPI

## Project Structure

```
ML-Model-Deployment-Toolkit/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── models/            # Model wrappers
│   ├── handlers/          # Request handlers
│   └── utils/             # Utilities
├── templates/             # Deployment templates
│   ├── aws-lambda/
│   ├── gcp-cloud-run/
│   └── vercel/
├── examples/              # Example models
├── tests/                 # Test suite
├── docs/                  # Documentation
└── cli/                   # CLI tool
```

## Questions?

- Open an issue for questions
- Join our Discord community
- Email: gdsc@unc.edu

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉

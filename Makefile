.PHONY: help install test lint format clean docker-build docker-run deploy-aws deploy-gcp deploy-vercel docs

help:
	@echo "ML Model Deployment Toolkit - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make docker-test    Run tests in Docker"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-aws     Deploy to AWS Lambda"
	@echo "  make deploy-gcp     Deploy to GCP Cloud Run"
	@echo "  make deploy-vercel  Deploy to Vercel"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	pylint src/

format:
	black src/ tests/ cli/ examples/
	isort src/ tests/ cli/ examples/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .mypy_cache/

docker-build:
	docker-compose build

docker-run:
	docker-compose up dev

docker-test:
	docker-compose run --rm test

docker-down:
	docker-compose down

deploy-aws:
	cd templates/aws-lambda && ./deploy.sh

deploy-gcp:
	cd templates/gcp-cloud-run && ./deploy.sh

deploy-vercel:
	cd templates/vercel && ./deploy.sh

docs:
	mkdocs build

docs-serve:
	mkdocs serve

release:
	python -m build
	twine check dist/*

publish:
	twine upload dist/*

train-example:
	cd examples/sklearn && python train_model.py

validate-model:
	ml-deploy validate examples/sklearn/models/sklearn_model.pkl

all: install-dev lint test

ci: install test lint

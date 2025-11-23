# Multi-stage build for SageMaker custom container
# This Dockerfile creates a custom SageMaker inference container

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
    SAGEMAKER_PROGRAM=inference.py

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create necessary directories
RUN mkdir -p /opt/ml/code /opt/ml/model

# Copy application code
COPY inference.py /opt/ml/code/
COPY src/ /opt/ml/code/src/

# Set working directory
WORKDIR /opt/ml/code

# SageMaker uses these environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
    SAGEMAKER_MODEL_SERVER_TIMEOUT=3600 \
    SAGEMAKER_MODEL_SERVER_WORKERS=1

# Expose port for SageMaker
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Entry point for SageMaker
ENTRYPOINT ["python", "inference.py"]

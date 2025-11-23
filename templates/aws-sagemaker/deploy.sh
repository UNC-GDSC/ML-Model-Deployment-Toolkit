#!/bin/bash

# AWS SageMaker Deployment Script
#
# This script packages and deploys a model to AWS SageMaker.
# It handles model packaging, S3 upload, and Terraform deployment.

set -e

# Configuration
MODEL_NAME="${1:-my-model}"
ENVIRONMENT="${2:-dev}"
AWS_REGION="${3:-us-east-1}"
INSTANCE_TYPE="${4:-ml.t2.medium}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not found. Please install it first."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Package model for SageMaker
package_model() {
    log_info "Packaging model for SageMaker..."

    # Create temporary directory
    PACKAGE_DIR="./sagemaker-package-${MODEL_NAME}"
    mkdir -p "$PACKAGE_DIR"

    # Copy model files
    if [ -d "../../models/${MODEL_NAME}" ]; then
        cp -r "../../models/${MODEL_NAME}"/* "$PACKAGE_DIR/"
    else
        log_warn "Model directory not found at ../../models/${MODEL_NAME}"
        log_info "Creating sample model structure..."
    fi

    # Copy inference code
    cp inference.py "$PACKAGE_DIR/"

    # Copy source code if exists
    if [ -d "../../src" ]; then
        cp -r "../../src" "$PACKAGE_DIR/"
    fi

    # Create metadata
    cat > "$PACKAGE_DIR/metadata.json" << EOF
{
    "model_name": "${MODEL_NAME}",
    "model_type": "sklearn",
    "version": "1.0.0",
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "framework": "scikit-learn"
}
EOF

    # Create tarball
    log_info "Creating model archive..."
    tar -czf "${MODEL_NAME}-model.tar.gz" -C "$PACKAGE_DIR" .

    # Cleanup
    rm -rf "$PACKAGE_DIR"

    log_info "Model packaged successfully: ${MODEL_NAME}-model.tar.gz"
}

# Upload model to S3
upload_to_s3() {
    log_info "Uploading model to S3..."

    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

    # Create S3 bucket if it doesn't exist
    BUCKET_NAME="sagemaker-${AWS_REGION}-${ACCOUNT_ID}"
    if ! aws s3 ls "s3://${BUCKET_NAME}" 2>&1 | grep -q 'NoSuchBucket'; then
        log_info "Using existing S3 bucket: ${BUCKET_NAME}"
    else
        log_info "Creating S3 bucket: ${BUCKET_NAME}"
        aws s3 mb "s3://${BUCKET_NAME}" --region "${AWS_REGION}"
    fi

    # Upload model
    S3_MODEL_PATH="s3://${BUCKET_NAME}/${MODEL_NAME}/model.tar.gz"
    aws s3 cp "${MODEL_NAME}-model.tar.gz" "$S3_MODEL_PATH"

    log_info "Model uploaded to: ${S3_MODEL_PATH}"

    # Export for Terraform
    export TF_VAR_model_data_url="$S3_MODEL_PATH"
}

# Deploy with Terraform
deploy_terraform() {
    log_info "Deploying SageMaker endpoint with Terraform..."

    cd terraform

    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init

    # Create terraform.tfvars
    cat > terraform.tfvars << EOF
model_name                       = "${MODEL_NAME}"
model_data_url                   = "${TF_VAR_model_data_url}"
instance_type                    = "${INSTANCE_TYPE}"
environment                      = "${ENVIRONMENT}"
aws_region                       = "${AWS_REGION}"
auto_scaling_enabled             = true
min_capacity                     = 1
max_capacity                     = 3
target_invocations_per_instance  = 1000
EOF

    # Plan
    log_info "Planning Terraform changes..."
    terraform plan -out=tfplan

    # Apply
    log_info "Applying Terraform configuration..."
    terraform apply tfplan

    # Get outputs
    ENDPOINT_NAME=$(terraform output -raw endpoint_name)
    ENDPOINT_ARN=$(terraform output -raw endpoint_arn)

    cd ..

    log_info "SageMaker endpoint deployed successfully!"
    log_info "Endpoint name: ${ENDPOINT_NAME}"
    log_info "Endpoint ARN: ${ENDPOINT_ARN}"
}

# Test endpoint
test_endpoint() {
    log_info "Testing SageMaker endpoint..."

    cd terraform
    ENDPOINT_NAME=$(terraform output -raw endpoint_name)
    cd ..

    # Create test payload
    TEST_PAYLOAD='{"features": [5.1, 3.5, 1.4, 0.2]}'

    # Invoke endpoint
    log_info "Sending test request..."
    aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$ENDPOINT_NAME" \
        --content-type application/json \
        --body "$TEST_PAYLOAD" \
        --region "$AWS_REGION" \
        output.json

    # Show response
    log_info "Response:"
    cat output.json
    echo ""

    # Cleanup
    rm output.json
}

# Main execution
main() {
    log_info "Starting SageMaker deployment for ${MODEL_NAME}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Region: ${AWS_REGION}"
    log_info "Instance Type: ${INSTANCE_TYPE}"
    echo ""

    check_prerequisites
    package_model
    upload_to_s3
    deploy_terraform

    echo ""
    log_info "Deployment complete!"
    log_info "You can test the endpoint with:"
    log_info "  ./test.sh ${MODEL_NAME} ${AWS_REGION}"
    echo ""
    log_warn "Remember to destroy resources when done:"
    log_warn "  cd terraform && terraform destroy"
}

# Run main function
main

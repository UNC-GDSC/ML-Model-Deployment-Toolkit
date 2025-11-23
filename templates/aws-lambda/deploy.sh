#!/bin/bash

# AWS Lambda deployment script
set -e

echo "🚀 Deploying ML Model to AWS Lambda..."

# Configuration
PROJECT_NAME=${PROJECT_NAME:-"ml-model"}
ENVIRONMENT=${ENVIRONMENT:-"prod"}
AWS_REGION=${AWS_REGION:-"us-east-1"}
MODEL_PATH=${MODEL_PATH:-"../../models/model.pkl"}

# Create deployment directory
DEPLOY_DIR="deployment"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

echo "📦 Packaging application..."

# Copy handler
cp handler.py $DEPLOY_DIR/

# Copy source code
cp -r ../../src $DEPLOY_DIR/

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt -t $DEPLOY_DIR/package --upgrade --no-cache-dir

# Copy model if exists
if [ -f "$MODEL_PATH" ]; then
    echo "📊 Copying model..."
    mkdir -p $DEPLOY_DIR/model
    cp $MODEL_PATH $DEPLOY_DIR/model/model.pkl
fi

# Create deployment package
echo "📦 Creating deployment package..."
cd $DEPLOY_DIR
zip -r ../deployment.zip . -q
cd ..

echo "✅ Deployment package created: deployment.zip"

# Deploy with Terraform
if command -v terraform &> /dev/null; then
    echo "🏗️  Deploying infrastructure with Terraform..."
    cd terraform

    terraform init
    terraform plan \
        -var="project_name=$PROJECT_NAME" \
        -var="environment=$ENVIRONMENT" \
        -var="aws_region=$AWS_REGION"

    echo "Apply Terraform changes? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        terraform apply \
            -var="project_name=$PROJECT_NAME" \
            -var="environment=$ENVIRONMENT" \
            -var="aws_region=$AWS_REGION" \
            -auto-approve

        echo "✅ Deployment complete!"
        echo "📡 Endpoint URL:"
        terraform output endpoint_url
    fi

    cd ..
else
    echo "⚠️  Terraform not found. Skipping infrastructure deployment."
    echo "You can deploy manually using the AWS CLI or console."
fi

# Cleanup
echo "🧹 Cleaning up..."
rm -rf $DEPLOY_DIR

echo "✨ Done!"

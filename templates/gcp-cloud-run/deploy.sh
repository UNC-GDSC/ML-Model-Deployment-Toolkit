#!/bin/bash

# GCP Cloud Run deployment script
set -e

echo "🚀 Deploying ML Model to GCP Cloud Run..."

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-""}
PROJECT_NAME=${PROJECT_NAME:-"ml-model"}
ENVIRONMENT=${ENVIRONMENT:-"prod"}
REGION=${GCP_REGION:-"us-central1"}
MODEL_PATH=${MODEL_PATH:-"../../models/model.pkl"}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP_PROJECT_ID environment variable is required"
    exit 1
fi

# Set gcloud project
gcloud config set project $PROJECT_ID

# Build container image
echo "🐳 Building container image..."
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${PROJECT_NAME}-models/${PROJECT_NAME}:${ENVIRONMENT}"

# Copy source code
echo "📦 Preparing deployment..."
rm -rf deployment
mkdir -p deployment
cp Dockerfile deployment/
cp main.py deployment/
cp requirements.txt deployment/
cp -r ../../src deployment/

# Copy model if exists
if [ -f "$MODEL_PATH" ]; then
    echo "📊 Copying model..."
    mkdir -p deployment/models
    cp $MODEL_PATH deployment/models/model.pkl
fi

cd deployment

# Build and push image
echo "🏗️  Building and pushing Docker image..."
gcloud builds submit \
    --tag $IMAGE_NAME \
    --timeout=20m \
    .

cd ..
rm -rf deployment

# Deploy with Terraform or gcloud
if command -v terraform &> /dev/null; then
    echo "🏗️  Deploying with Terraform..."
    cd terraform

    terraform init

    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="project_name=$PROJECT_NAME" \
        -var="environment=$ENVIRONMENT" \
        -var="region=$REGION" \
        -var="container_image=$IMAGE_NAME"

    echo "Apply Terraform changes? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        terraform apply \
            -var="project_id=$PROJECT_ID" \
            -var="project_name=$PROJECT_NAME" \
            -var="environment=$ENVIRONMENT" \
            -var="region=$REGION" \
            -var="container_image=$IMAGE_NAME" \
            -auto-approve

        echo "✅ Deployment complete!"
        echo "📡 Service URL:"
        terraform output service_url
    fi

    cd ..
else
    echo "⚠️  Terraform not found. Deploying with gcloud..."

    SERVICE_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 1 \
        --timeout 300 \
        --min-instances 0 \
        --max-instances 100 \
        --set-env-vars "MODEL_VERSION=1.0.0,MODEL_TYPE=sklearn,LOG_LEVEL=INFO,ENVIRONMENT=$ENVIRONMENT"

    echo "✅ Deployment complete!"
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
    echo "📡 Service URL: $SERVICE_URL"
fi

echo "✨ Done!"

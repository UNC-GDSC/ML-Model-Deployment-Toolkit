#!/bin/bash

# Kubernetes deployment script for ML models
set -e

echo "🚀 Deploying ML Model to Kubernetes..."

# Configuration
NAMESPACE=${NAMESPACE:-"ml-models"}
CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-"gcr.io/my-project"}
VERSION=${VERSION:-"latest"}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-"standard"}  # standard, canary, blue-green

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
echo "⚙️  Applying configurations..."
envsubst < configmap.yaml | kubectl apply -n $NAMESPACE -f -
envsubst < secret.yaml | kubectl apply -n $NAMESPACE -f -
envsubst < pvc.yaml | kubectl apply -n $NAMESPACE -f -

# Build and push Docker image
echo "🐳 Building Docker image..."
cd ../../templates/gcp-cloud-run
docker build -t ${CONTAINER_REGISTRY}/ml-model:${VERSION} .

echo "📤 Pushing Docker image..."
docker push ${CONTAINER_REGISTRY}/ml-model:${VERSION}

cd ../../templates/kubernetes

# Deploy based on type
if [ "$DEPLOYMENT_TYPE" = "canary" ]; then
    echo "🐤 Deploying canary release..."
    envsubst < canary-deployment.yaml | kubectl apply -n $NAMESPACE -f -
elif [ "$DEPLOYMENT_TYPE" = "blue-green" ]; then
    echo "🔵 Deploying blue-green release..."
    # Blue-green deployment logic here
    envsubst < deployment.yaml | kubectl apply -n $NAMESPACE -f -
else
    echo "📦 Deploying standard release..."
    envsubst < deployment.yaml | kubectl apply -n $NAMESPACE -f -
fi

# Apply service and ingress
echo "🌐 Setting up networking..."
kubectl apply -n $NAMESPACE -f ingress.yaml

# Setup monitoring
echo "📊 Setting up monitoring..."
kubectl apply -n $NAMESPACE -f servicemonitor.yaml

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/ml-model-deployment -n $NAMESPACE --timeout=5m

# Get service details
echo "✅ Deployment complete!"
echo ""
echo "Service details:"
kubectl get svc ml-model-service -n $NAMESPACE
echo ""
echo "Pod status:"
kubectl get pods -n $NAMESPACE -l app=ml-model
echo ""
echo "Ingress:"
kubectl get ingress ml-model-ingress -n $NAMESPACE

# Test endpoint
echo ""
echo "🧪 Testing endpoint..."
SERVICE_URL=$(kubectl get svc ml-model-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$SERVICE_URL" ]; then
    echo "Testing health endpoint at http://${SERVICE_URL}/health"
    curl -f http://${SERVICE_URL}/health || echo "Health check failed (this is normal if load balancer is still provisioning)"
fi

echo ""
echo "✨ Done!"
echo ""
echo "To access the service:"
echo "  kubectl port-forward -n $NAMESPACE svc/ml-model-service 8080:80"
echo "  curl http://localhost:8080/health"

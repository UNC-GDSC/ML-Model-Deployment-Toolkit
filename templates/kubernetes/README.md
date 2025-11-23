## Kubernetes Deployment Template

Deploy ML models to Kubernetes with auto-scaling, load balancing, and advanced deployment strategies.

## Features

- ✅ Auto-scaling with HPA (Horizontal Pod Autoscaler)
- ✅ Load balancing and service discovery
- ✅ Health checks (liveness & readiness probes)
- ✅ ConfigMaps and Secrets management
- ✅ Ingress with TLS/SSL
- ✅ Prometheus metrics monitoring
- ✅ Canary deployments support
- ✅ Blue-green deployment capability
- ✅ Persistent volume for models
- ✅ Resource limits and requests

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local like minikube)
- kubectl configured
- Docker registry access
- (Optional) Istio for advanced traffic management
- (Optional) Prometheus Operator for monitoring

## Quick Start

### 1. Configure Environment

```bash
export NAMESPACE="ml-models"
export CONTAINER_REGISTRY="gcr.io/my-project"
export VERSION="1.0.0"
```

### 2. Update Configurations

Edit `configmap.yaml` and `secret.yaml` with your settings.

### 3. Deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

## Manual Deployment

### 1. Create Namespace

```bash
kubectl create namespace ml-models
```

### 2. Apply Configurations

```bash
kubectl apply -f configmap.yaml -n ml-models
kubectl apply -f secret.yaml -n ml-models
kubectl apply -f pvc.yaml -n ml-models
```

### 3. Deploy Application

```bash
# Substitute environment variables
export CONTAINER_REGISTRY="your-registry"
export VERSION="1.0.0"

envsubst < deployment.yaml | kubectl apply -n ml-models -f -
```

### 4. Setup Networking

```bash
kubectl apply -f ingress.yaml -n ml-models
```

### 5. Setup Monitoring

```bash
kubectl apply -f servicemonitor.yaml -n ml-models
```

## Deployment Strategies

### Standard Deployment

Default rolling update:

```bash
kubectl apply -f deployment.yaml -n ml-models
```

### Canary Deployment

Gradual rollout with traffic splitting:

```bash
export CANARY_VERSION="2.0.0"
kubectl apply -f canary-deployment.yaml -n ml-models
```

This deploys a canary version receiving 10% of traffic. Monitor metrics and gradually increase traffic:

```bash
# Increase canary traffic to 50%
kubectl patch virtualservice ml-model-vs -n ml-models --type='json' \
  -p='[{"op": "replace", "path": "/spec/http/0/route/0/weight", "value":50}]'
```

### Blue-Green Deployment

Zero-downtime deployment:

```bash
# Deploy green version
kubectl apply -f deployment-green.yaml -n ml-models

# Test green version
kubectl port-forward svc/ml-model-service-green 8080:80 -n ml-models

# Switch traffic
kubectl patch service ml-model-service -n ml-models \
  -p '{"spec":{"selector":{"version":"green"}}}'
```

## Scaling

### Manual Scaling

```bash
kubectl scale deployment ml-model-deployment --replicas=5 -n ml-models
```

### Auto-scaling

HPA is configured to scale based on CPU and memory:

```bash
# View HPA status
kubectl get hpa ml-model-hpa -n ml-models

# Adjust scaling parameters
kubectl edit hpa ml-model-hpa -n ml-models
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -l app=ml-model -n ml-models --tail=100 -f

# Specific pod
kubectl logs <pod-name> -n ml-models --tail=100 -f
```

### Metrics

```bash
# Pod metrics
kubectl top pods -n ml-models -l app=ml-model

# Node metrics
kubectl top nodes
```

### Prometheus

If Prometheus Operator is installed:

```bash
# Check ServiceMonitor
kubectl get servicemonitor ml-model-metrics -n ml-models

# Access Prometheus (if using port-forward)
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n ml-models -l app=ml-model

# Describe pod
kubectl describe pod <pod-name> -n ml-models

# Check logs
kubectl logs <pod-name> -n ml-models
```

### Service Not Accessible

```bash
# Check service
kubectl get svc ml-model-service -n ml-models

# Check endpoints
kubectl get endpoints ml-model-service -n ml-models

# Port forward for testing
kubectl port-forward svc/ml-model-service 8080:80 -n ml-models
curl http://localhost:8080/health
```

### Ingress Issues

```bash
# Check ingress
kubectl get ingress ml-model-ingress -n ml-models

# Describe ingress
kubectl describe ingress ml-model-ingress -n ml-models

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

## Configuration

### Resource Limits

Edit `deployment.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Auto-scaling Thresholds

Edit HPA section in `deployment.yaml`:

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70  # Adjust this
```

### Health Check Parameters

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30  # Adjust based on startup time
  periodSeconds: 10
  failureThreshold: 3
```

## Security

### Secrets Management

```bash
# Create secret from file
kubectl create secret generic ml-model-secrets \
  --from-file=api-key=./api-key.txt \
  -n ml-models

# Update secret
kubectl create secret generic ml-model-secrets \
  --from-literal=API_KEY=new-key \
  --dry-run=client -o yaml | kubectl apply -n ml-models -f -
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-model-netpol
spec:
  podSelector:
    matchLabels:
      app: ml-model
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 8080
```

## Advanced Features

### Istio Service Mesh

For advanced traffic management, install Istio:

```bash
# Enable Istio injection
kubectl label namespace ml-models istio-injection=enabled

# Apply canary with traffic splitting
kubectl apply -f canary-deployment.yaml -n ml-models
```

### GPU Support

For GPU-accelerated inference:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

### Model Persistence

Mount models from various sources:

```yaml
# From S3
volumes:
- name: model-storage
  csi:
    driver: secrets-store.csi.k8s.io
    readOnly: true
    volumeAttributes:
      secretProviderClass: "aws-secrets"
```

## Cost Optimization

1. **Use Cluster Autoscaler**: Scale nodes based on pod requirements
2. **Set Resource Limits**: Prevent over-provisioning
3. **Use Spot Instances**: For non-critical workloads
4. **Enable Pod Disruption Budgets**: Maintain availability during updates

## CI/CD Integration

### GitHub Actions

```yaml
- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/ml-model-deployment \
      ml-model=${{ env.REGISTRY }}/ml-model:${{ github.sha }} \
      -n ml-models
    kubectl rollout status deployment/ml-model-deployment -n ml-models
```

### ArgoCD

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-model
spec:
  source:
    repoURL: https://github.com/your-org/ml-model
    path: templates/kubernetes
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-models
```

## Clean Up

```bash
# Delete all resources
kubectl delete namespace ml-models

# Or selectively delete
kubectl delete -f deployment.yaml -n ml-models
kubectl delete -f ingress.yaml -n ml-models
```

## Support

- Kubernetes Documentation: https://kubernetes.io/docs/
- GitHub Issues: https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues

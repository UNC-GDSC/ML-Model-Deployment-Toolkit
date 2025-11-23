# Azure Functions Deployment Template

Deploy ML models to Azure Functions for serverless inference.

## Features

- ⚡ Serverless execution with auto-scaling
- 💰 Cost-effective pay-per-execution model
- 🔄 Queue-based batch processing
- 📊 Application Insights integration
- 🔐 Built-in authentication
- 🌍 Global distribution with Azure CDN
- 💾 Blob storage integration for models

## Prerequisites

- Azure subscription
- Azure CLI installed
- Azure Functions Core Tools
- Python 3.8+

## Quick Start

### 1. Install Azure Functions Core Tools

```bash
# macOS
brew tap azure/functions
brew install azure-functions-core-tools@4

# Linux
wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install azure-functions-core-tools-4

# Windows (via npm)
npm install -g azure-functions-core-tools@4
```

### 2. Configure Azure

```bash
# Login to Azure
az login

# Create resource group
az group create --name ml-model-rg --location eastus

# Create storage account
az storage account create \
  --name mlmodelstorage \
  --resource-group ml-model-rg \
  --location eastus \
  --sku Standard_LRS

# Create function app
az functionapp create \
  --resource-group ml-model-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name ml-model-function \
  --storage-account mlmodelstorage \
  --os-type Linux
```

### 3. Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Deploy function
func azure functionapp publish ml-model-function
```

### 4. Test

```bash
# Get function URL
FUNCTION_URL=$(az functionapp function show \
  --resource-group ml-model-rg \
  --name ml-model-function \
  --function-name predict \
  --query "invokeUrlTemplate" -o tsv)

# Get function key
FUNCTION_KEY=$(az functionapp keys list \
  --resource-group ml-model-rg \
  --name ml-model-function \
  --query "functionKeys.default" -o tsv)

# Test prediction
curl -X POST "$FUNCTION_URL?code=$FUNCTION_KEY" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Configuration

### Environment Variables

Set in Azure Portal or via CLI:

```bash
az functionapp config appsettings set \
  --name ml-model-function \
  --resource-group ml-model-rg \
  --settings \
    MODEL_VERSION=1.0.0 \
    MODEL_TYPE=sklearn \
    LOG_LEVEL=INFO
```

### App Settings

```json
{
  "MODEL_PATH": "models/model.pkl",
  "MODEL_VERSION": "1.0.0",
  "MODEL_TYPE": "sklearn",
  "LOG_LEVEL": "INFO",
  "ENVIRONMENT": "production"
}
```

## Endpoints

### Health Check

```bash
GET /api/health
```

### Prediction

```bash
POST /api/predict
Content-Type: application/json
x-functions-key: <function-key>

{
  "features": [1.0, 2.0, 3.0, 4.0],
  "return_probabilities": false
}
```

### Model Info

```bash
GET /api/info
```

## Batch Processing

Azure Functions can process batch predictions using queues:

```python
# Add message to queue
from azure.storage.queue import QueueClient

queue_client = QueueClient.from_connection_string(
    conn_str,
    queue_name="prediction-queue"
)

message = {
    "features_list": [[1, 2, 3], [4, 5, 6]],
    "callback_url": "https://callback.example.com"
}

queue_client.send_message(json.dumps(message))
```

## Monitoring

### Application Insights

View metrics in Azure Portal:
- Request rate
- Response time
- Failure rate
- Dependencies

### Logs

```bash
# Stream logs
func azure functionapp logstream ml-model-function

# Query logs with Azure CLI
az monitor app-insights query \
  --app <app-insights-name> \
  --analytics-query "requests | summarize count() by bin(timestamp, 1h)"
```

## Scaling

Azure Functions automatically scales based on load. Configure limits:

```bash
az functionapp config set \
  --name ml-model-function \
  --resource-group ml-model-rg \
  --linux-fx-version "PYTHON|3.11" \
  --always-on true  # For Premium plan
```

## Cost Optimization

1. **Use Consumption Plan**: Pay per execution
2. **Optimize Memory**: Right-size function memory
3. **Use Blob Storage**: Store large models externally
4. **Implement Caching**: Reduce redundant computations

Estimated costs:
- First 1M executions: Free
- Additional executions: $0.20 per million
- Execution time: $0.000016 per GB-second

## Advanced Features

### Durable Functions

For long-running predictions:

```python
import azure.durable_functions as df

@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext):
    features = context.get_input()

    # Fan-out pattern for parallel processing
    tasks = []
    for feature in features:
        tasks.append(context.call_activity("predict_single", feature))

    results = yield context.task_all(tasks)
    return results
```

### Custom Handlers

Integrate with non-Python ML frameworks

### API Management

Add API Management for:
- Rate limiting
- Authentication
- API versioning
- Developer portal

## Security

### Managed Identity

```bash
# Enable managed identity
az functionapp identity assign \
  --name ml-model-function \
  --resource-group ml-model-rg

# Grant access to Key Vault
az keyvault set-policy \
  --name <keyvault-name> \
  --object-id <identity-principal-id> \
  --secret-permissions get list
```

### Private Endpoints

Deploy in VNet for enhanced security:

```bash
az functionapp vnet-integration add \
  --name ml-model-function \
  --resource-group ml-model-rg \
  --vnet <vnet-name> \
  --subnet <subnet-name>
```

## CI/CD

### GitHub Actions

```yaml
name: Deploy to Azure Functions

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Deploy to Azure Functions
        uses: Azure/functions-action@v1
        with:
          app-name: ml-model-function
          package: .
          publish-profile: ${{ secrets.AZURE_FUNCTIONAPP_PUBLISH_PROFILE }}
```

## Troubleshooting

### Function Not Starting

Check logs:
```bash
func azure functionapp logstream ml-model-function
```

### Cold Start Issues

- Use Premium plan for always-on
- Implement model caching
- Optimize dependencies

### Memory Issues

Increase memory allocation:
```bash
az functionapp config set \
  --name ml-model-function \
  --resource-group ml-model-rg \
  --linux-fx-version "PYTHON|3.11" \
  --number-of-workers 1
```

## Clean Up

```bash
az group delete --name ml-model-rg --yes
```

## Support

- Azure Functions Docs: https://docs.microsoft.com/azure/azure-functions/
- GitHub Issues: https://github.com/UNC-GDSC/ML-Model-Deployment-Toolkit/issues

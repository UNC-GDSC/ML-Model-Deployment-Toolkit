# Scikit-learn Example

This example demonstrates training and deploying a scikit-learn Random Forest classifier.

## Quick Start

### 1. Train the Model

```bash
cd examples/sklearn
python train_model.py
```

This will:
- Generate a synthetic classification dataset
- Train a Random Forest model
- Evaluate the model
- Save it to `models/sklearn_model.pkl`

### 2. Test Predictions

```bash
python test_prediction.py
```

### 3. Deploy the Model

#### AWS Lambda

```bash
cd ../../templates/aws-lambda
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
./deploy.sh
```

#### GCP Cloud Run

```bash
cd ../../templates/gcp-cloud-run
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
export GCP_PROJECT_ID="your-project-id"
./deploy.sh
```

#### Vercel

```bash
cd ../../templates/vercel
export MODEL_PATH=../../examples/sklearn/models/sklearn_model.pkl
./deploy.sh
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 20 features (15 informative, 5 redundant)
- **Classes**: 2 (binary classification)
- **Training samples**: 800
- **Test samples**: 200

## API Usage

Once deployed, make predictions:

```bash
curl -X POST https://your-endpoint/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -1.2, 0.8, -0.3, 1.1, 0.2, -0.9, 0.4, -0.6, 0.7,
                 0.1, -0.4, 0.9, -1.0, 0.3, -0.2, 0.6, -0.8, 0.5, -0.1],
    "return_probabilities": true
  }'
```

Response:
```json
{
  "prediction": 1,
  "probabilities": [0.23, 0.77],
  "model_version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "latency_ms": 45.2
}
```

## Customization

Modify `train_model.py` to:
- Use your own dataset
- Change model hyperparameters
- Add feature engineering
- Use different algorithms

Example with your own data:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/your_model.pkl')
```

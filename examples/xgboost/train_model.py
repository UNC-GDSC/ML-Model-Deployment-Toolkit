"""Train an XGBoost model for demonstration."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os

# Create output directory
os.makedirs('models', exist_ok=True)

# Generate synthetic dataset
print("Generating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train model using sklearn API
print("\nTraining XGBoost model (sklearn API)...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save sklearn-style model
import joblib
sklearn_model_path = 'models/xgboost_sklearn_model.pkl'
joblib.dump(model, sklearn_model_path)
print(f"\nSaved sklearn-style model to: {sklearn_model_path}")

# Also train and save using native API
print("\nTraining XGBoost model (native API)...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

bst = xgb.train(params, dtrain, num_boost_round=100)

# Save native model
native_model_path = 'models/xgboost_native_model.json'
bst.save_model(native_model_path)
print(f"Saved native model to: {native_model_path}")

# Test both models
print("\n✅ Model training complete!")
print(f"Sklearn model size: {os.path.getsize(sklearn_model_path) / 1024:.2f} KB")
print(f"Native model size: {os.path.getsize(native_model_path) / 1024:.2f} KB")

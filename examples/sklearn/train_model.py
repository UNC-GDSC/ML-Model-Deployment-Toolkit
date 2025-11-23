"""Train a simple scikit-learn model for demonstration."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
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

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = 'models/sklearn_model.pkl'
print(f"\nSaving model to {model_path}...")
joblib.dump(model, model_path)

# Test loading
print("Testing model loading...")
loaded_model = joblib.load(model_path)
test_pred = loaded_model.predict(X_test[:5])
print(f"Test predictions: {test_pred}")

print("\n✅ Model training complete!")
print(f"Model saved to: {model_path}")
print(f"Model size: {os.path.getsize(model_path) / 1024:.2f} KB")

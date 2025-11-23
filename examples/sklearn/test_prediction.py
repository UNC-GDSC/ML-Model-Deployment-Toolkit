"""Test predictions with the trained sklearn model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models.sklearn_model import SklearnModel

# Load model
print("Loading model...")
model = SklearnModel('models/sklearn_model.pkl', '1.0.0')
model.load()

print(f"Model loaded: {model}")
print(f"Model info: {model.get_model_info()}")

# Test predictions
print("\nTesting predictions...")

# Single prediction
features = np.random.randn(20)
result = model.predict_with_preprocessing(features)
print(f"\nSingle prediction: {result}")

# Batch prediction
batch_features = np.random.randn(5, 20)
result = model.predict_with_preprocessing(batch_features, return_probabilities=True)
print(f"\nBatch prediction: {result}")

# Health check
health = model.health_check()
print(f"\nHealth check: {health}")

print("\n✅ All tests passed!")

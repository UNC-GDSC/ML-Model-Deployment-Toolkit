"""Example of model explainability using SHAP."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models.sklearn_model import SklearnModel

print("🔬 Model Explainability Example\n")

# Load model
print("Loading model...")
model = SklearnModel('../sklearn/models/sklearn_model.pkl', '1.0.0')
model.load()

print("Model loaded successfully!")

# Try using SHAP
try:
    from src.explainability.shap_explainer import ModelExplainer

    print("\n--- SHAP Explainability ---")

    # Create explainer
    # For tree-based models, SHAP can work without background data
    explainer = ModelExplainer(model)

    # Generate sample feature
    features = np.random.randn(20)
    feature_names = [f'feature_{i}' for i in range(20)]

    print("Explaining prediction...")
    explanation = explainer.explain_prediction(features, feature_names)

    print(f"\nPrediction: {explanation['prediction']}")
    if explanation['base_value'] is not None:
        print(f"Base value: {explanation['base_value']:.4f}")

    print("\nTop 10 Feature Contributions:")
    for i, (name, value) in enumerate(list(explanation['feature_importance'].items())[:10]):
        direction = "increases" if value > 0 else "decreases"
        print(f"{i+1}. {name}: {value:+.4f} ({direction} prediction)")

    # Get top features
    print("\n--- Top 5 Features ---")
    top_features = explainer.get_top_features(features, top_k=5, feature_names=feature_names)

    for feat in top_features:
        print(f"Rank {feat['rank']}: {feat['feature']} (SHAP value: {feat['shap_value']:+.4f})")

    # Global importance
    print("\n--- Global Feature Importance ---")
    print("Generating sample data for global importance...")

    # Generate sample dataset
    sample_data = np.random.randn(100, 20)

    global_importance = explainer.get_global_importance(sample_data, feature_names)

    print("\nTop 10 Most Important Features (Global):")
    for i, (name, importance) in enumerate(list(global_importance.items())[:10]):
        print(f"{i+1}. {name}: {importance:.4f}")

    print("\n✅ SHAP explainability complete!")

except ImportError:
    print("\n⚠️  SHAP not installed. Install with: pip install shap")
    print("Skipping SHAP examples...")

# Try using LIME
try:
    from src.explainability.shap_explainer import LIMEExplainer

    print("\n--- LIME Explainability ---")

    # Create explainer
    lime_explainer = LIMEExplainer(model, feature_names)

    # Initialize with training data
    training_data = np.random.randn(100, 20)
    lime_explainer.initialize_explainer(training_data, mode='classification')

    # Explain prediction
    features = np.random.randn(20)
    explanation = lime_explainer.explain_prediction(features, num_features=10)

    print(f"\nPrediction: {explanation['prediction']}")
    print(f"Explanation score: {explanation['score']:.4f}")

    print("\nTop Feature Contributions:")
    for name, value in list(explanation['feature_importance'].items())[:10]:
        print(f"  {name}: {value:+.4f}")

    print("\n✅ LIME explainability complete!")

except ImportError:
    print("\n⚠️  LIME not installed. Install with: pip install lime")
    print("Skipping LIME examples...")

print("\n✅ Explainability example complete!")

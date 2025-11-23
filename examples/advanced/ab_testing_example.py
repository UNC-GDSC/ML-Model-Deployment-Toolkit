"""Example of A/B testing with multiple model versions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models.sklearn_model import SklearnModel
from src.versioning.ab_testing import ABTestManager

# Create two model versions (using same model for demo)
print("Loading models...")

model_v1 = SklearnModel('../sklearn/models/sklearn_model.pkl', '1.0.0')
model_v1.load()

model_v2 = SklearnModel('../sklearn/models/sklearn_model.pkl', '2.0.0')
model_v2.load()

models = {
    'v1': model_v1,
    'v2': model_v2
}

# Create A/B test manager
print("\nCreating A/B test...")
ab_manager = ABTestManager(models)

# Create experiment with 80/20 split
experiment = ab_manager.create_experiment(
    name='model_comparison',
    variants={'v1': 80.0, 'v2': 20.0},
    description='Testing new model version with 20% traffic'
)

print(f"Created experiment: {experiment.name}")
print(f"Traffic allocation: {experiment.variants}")

# Simulate predictions
print("\nSimulating predictions...")
num_requests = 100

v1_count = 0
v2_count = 0

for i in range(num_requests):
    # Generate random features
    features = np.random.randn(20)

    # Make prediction with A/B test
    result = ab_manager.predict_with_ab_test(
        'model_comparison',
        features,
        return_probabilities=True
    )

    # Track variant selection
    if result['variant'] == 'v1':
        v1_count += 1
    else:
        v2_count += 1

    # Simulate recording metrics (e.g., accuracy, latency)
    if i % 10 == 0:
        # Record latency metric
        ab_manager.record_metric(
            'model_comparison',
            result['variant'],
            'latency_ms',
            result['latency_ms']
        )

print(f"\nVariant distribution:")
print(f"v1: {v1_count} requests ({v1_count/num_requests*100:.1f}%)")
print(f"v2: {v2_count} requests ({v2_count/num_requests*100:.1f}%)")

# Get experiment stats
print("\nExperiment statistics:")
stats = ab_manager.get_experiment_stats('model_comparison')
print(f"Total requests: {num_requests}")

for variant, variant_stats in stats['variants'].items():
    print(f"\n{variant}:")
    print(f"  Requests: {variant_stats['requests']}")
    print(f"  Errors: {variant_stats['errors']}")
    print(f"  Error rate: {variant_stats['error_rate']:.4f}")

    if 'latency_ms' in variant_stats['metrics']:
        latency = variant_stats['metrics']['latency_ms']
        print(f"  Avg latency: {latency['mean']:.2f}ms")

print("\n✅ A/B testing example complete!")

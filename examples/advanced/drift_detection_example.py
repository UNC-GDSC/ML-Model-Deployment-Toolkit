"""Example of data drift detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.monitoring.drift_detection import DataDriftDetector

print("🔍 Data Drift Detection Example\n")

# Generate reference data (training data)
print("Generating reference dataset...")
np.random.seed(42)
reference_data = np.random.randn(1000, 10)

# Create drift detector
feature_names = [f'feature_{i}' for i in range(10)]
detector = DataDriftDetector(
    reference_data,
    feature_names=feature_names,
    threshold=0.05
)

print(f"Initialized drift detector with {len(feature_names)} features")

# Scenario 1: No drift (similar distribution)
print("\n--- Scenario 1: No Drift ---")
current_data = np.random.randn(200, 10)

drift_result = detector.detect_drift(current_data, method='ks')

print(f"Drift detected: {drift_result['drift_detected']}")
print(f"Features drifted: {drift_result['num_features_drifted']}/{drift_result['total_features']}")
print(f"Drift percentage: {drift_result['drift_percentage']:.1f}%")

# Show drifted features
if drift_result['drift_detected']:
    drifted = [name for name, drifted in drift_result['drift_by_feature'].items() if drifted]
    print(f"Drifted features: {drifted}")

# Scenario 2: With drift (shifted distribution)
print("\n--- Scenario 2: With Drift ---")
# Shift some features
drifted_data = np.random.randn(200, 10)
drifted_data[:, 0] += 2.0  # Shift feature 0
drifted_data[:, 1] *= 3.0  # Scale feature 1

drift_result = detector.detect_drift(drifted_data, method='ks')

print(f"Drift detected: {drift_result['drift_detected']}")
print(f"Features drifted: {drift_result['num_features_drifted']}/{drift_result['total_features']}")
print(f"Drift percentage: {drift_result['drift_percentage']:.1f}%")

if drift_result['drift_detected']:
    drifted = [name for name, drifted in drift_result['drift_by_feature'].items() if drifted]
    print(f"Drifted features: {drifted}")

    # Show p-values
    print("\nP-values:")
    for name, p_value in drift_result['p_values'].items():
        status = "DRIFT" if p_value < 0.05 else "OK"
        print(f"  {name}: {p_value:.4f} [{status}]")

# Calculate PSI (Population Stability Index)
print("\n--- Population Stability Index ---")
psi_values = detector.calculate_psi(drifted_data)

print("PSI values:")
for name, psi in sorted(psi_values.items(), key=lambda x: x[1], reverse=True):
    if psi < 0.1:
        status = "No change"
    elif psi < 0.25:
        status = "Moderate change"
    else:
        status = "Significant change"

    print(f"  {name}: {psi:.4f} [{status}]")

# Generate drift report
print("\n--- Drift Report ---")
report = detector.get_drift_report()

print(f"Total checks: {report['total_checks']}")
print(f"Drift count: {report['drift_count']}")
print(f"Drift rate: {report['drift_rate']:.1f}%")

print("\nMost frequently drifted features:")
for name, frequency in report['most_drifted_features']:
    print(f"  {name}: {frequency:.1f}%")

print("\n✅ Drift detection example complete!")

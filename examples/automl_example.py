"""
AutoML Pipeline Example

This example demonstrates how to use the AutoML pipeline for automatic
model selection, feature engineering, and hyperparameter tuning.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.automl.auto_pipeline import AutoMLPipeline
from src.automl.hyperparameter_tuner import HyperparameterTuner, TuningMethod


def classification_example():
    """Example: Automatic classification pipeline."""
    print("=" * 70)
    print("AutoML Classification Example")
    print("=" * 70)
    print()

    # Generate sample data (in production, use real data)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print()

    # Create AutoML pipeline
    print("Creating AutoML pipeline...")
    automl = AutoMLPipeline(
        task="auto",  # Automatically detect classification/regression
        metric="auto",  # Automatically select appropriate metric
        time_budget=60,  # 60 seconds time budget
        n_jobs=-1  # Use all CPU cores
    )

    # Fit pipeline (this will automatically:
    # 1. Detect task type
    # 2. Engineer features
    # 3. Select best model
    # 4. Tune hyperparameters
    # 5. Train final model)
    print("Fitting AutoML pipeline...")
    print("-" * 70)
    automl.fit(X_train, y_train)
    print("-" * 70)
    print()

    # Make predictions
    y_pred = automl.predict(X_test)

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()

    # Get model information
    model_info = automl.get_model_info()
    print("Best Model Information:")
    print(f"  Model: {model_info['model_class']}")
    print(f"  Best Score (CV): {model_info['best_score']:.4f}")
    print(f"  Best Parameters:")
    for param, value in model_info['best_params'].items():
        print(f"    {param}: {value}")
    print()

    # Save model
    print("Saving model...")
    automl.save_model('models/automl_classifier.pkl')
    print("Model saved to: models/automl_classifier.pkl")
    print()

    # Generate deployment config
    deploy_config = automl.generate_deployment_config()
    print("Deployment Recommendations:")
    for rec in deploy_config['deployment_recommendations']['recommended_platforms']:
        print(f"  - {rec['platform']}: {rec['reason']}")
    print()


def regression_example():
    """Example: Automatic regression pipeline."""
    print("=" * 70)
    print("AutoML Regression Example")
    print("=" * 70)
    print()

    # Generate sample data
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} training samples")
    print()

    # Create and fit AutoML pipeline
    automl = AutoMLPipeline(time_budget=60)
    automl.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = automl.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print()


def hyperparameter_tuning_example():
    """Example: Manual hyperparameter tuning."""
    print("=" * 70)
    print("Hyperparameter Tuning Example")
    print("=" * 70)
    print()

    # Generate data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define search space
    param_space = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
    }

    # Try different tuning methods
    methods = [
        TuningMethod.RANDOM_SEARCH,
        TuningMethod.BAYESIAN,
        TuningMethod.EVOLUTIONARY
    ]

    results = {}

    for method in methods:
        print(f"\nTuning with {method.value}...")
        print("-" * 70)

        tuner = HyperparameterTuner(
            method=method,
            metric='accuracy',
            n_trials=20,
            random_state=42
        )

        result = tuner.tune(
            RandomForestClassifier,
            param_space,
            X_train,
            y_train,
            cv=3
        )

        results[method.value] = result

        print(f"Best Score: {result['best_score']:.4f}")
        print(f"Best Params: {result['best_params']}")
        print(f"Time: {result['tuning_time_seconds']:.2f}s")

    # Compare methods
    print("\n" + "=" * 70)
    print("Comparison of Tuning Methods")
    print("=" * 70)
    print(f"{'Method':<20} {'Score':<10} {'Time (s)':<10}")
    print("-" * 70)

    for method_name, result in results.items():
        print(f"{method_name:<20} {result['best_score']:<10.4f} {result['tuning_time_seconds']:<10.2f}")


def main():
    """Run all examples."""
    import os
    os.makedirs('models', exist_ok=True)

    print("\n")
    classification_example()

    print("\n")
    regression_example()

    print("\n")
    hyperparameter_tuning_example()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Deploy your model:")
    print("   cd templates/aws-sagemaker")
    print("   ./deploy.sh automl-classifier prod")
    print()
    print("2. Or use the CLI:")
    print("   ml-deploy aws-lambda --model-path models/automl_classifier.pkl")
    print()


if __name__ == '__main__':
    main()

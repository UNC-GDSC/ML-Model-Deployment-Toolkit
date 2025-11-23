"""Tests for model wrappers."""

import pytest
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.models.sklearn_model import SklearnModel


@pytest.fixture
def sample_sklearn_model(tmp_path):
    """Create a sample sklearn model for testing."""
    # Create simple model
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)

    return str(model_path)


class TestSklearnModel:
    """Tests for SklearnModel class."""

    def test_model_initialization(self, sample_sklearn_model):
        """Test model initialization."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")
        assert model.model_path == sample_sklearn_model
        assert model.model_version == "1.0.0"
        assert not model.loaded

    def test_model_loading(self, sample_sklearn_model):
        """Test model loading."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")
        model.load()

        assert model.loaded
        assert model.model is not None
        assert model.load_timestamp is not None

    def test_prediction_single(self, sample_sklearn_model):
        """Test single prediction."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")
        model.load()

        features = np.random.rand(4)
        result = model.predict_with_preprocessing(features)

        assert 'prediction' in result
        assert 'model_version' in result
        assert 'timestamp' in result
        assert 'latency_ms' in result
        assert result['model_version'] == "1.0.0"

    def test_prediction_batch(self, sample_sklearn_model):
        """Test batch prediction."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")
        model.load()

        features = np.random.rand(5, 4)
        result = model.predict_with_preprocessing(features)

        assert 'prediction' in result
        assert isinstance(result['prediction'], list)
        assert len(result['prediction']) == 5

    def test_prediction_with_probabilities(self, sample_sklearn_model):
        """Test prediction with probabilities."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")
        model.load()

        features = np.random.rand(4)
        result = model.predict_with_preprocessing(features, return_probabilities=True)

        assert 'probabilities' in result
        assert isinstance(result['probabilities'], list)

    def test_preprocess(self, sample_sklearn_model):
        """Test preprocessing."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")

        # Test list input
        features_list = [1.0, 2.0, 3.0, 4.0]
        processed = model.preprocess(features_list)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (1, 4)

        # Test numpy input
        features_array = np.array([1.0, 2.0, 3.0, 4.0])
        processed = model.preprocess(features_array)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (1, 4)

    def test_health_check(self, sample_sklearn_model):
        """Test health check."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")

        # Before loading
        health = model.health_check()
        assert health['status'] == 'unhealthy'
        assert not health['model_loaded']

        # After loading
        model.load()
        health = model.health_check()
        assert health['status'] == 'healthy'
        assert health['model_loaded']
        assert health['uptime_seconds'] is not None

    def test_prediction_without_loading(self, sample_sklearn_model):
        """Test that prediction fails without loading model."""
        model = SklearnModel(sample_sklearn_model, "1.0.0")

        with pytest.raises(RuntimeError):
            model.predict_with_preprocessing([1.0, 2.0, 3.0, 4.0])

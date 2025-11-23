"""Tests for request handlers."""

import pytest
import json
import numpy as np
from unittest.mock import Mock, MagicMock

from src.handlers.lambda_handler import LambdaHandler
from src.core.base_model import BaseModel


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self):
        super().__init__("mock_model.pkl", "1.0.0")
        self.loaded = True

    def load(self):
        self.loaded = True

    def predict(self, features):
        return np.array([1])

    def preprocess(self, features):
        return np.array(features).reshape(1, -1)

    def postprocess(self, predictions):
        return int(predictions[0])


class TestLambdaHandler:
    """Tests for Lambda handler."""

    def test_health_check(self):
        """Test health check endpoint."""
        model = MockModel()
        handler = LambdaHandler(model)

        event = {
            'path': '/health',
            'httpMethod': 'GET'
        }

        response = handler.handle_request(event)

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['status'] == 'healthy'
        assert body['model_loaded'] is True

    def test_prediction_success(self):
        """Test successful prediction."""
        model = MockModel()
        handler = LambdaHandler(model)

        event = {
            'path': '/predict',
            'httpMethod': 'POST',
            'body': json.dumps({
                'features': [1.0, 2.0, 3.0, 4.0]
            })
        }

        response = handler.handle_request(event)

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'prediction' in body
        assert 'model_version' in body

    def test_prediction_validation_error(self):
        """Test prediction with invalid input."""
        model = MockModel()
        handler = LambdaHandler(model)

        event = {
            'path': '/predict',
            'httpMethod': 'POST',
            'body': json.dumps({})  # Missing features
        }

        response = handler.handle_request(event)

        assert response['statusCode'] == 400

    def test_invalid_path(self):
        """Test request to invalid path."""
        model = MockModel()
        handler = LambdaHandler(model)

        event = {
            'path': '/invalid',
            'httpMethod': 'GET'
        }

        response = handler.handle_request(event)

        assert response['statusCode'] == 404

    def test_model_info(self):
        """Test model info endpoint."""
        model = MockModel()
        handler = LambdaHandler(model)

        event = {
            'path': '/info',
            'httpMethod': 'GET'
        }

        response = handler.handle_request(event)

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'model_version' in body

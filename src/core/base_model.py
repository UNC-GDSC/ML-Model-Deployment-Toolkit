"""Base model class for all ML model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json
import logging
from datetime import datetime

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field, validator


logger = logging.getLogger(__name__)


class PredictionRequest(PydanticBaseModel):
    """Schema for prediction requests."""

    features: Union[List[float], List[List[float]], Dict[str, Any]] = Field(
        ..., description="Input features for prediction"
    )
    model_version: Optional[str] = Field(None, description="Model version to use")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    @validator('features')
    def validate_features(cls, v):
        """Validate features format."""
        if not v:
            raise ValueError("Features cannot be empty")
        return v


class PredictionResponse(PydanticBaseModel):
    """Schema for prediction responses."""

    prediction: Union[float, int, List[float], List[int], str]
    probabilities: Optional[List[float]] = None
    model_version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    latency_ms: Optional[float] = None


class HealthResponse(PydanticBaseModel):
    """Schema for health check responses."""

    status: str = "healthy"
    model_loaded: bool
    model_version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    uptime_seconds: Optional[float] = None


class BaseModel(ABC):
    """
    Abstract base class for all ML model wrappers.

    This class provides a common interface for loading, predicting,
    and managing ML models across different frameworks.
    """

    def __init__(self, model_path: str, model_version: str = "1.0.0"):
        """
        Initialize the base model.

        Args:
            model_path: Path to the serialized model file
            model_version: Version identifier for the model
        """
        self.model_path = model_path
        self.model_version = model_version
        self.model = None
        self.loaded = False
        self.load_timestamp = None

    @abstractmethod
    def load(self) -> None:
        """
        Load the model from disk.

        This method should be implemented by subclasses to handle
        framework-specific model loading.
        """
        pass

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """
        Make predictions using the loaded model.

        Args:
            features: Input features for prediction

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def preprocess(self, features: Any) -> Any:
        """
        Preprocess input features before prediction.

        Args:
            features: Raw input features

        Returns:
            Preprocessed features ready for model input
        """
        pass

    @abstractmethod
    def postprocess(self, predictions: Any) -> Any:
        """
        Postprocess model predictions.

        Args:
            predictions: Raw model predictions

        Returns:
            Processed predictions ready for response
        """
        pass

    def predict_with_preprocessing(
        self,
        features: Any,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline with pre/post processing.

        Args:
            features: Raw input features
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = datetime.utcnow()

        # Preprocess
        processed_features = self.preprocess(features)

        # Predict
        predictions = self.predict(processed_features)

        # Postprocess
        final_predictions = self.postprocess(predictions)

        # Calculate latency
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = {
            "prediction": final_predictions,
            "model_version": self.model_version,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": latency
        }

        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_features)
            result["probabilities"] = probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities

        return result

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the model.

        Returns:
            Dictionary containing health status information
        """
        uptime = None
        if self.load_timestamp:
            uptime = (datetime.utcnow() - self.load_timestamp).total_seconds()

        return {
            "status": "healthy" if self.loaded else "unhealthy",
            "model_loaded": self.loaded,
            "model_version": self.model_version,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_path": self.model_path,
            "model_version": self.model_version,
            "loaded": self.loaded,
            "load_timestamp": self.load_timestamp.isoformat() if self.load_timestamp else None,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(version={self.model_version}, loaded={self.loaded})"

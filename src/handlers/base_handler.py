"""Base handler for all platforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
from datetime import datetime

from src.core.base_model import BaseModel, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """
    Abstract base class for platform-specific request handlers.

    This class provides common functionality for handling prediction
    requests across different deployment platforms.
    """

    def __init__(self, model: BaseModel):
        """
        Initialize the handler.

        Args:
            model: Loaded BaseModel instance
        """
        self.model = model
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0

    @abstractmethod
    def handle_request(self, event: Any, context: Any = None) -> Dict[str, Any]:
        """
        Handle an incoming request.

        Args:
            event: Platform-specific event object
            context: Platform-specific context object

        Returns:
            Platform-specific response object
        """
        pass

    def process_prediction(
        self,
        features: Any,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Process a prediction request.

        Args:
            features: Input features
            return_probabilities: Whether to return class probabilities

        Returns:
            Prediction results
        """
        start_time = time.time()

        try:
            # Make prediction
            result = self.model.predict_with_preprocessing(
                features,
                return_probabilities=return_probabilities
            )

            # Update metrics
            self.request_count += 1
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency

            logger.info(f"Prediction successful. Latency: {latency:.2f}ms")

            return result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check.

        Returns:
            Health status information
        """
        health = self.model.health_check()

        # Add handler metrics
        avg_latency = (
            self.total_latency / self.request_count
            if self.request_count > 0
            else 0
        )

        health.update({
            'handler_metrics': {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': (
                    self.error_count / self.request_count
                    if self.request_count > 0
                    else 0
                ),
                'avg_latency_ms': avg_latency
            }
        })

        return health

    def validate_request(self, data: Dict[str, Any]) -> PredictionRequest:
        """
        Validate incoming request data.

        Args:
            data: Request data dictionary

        Returns:
            Validated PredictionRequest object

        Raises:
            ValueError: If validation fails
        """
        try:
            return PredictionRequest(**data)
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            raise ValueError(f"Invalid request format: {e}")

    def format_error_response(
        self,
        error: Exception,
        status_code: int = 500
    ) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error: Exception that occurred
            status_code: HTTP status code

        Returns:
            Formatted error response
        """
        return {
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.utcnow().isoformat(),
            'status_code': status_code
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get handler metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': (
                self.error_count / self.request_count
                if self.request_count > 0
                else 0
            ),
            'avg_latency_ms': (
                self.total_latency / self.request_count
                if self.request_count > 0
                else 0
            )
        }

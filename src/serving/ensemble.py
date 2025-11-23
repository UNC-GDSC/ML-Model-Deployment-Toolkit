"""Model ensemble serving for improved predictions."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum

from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    STACKING = "stacking"
    MAX = "max"
    MIN = "min"


class ModelEnsemble:
    """
    Multi-model ensemble serving.

    Combines predictions from multiple models for improved accuracy.
    """

    def __init__(
        self,
        models: List[BaseModel],
        method: EnsembleMethod = EnsembleMethod.AVERAGING,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize model ensemble.

        Args:
            models: List of model instances
            method: Ensemble method
            weights: Weights for weighted averaging (must sum to 1.0)
        """
        self.models = models
        self.method = method
        self.weights = weights

        # Validate weights
        if method == EnsembleMethod.WEIGHTED_AVERAGING:
            if weights is None:
                raise ValueError("Weights required for weighted averaging")
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")

        logger.info(f"Initialized ensemble with {len(models)} models using {method.value}")

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Make ensemble prediction.

        Args:
            features: Input features

        Returns:
            Ensemble prediction result
        """
        # Get predictions from all models
        predictions = []
        model_results = []

        for i, model in enumerate(self.models):
            result = model.predict_with_preprocessing(features)
            predictions.append(result['prediction'])
            model_results.append(result)

        # Combine predictions
        ensemble_prediction = self._combine_predictions(predictions)

        # Calculate consensus/variance
        variance = np.var(predictions) if len(predictions) > 1 else 0.0
        consensus = 1.0 - (variance / (np.mean(predictions) + 1e-10))

        return {
            'prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'ensemble_method': self.method.value,
            'model_count': len(self.models),
            'consensus_score': float(consensus),
            'variance': float(variance),
            'model_results': model_results
        }

    def _combine_predictions(self, predictions: List) -> Any:
        """
        Combine predictions using ensemble method.

        Args:
            predictions: List of predictions

        Returns:
            Combined prediction
        """
        if self.method == EnsembleMethod.VOTING:
            # Majority voting for classification
            from collections import Counter
            votes = Counter(predictions)
            return votes.most_common(1)[0][0]

        elif self.method == EnsembleMethod.AVERAGING:
            # Simple averaging
            return np.mean(predictions)

        elif self.method == EnsembleMethod.WEIGHTED_AVERAGING:
            # Weighted averaging
            return np.average(predictions, weights=self.weights)

        elif self.method == EnsembleMethod.MAX:
            return np.max(predictions)

        elif self.method == EnsembleMethod.MIN:
            return np.min(predictions)

        elif self.method == EnsembleMethod.STACKING:
            # Simple stacking (can be enhanced with meta-learner)
            return np.mean(predictions)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_with_probabilities(self, features: Any) -> Dict[str, Any]:
        """
        Make ensemble prediction with probability averaging.

        Args:
            features: Input features

        Returns:
            Ensemble prediction with averaged probabilities
        """
        all_probabilities = []

        for model in self.models:
            result = model.predict_with_preprocessing(features, return_probabilities=True)
            if 'probabilities' in result:
                all_probabilities.append(result['probabilities'])

        if all_probabilities:
            # Average probabilities
            avg_probabilities = np.mean(all_probabilities, axis=0)
            prediction = np.argmax(avg_probabilities)

            return {
                'prediction': int(prediction),
                'probabilities': avg_probabilities.tolist(),
                'individual_probabilities': all_probabilities,
                'ensemble_method': self.method.value
            }

        # Fall back to regular prediction
        return self.predict(features)


class ModelRouter:
    """
    Route requests to different models based on conditions.
    """

    def __init__(self, models: Dict[str, BaseModel]):
        """
        Initialize model router.

        Args:
            models: Dictionary of model_name -> BaseModel
        """
        self.models = models
        self.routes = []

        logger.info(f"Initialized model router with {len(models)} models")

    def add_route(
        self,
        model_name: str,
        condition: callable,
        priority: int = 0
    ):
        """
        Add routing rule.

        Args:
            model_name: Name of model to route to
            condition: Function that returns True if route should be used
            priority: Route priority (higher = checked first)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        self.routes.append({
            'model_name': model_name,
            'condition': condition,
            'priority': priority
        })

        # Sort by priority
        self.routes.sort(key=lambda x: x['priority'], reverse=True)

        logger.info(f"Added route to {model_name} with priority {priority}")

    def predict(
        self,
        features: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route prediction to appropriate model.

        Args:
            features: Input features
            metadata: Additional metadata for routing

        Returns:
            Prediction result with routing info
        """
        metadata = metadata or {}

        # Find matching route
        selected_model = None
        selected_route = None

        for route in self.routes:
            try:
                if route['condition'](features, metadata):
                    selected_model = self.models[route['model_name']]
                    selected_route = route['model_name']
                    break
            except Exception as e:
                logger.warning(f"Route condition failed: {e}")
                continue

        # Default to first model if no match
        if selected_model is None:
            selected_route = list(self.models.keys())[0]
            selected_model = self.models[selected_route]
            logger.warning(f"No matching route, using default: {selected_route}")

        # Make prediction
        result = selected_model.predict_with_preprocessing(features)
        result['selected_model'] = selected_route
        result['routing_metadata'] = metadata

        return result


class ModelChain:
    """
    Chain multiple models in sequence.

    Output of one model becomes input to the next.
    """

    def __init__(self, models: List[BaseModel]):
        """
        Initialize model chain.

        Args:
            models: List of models in chain order
        """
        self.models = models
        logger.info(f"Initialized model chain with {len(models)} models")

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Make prediction through model chain.

        Args:
            features: Initial input features

        Returns:
            Final prediction result
        """
        current_input = features
        intermediate_results = []

        for i, model in enumerate(self.models):
            result = model.predict_with_preprocessing(current_input)
            intermediate_results.append({
                'model_index': i,
                'model_version': model.model_version,
                'output': result['prediction']
            })

            # Use prediction as input for next model
            current_input = result['prediction']

        return {
            'final_prediction': current_input,
            'intermediate_results': intermediate_results,
            'chain_length': len(self.models)
        }


class AdaptiveEnsemble:
    """
    Adaptive ensemble that adjusts weights based on performance.
    """

    def __init__(self, models: List[BaseModel]):
        """
        Initialize adaptive ensemble.

        Args:
            models: List of model instances
        """
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Start with equal weights
        self.performance_history = [[] for _ in models]

        logger.info(f"Initialized adaptive ensemble with {len(models)} models")

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Make adaptive ensemble prediction.

        Args:
            features: Input features

        Returns:
            Prediction result
        """
        predictions = []

        for model in self.models:
            result = model.predict_with_preprocessing(features)
            predictions.append(result['prediction'])

        # Weighted average
        ensemble_prediction = np.average(predictions, weights=self.weights)

        return {
            'prediction': ensemble_prediction,
            'weights': self.weights.tolist(),
            'individual_predictions': predictions
        }

    def update_weights(self, model_index: int, performance_score: float):
        """
        Update model weights based on performance.

        Args:
            model_index: Index of model
            performance_score: Performance score (higher is better)
        """
        self.performance_history[model_index].append(performance_score)

        # Update weights based on recent performance
        if len(self.performance_history[model_index]) >= 10:
            recent_performance = np.mean(self.performance_history[model_index][-10:])

            # Update weight (simple exponential moving average)
            alpha = 0.1
            self.weights[model_index] = (
                alpha * recent_performance +
                (1 - alpha) * self.weights[model_index]
            )

            # Normalize weights
            self.weights = self.weights / np.sum(self.weights)

            logger.info(f"Updated weights for model {model_index}: {self.weights[model_index]:.4f}")

"""Model explainability utilities using SHAP."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Model explainability using SHAP (SHapley Additive exPlanations).

    Provides feature importance and prediction explanations.
    """

    def __init__(self, model: BaseModel, background_data: Optional[np.ndarray] = None):
        """
        Initialize model explainer.

        Args:
            model: Model instance to explain
            background_data: Background dataset for SHAP (optional)
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. "
                "Install it with: pip install shap"
            )

        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(self.model.model, 'estimators_') or \
               hasattr(self.model.model, 'get_booster'):
                logger.info("Using TreeExplainer")
                self.explainer = shap.TreeExplainer(self.model.model)
                return

        except Exception as e:
            logger.debug(f"TreeExplainer failed: {e}")

        try:
            # Fall back to KernelExplainer
            if self.background_data is not None:
                logger.info("Using KernelExplainer")

                def predict_fn(X):
                    return self.model.predict(self.model.preprocess(X))

                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    self.background_data
                )
                return

        except Exception as e:
            logger.debug(f"KernelExplainer failed: {e}")

        logger.warning("Could not initialize SHAP explainer")

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            features: Input features
            feature_names: Names of features (optional)

        Returns:
            Dictionary containing explanation
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized")

        # Preprocess features
        processed = self.model.preprocess(features)

        # Get SHAP values
        shap_values = self.explainer.shap_values(processed)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 for binary

        # Get prediction
        prediction = self.model.predict(processed)

        # Create feature importance dict
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(processed.shape[1])]

        feature_importance = {
            name: float(value)
            for name, value in zip(feature_names, shap_values[0])
        }

        # Sort by absolute importance
        sorted_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )

        return {
            'prediction': float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction),
            'feature_importance': sorted_importance,
            'base_value': float(self.explainer.expected_value)
            if hasattr(self.explainer, 'expected_value')
            else None
        }

    def get_global_importance(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get global feature importance across a dataset.

        Args:
            data: Dataset to analyze
            feature_names: Names of features (optional)

        Returns:
            Dictionary of feature importances
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized")

        # Preprocess data
        processed = self.model.preprocess(data)

        # Get SHAP values
        shap_values = self.explainer.shap_values(processed)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance dict
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(processed.shape[1])]

        importance = {
            name: float(value)
            for name, value in zip(feature_names, mean_abs_shap)
        }

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_top_features(
        self,
        features: np.ndarray,
        top_k: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top K most important features for a prediction.

        Args:
            features: Input features
            top_k: Number of top features to return
            feature_names: Names of features (optional)

        Returns:
            List of top features with their contributions
        """
        explanation = self.explain_prediction(features, feature_names)
        importance = explanation['feature_importance']

        top_features = []
        for i, (name, value) in enumerate(importance.items()):
            if i >= top_k:
                break

            top_features.append({
                'feature': name,
                'shap_value': value,
                'rank': i + 1
            })

        return top_features


class LIMEExplainer:
    """
    Model explainability using LIME (Local Interpretable Model-agnostic Explanations).

    Alternative to SHAP for certain use cases.
    """

    def __init__(self, model: BaseModel, feature_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.

        Args:
            model: Model instance to explain
            feature_names: Names of features
        """
        try:
            from lime import lime_tabular
            self.lime_tabular = lime_tabular
        except ImportError:
            raise ImportError(
                "LIME is not installed. "
                "Install it with: pip install lime"
            )

        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def initialize_explainer(
        self,
        training_data: np.ndarray,
        mode: str = "classification"
    ) -> None:
        """
        Initialize LIME explainer with training data.

        Args:
            training_data: Training dataset
            mode: 'classification' or 'regression'
        """
        self.explainer = self.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            mode=mode
        )

    def explain_prediction(
        self,
        features: np.ndarray,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            features: Input features
            num_features: Number of features to include

        Returns:
            Explanation dictionary
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call initialize_explainer() first.")

        # Define prediction function
        def predict_fn(X):
            predictions = []
            for x in X:
                pred = self.model.predict(self.model.preprocess(x))
                predictions.append(pred)
            return np.array(predictions)

        # Get explanation
        exp = self.explainer.explain_instance(
            features,
            predict_fn,
            num_features=num_features
        )

        # Extract feature importance
        feature_importance = dict(exp.as_list())

        return {
            'prediction': float(self.model.predict(self.model.preprocess(features))[0]),
            'feature_importance': feature_importance,
            'score': exp.score
        }

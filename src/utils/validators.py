"""Input validation utilities."""

from pathlib import Path
from typing import Any, Union, List
import numpy as np


def validate_model_path(model_path: str) -> Path:
    """
    Validate that a model path exists and is a file.

    Args:
        model_path: Path to model file

    Returns:
        Path object

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path is not a file
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not path.is_file():
        # Could be a directory (SavedModel format for TensorFlow)
        if not path.is_dir():
            raise ValueError(f"Model path is not a file or directory: {model_path}")

    return path


def validate_features(
    features: Union[List, np.ndarray],
    expected_shape: tuple = None,
    expected_dtype: type = None
) -> np.ndarray:
    """
    Validate input features.

    Args:
        features: Input features
        expected_shape: Expected shape (optional)
        expected_dtype: Expected data type (optional)

    Returns:
        Validated numpy array

    Raises:
        ValueError: If validation fails
    """
    # Convert to numpy array
    if isinstance(features, list):
        try:
            features = np.array(features)
        except Exception as e:
            raise ValueError(f"Could not convert features to array: {e}")

    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be a list or numpy array")

    # Check for empty array
    if features.size == 0:
        raise ValueError("Features array is empty")

    # Check for NaN or inf values
    if np.any(np.isnan(features)):
        raise ValueError("Features contain NaN values")

    if np.any(np.isinf(features)):
        raise ValueError("Features contain infinite values")

    # Validate shape if provided
    if expected_shape is not None:
        if features.shape != expected_shape:
            # Allow for flexible batch dimension
            if len(features.shape) == len(expected_shape):
                for i, (actual, expected) in enumerate(zip(features.shape, expected_shape)):
                    if expected != -1 and actual != expected:
                        raise ValueError(
                            f"Feature shape mismatch at dimension {i}: "
                            f"expected {expected}, got {actual}"
                        )

    # Validate dtype if provided
    if expected_dtype is not None:
        if features.dtype != expected_dtype:
            try:
                features = features.astype(expected_dtype)
            except Exception as e:
                raise ValueError(f"Could not convert features to {expected_dtype}: {e}")

    return features


def validate_api_key(api_key: str, valid_keys: List[str]) -> bool:
    """
    Validate API key.

    Args:
        api_key: API key to validate
        valid_keys: List of valid API keys

    Returns:
        True if valid, False otherwise
    """
    return api_key in valid_keys


def validate_prediction_output(
    prediction: Any,
    allowed_types: tuple = (int, float, list, np.ndarray)
) -> bool:
    """
    Validate prediction output.

    Args:
        prediction: Model prediction
        allowed_types: Tuple of allowed types

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(prediction, allowed_types):
        raise ValueError(
            f"Invalid prediction type: {type(prediction)}. "
            f"Expected one of {allowed_types}"
        )

    return True

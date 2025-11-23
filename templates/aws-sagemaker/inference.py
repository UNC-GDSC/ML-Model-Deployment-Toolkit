"""
SageMaker inference script for model deployment.

This script provides the entry point for SageMaker real-time inference endpoints.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, '/opt/ml/code')


def model_fn(model_dir: str):
    """
    Load the model for inference.

    This function is called once when the endpoint is created.

    Args:
        model_dir: Path to the directory containing model artifacts

    Returns:
        Loaded model object
    """
    logger.info(f"Loading model from {model_dir}")

    # Determine model type from metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            model_type = metadata.get('model_type', 'sklearn')
    else:
        model_type = 'sklearn'

    logger.info(f"Model type: {model_type}")

    # Load model based on type
    if model_type == 'sklearn':
        import joblib
        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path)

    elif model_type == 'tensorflow':
        import tensorflow as tf
        model_path = os.path.join(model_dir, 'model')
        model = tf.keras.models.load_model(model_path)

    elif model_type == 'pytorch':
        import torch
        model_path = os.path.join(model_dir, 'model.pth')
        model = torch.load(model_path)
        model.eval()

    elif model_type == 'onnx':
        import onnxruntime as ort
        model_path = os.path.join(model_dir, 'model.onnx')
        model = ort.InferenceSession(model_path)

    elif model_type == 'xgboost':
        import xgboost as xgb
        model_path = os.path.join(model_dir, 'model.xgb')
        model = xgb.Booster()
        model.load_model(model_path)

    elif model_type == 'lightgbm':
        import lightgbm as lgb
        model_path = os.path.join(model_dir, 'model.txt')
        model = lgb.Booster(model_file=model_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Model loaded successfully")

    return {
        'model': model,
        'model_type': model_type,
        'metadata': metadata if os.path.exists(metadata_path) else {}
    }


def input_fn(request_body: str, request_content_type: str = 'application/json'):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body: The request body
        request_content_type: The request content type

    Returns:
        Deserialized input data
    """
    logger.info(f"Processing input with content type: {request_content_type}")

    if request_content_type == 'application/json':
        data = json.loads(request_body)

        # Handle different input formats
        if isinstance(data, dict):
            if 'instances' in data:
                # Batch prediction format
                return np.array(data['instances'])
            elif 'features' in data:
                # Single prediction with features key
                features = data['features']
                return np.array([features]) if isinstance(features, list) else features
            else:
                # Assume dict values are features
                return np.array([list(data.values())])
        elif isinstance(data, list):
            # Direct array input
            return np.array(data)
        else:
            raise ValueError(f"Unsupported input format: {type(data)}")

    elif request_content_type == 'text/csv':
        # Parse CSV input
        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df.values

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict: Dict[str, Any]):
    """
    Make predictions using the loaded model.

    Args:
        input_data: Preprocessed input data
        model_dict: Dictionary containing model and metadata

    Returns:
        Prediction results
    """
    model = model_dict['model']
    model_type = model_dict['model_type']

    logger.info(f"Making prediction with {model_type} model")
    logger.info(f"Input shape: {input_data.shape if hasattr(input_data, 'shape') else 'N/A'}")

    try:
        if model_type == 'sklearn':
            predictions = model.predict(input_data)

            # Try to get prediction probabilities if available
            try:
                probabilities = model.predict_proba(input_data)
                return {
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist()
                }
            except AttributeError:
                return {'predictions': predictions.tolist()}

        elif model_type == 'tensorflow':
            predictions = model.predict(input_data)
            return {'predictions': predictions.tolist()}

        elif model_type == 'pytorch':
            import torch
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data)
                predictions = model(input_tensor)
                return {'predictions': predictions.numpy().tolist()}

        elif model_type == 'onnx':
            input_name = model.get_inputs()[0].name
            predictions = model.run(None, {input_name: input_data.astype(np.float32)})
            return {'predictions': predictions[0].tolist()}

        elif model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(input_data)
            predictions = model.predict(dmatrix)
            return {'predictions': predictions.tolist()}

        elif model_type == 'lightgbm':
            predictions = model.predict(input_data)
            return {'predictions': predictions.tolist()}

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


def output_fn(prediction, response_content_type: str = 'application/json'):
    """
    Serialize the prediction output.

    Args:
        prediction: The prediction result
        response_content_type: The response content type

    Returns:
        Serialized prediction
    """
    logger.info(f"Formatting output with content type: {response_content_type}")

    if response_content_type == 'application/json':
        return json.dumps(prediction)
    elif response_content_type == 'text/csv':
        # Convert predictions to CSV
        if isinstance(prediction, dict) and 'predictions' in prediction:
            predictions = prediction['predictions']
            return ','.join(map(str, predictions))
        else:
            return str(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


# Health check endpoint for SageMaker
def ping():
    """
    Health check endpoint.

    Returns:
        200 if the model is loaded successfully
    """
    return 200, "OK"

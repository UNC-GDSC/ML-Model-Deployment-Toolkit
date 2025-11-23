"""AWS Lambda function handler for ML model deployment."""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'package'))

from src.models.sklearn_model import SklearnModel
from src.handlers.lambda_handler import LambdaHandler
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=True,
    json_logs=True
)

logger = logging.getLogger(__name__)

# Global model instance (loaded once per container)
MODEL = None
HANDLER = None


def load_model():
    """Load model on cold start."""
    global MODEL, HANDLER

    if MODEL is None:
        logger.info("Loading model...")
        model_path = os.getenv('MODEL_PATH', '/opt/model/model.pkl')
        model_version = os.getenv('MODEL_VERSION', '1.0.0')
        model_type = os.getenv('MODEL_TYPE', 'sklearn')

        # Load appropriate model type
        if model_type == 'sklearn':
            MODEL = SklearnModel(model_path, model_version)
        elif model_type == 'tensorflow':
            from src.models.tensorflow_model import TensorFlowModel
            MODEL = TensorFlowModel(model_path, model_version)
        elif model_type == 'pytorch':
            from src.models.pytorch_model import PyTorchModel
            MODEL = PyTorchModel(model_path, model_version)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        MODEL.load()
        HANDLER = LambdaHandler(MODEL)
        logger.info(f"Model loaded successfully: {model_version}")


def lambda_handler(event, context):
    """
    AWS Lambda function handler.

    Args:
        event: Lambda event object
        context: Lambda context object

    Returns:
        Lambda response object
    """
    # Load model if not already loaded
    load_model()

    # Handle request
    return HANDLER.handle_request(event, context)

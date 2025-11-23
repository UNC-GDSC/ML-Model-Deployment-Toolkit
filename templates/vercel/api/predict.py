"""Vercel serverless function for predictions."""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sklearn_model import SklearnModel
from src.core.base_model import PredictionRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (persists across invocations)
MODEL = None


def load_model():
    """Load model on cold start."""
    global MODEL

    if MODEL is None:
        logger.info("Loading model...")
        model_path = os.getenv('MODEL_PATH', 'models/model.pkl')
        model_version = os.getenv('MODEL_VERSION', '1.0.0')

        # For Vercel, we keep it simple with sklearn only
        # due to size constraints
        MODEL = SklearnModel(model_path, model_version)
        MODEL.load()
        logger.info(f"Model loaded successfully: {model_version}")


def handler(request):
    """
    Vercel serverless function handler for predictions.

    Args:
        request: Vercel request object

    Returns:
        Response with prediction results
    """
    # Load model if not already loaded
    load_model()

    try:
        # Parse request body
        if hasattr(request, 'body'):
            body = request.body
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            if isinstance(body, str):
                body = json.loads(body)
        elif hasattr(request, 'get_json'):
            body = request.get_json()
        else:
            body = {}

        # Validate request
        pred_request = PredictionRequest(**body)

        # Make prediction
        result = MODEL.predict_with_preprocessing(
            pred_request.features,
            pred_request.return_probabilities
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': f"Validation error: {str(e)}"})
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }

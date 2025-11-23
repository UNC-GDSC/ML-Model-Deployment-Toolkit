"""Azure Functions app for ML model deployment."""

import logging
import os
import json
import azure.functions as func

from src.models.sklearn_model import SklearnModel
from src.core.base_model import PredictionRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
MODEL = None


def load_model():
    """Load model on cold start."""
    global MODEL

    if MODEL is None:
        logger.info("Loading model...")
        model_path = os.getenv('MODEL_PATH', 'models/model.pkl')
        model_version = os.getenv('MODEL_VERSION', '1.0.0')
        model_type = os.getenv('MODEL_TYPE', 'sklearn')

        # Load appropriate model type
        if model_type == 'sklearn':
            MODEL = SklearnModel(model_path, model_version)
        # Add other model types as needed

        MODEL.load()
        logger.info(f"Model loaded successfully: {model_version}")


# Create function app
app = func.FunctionApp()


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    load_model()

    health_status = MODEL.health_check()

    return func.HttpResponse(
        body=json.dumps(health_status),
        mimetype="application/json",
        status_code=200
    )


@app.route(route="predict", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Prediction endpoint."""
    try:
        # Load model if not already loaded
        load_model()

        # Parse request body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                body=json.dumps({"error": "Invalid JSON"}),
                mimetype="application/json",
                status_code=400
            )

        # Validate request
        try:
            pred_request = PredictionRequest(**req_body)
        except Exception as e:
            return func.HttpResponse(
                body=json.dumps({"error": f"Validation error: {str(e)}"}),
                mimetype="application/json",
                status_code=400
            )

        # Make prediction
        result = MODEL.predict_with_preprocessing(
            pred_request.features,
            pred_request.return_probabilities
        )

        return func.HttpResponse(
            body=json.dumps(result),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="info", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def info(req: func.HttpRequest) -> func.HttpResponse:
    """Model information endpoint."""
    load_model()

    model_info = MODEL.get_model_info()

    return func.HttpResponse(
        body=json.dumps(model_info),
        mimetype="application/json",
        status_code=200
    )


# Batch prediction endpoint
@app.queue_trigger(arg_name="msg", queue_name="prediction-queue",
                   connection="AzureWebJobsStorage")
def batch_predict(msg: func.QueueMessage) -> None:
    """Process batch predictions from queue."""
    try:
        load_model()

        # Parse message
        batch_request = json.loads(msg.get_body().decode('utf-8'))

        features_list = batch_request.get('features_list', [])
        callback_url = batch_request.get('callback_url')

        # Make predictions
        results = []
        for features in features_list:
            result = MODEL.predict_with_preprocessing(features)
            results.append(result)

        logger.info(f"Processed batch of {len(results)} predictions")

        # TODO: Send results to callback URL or storage
        # This is a placeholder - implement according to your needs

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)


# Blob trigger for model updates
@app.blob_trigger(arg_name="myblob", path="models/{name}",
                  connection="AzureWebJobsStorage")
def model_update(myblob: func.InputStream):
    """Trigger when a new model is uploaded."""
    logging.info(f"New model uploaded: {myblob.name}")
    logging.info(f"Model size: {myblob.length} bytes")

    # TODO: Reload model
    # This would typically involve downloading the new model
    # and reloading it into memory

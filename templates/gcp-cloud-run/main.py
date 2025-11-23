"""GCP Cloud Run main application."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from src.models.sklearn_model import SklearnModel
from src.handlers.cloud_run_handler import CloudRunHandler
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=True,
    json_logs=True
)

logger = logging.getLogger(__name__)

# Global model instance
MODEL = None
HANDLER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global MODEL, HANDLER
    logger.info("Starting up application...")

    model_path = os.getenv('MODEL_PATH', '/app/models/model.pkl')
    model_version = os.getenv('MODEL_VERSION', '1.0.0')
    model_type = os.getenv('MODEL_TYPE', 'sklearn')

    logger.info(f"Loading model from {model_path}")

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
    HANDLER = CloudRunHandler(MODEL)

    logger.info(f"Model loaded successfully: {model_version}")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="ML Model API - Cloud Run",
    description="Machine Learning model deployment on GCP Cloud Run",
    version=os.getenv('MODEL_VERSION', '1.0.0'),
    lifespan=lifespan
)

# Mount the handler's FastAPI app
@app.on_event("startup")
async def startup_event():
    """Mount routes after startup."""
    if HANDLER:
        # Include routes from handler
        handler_app = HANDLER.get_app()
        app.mount("/", handler_app)


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )

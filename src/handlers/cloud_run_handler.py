"""GCP Cloud Run request handler using FastAPI."""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.base_model import BaseModel, PredictionRequest, PredictionResponse, HealthResponse
from src.handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class CloudRunHandler(BaseHandler):
    """Handler for GCP Cloud Run using FastAPI."""

    def __init__(self, model: BaseModel):
        """
        Initialize Cloud Run handler.

        Args:
            model: Loaded BaseModel instance
        """
        super().__init__(model)
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="ML Model API",
            description="Machine Learning model deployment on GCP Cloud Run",
            version=self.model.model_version
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register API routes."""

        @app.get("/", response_model=HealthResponse)
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health = self.health_check()
                return HealthResponse(**health)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint."""
            try:
                result = self.process_prediction(
                    request.features,
                    request.return_probabilities
                )
                return PredictionResponse(**result)
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/info")
        async def model_info():
            """Model information endpoint."""
            return self.model.get_model_info()

        @app.get("/metrics")
        async def metrics():
            """Metrics endpoint."""
            return self.get_metrics()

    def handle_request(self, event: any, context: any = None):
        """
        Not used for Cloud Run (uses FastAPI directly).

        This method is kept for interface compatibility.
        """
        raise NotImplementedError("Cloud Run uses FastAPI app directly")

    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application.

        Returns:
            Configured FastAPI app instance
        """
        return self.app

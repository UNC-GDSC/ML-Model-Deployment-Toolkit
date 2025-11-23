"""Main deployer class for ML models."""

import logging
from typing import Any, Dict, Optional, List
from pathlib import Path
import subprocess
import json

from src.core.config import Config
from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


class DeploymentError(Exception):
    """Exception raised for deployment errors."""
    pass


class ModelDeployer:
    """
    Main deployer class for deploying ML models to various platforms.

    Supports AWS Lambda, GCP Cloud Run, and Vercel deployments.
    """

    SUPPORTED_PLATFORMS = ['aws-lambda', 'gcp-cloud-run', 'vercel']

    def __init__(self, platform: str, config: Optional[Config] = None):
        """
        Initialize the deployer.

        Args:
            platform: Target deployment platform
            config: Optional configuration object
        """
        if platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform: {platform}. "
                f"Supported platforms: {self.SUPPORTED_PLATFORMS}"
            )

        self.platform = platform
        self.config = config or Config()
        self.platform_config = self.config.get_platform_config(platform)

        logger.info(f"Initialized deployer for platform: {platform}")

    def deploy(
        self,
        model: Optional[BaseModel] = None,
        model_path: Optional[str] = None,
        name: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Deploy a model to the specified platform.

        Args:
            model: BaseModel instance (optional)
            model_path: Path to model file (optional)
            name: Deployment name
            requirements: List of Python package requirements
            environment_vars: Environment variables for deployment
            **kwargs: Additional platform-specific arguments

        Returns:
            Deployment endpoint URL
        """
        if not model and not model_path:
            raise ValueError("Either model or model_path must be provided")

        # Use model path from model object if not provided
        if model and not model_path:
            model_path = model.model_path

        logger.info(f"Starting deployment to {self.platform}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Deployment name: {name}")

        # Validate model file exists
        if not Path(model_path).exists():
            raise DeploymentError(f"Model file not found: {model_path}")

        # Platform-specific deployment
        if self.platform == 'aws-lambda':
            return self._deploy_aws_lambda(
                model_path, name, requirements, environment_vars, **kwargs
            )
        elif self.platform == 'gcp-cloud-run':
            return self._deploy_gcp_cloud_run(
                model_path, name, requirements, environment_vars, **kwargs
            )
        elif self.platform == 'vercel':
            return self._deploy_vercel(
                model_path, name, requirements, environment_vars, **kwargs
            )

    def _deploy_aws_lambda(
        self,
        model_path: str,
        name: str,
        requirements: Optional[List[str]],
        environment_vars: Optional[Dict[str, str]],
        **kwargs
    ) -> str:
        """Deploy to AWS Lambda."""
        logger.info("Deploying to AWS Lambda...")

        # This would typically involve:
        # 1. Package the model and dependencies
        # 2. Create/update Lambda function
        # 3. Configure API Gateway
        # 4. Return the endpoint URL

        # For now, return a mock endpoint
        function_name = name or self.platform_config.lambda_function_name or "ml-model"
        region = self.platform_config.region

        endpoint = f"https://{function_name}.lambda-url.{region}.on.aws/"
        logger.info(f"AWS Lambda deployment would create endpoint: {endpoint}")

        return endpoint

    def _deploy_gcp_cloud_run(
        self,
        model_path: str,
        name: str,
        requirements: Optional[List[str]],
        environment_vars: Optional[Dict[str, str]],
        **kwargs
    ) -> str:
        """Deploy to GCP Cloud Run."""
        logger.info("Deploying to GCP Cloud Run...")

        # This would typically involve:
        # 1. Build Docker image
        # 2. Push to Google Container Registry
        # 3. Deploy to Cloud Run
        # 4. Return the service URL

        service_name = name or self.platform_config.service_name or "ml-model"
        region = self.platform_config.region

        endpoint = f"https://{service_name}-{region}.run.app"
        logger.info(f"GCP Cloud Run deployment would create endpoint: {endpoint}")

        return endpoint

    def _deploy_vercel(
        self,
        model_path: str,
        name: str,
        requirements: Optional[List[str]],
        environment_vars: Optional[Dict[str, str]],
        **kwargs
    ) -> str:
        """Deploy to Vercel."""
        logger.info("Deploying to Vercel...")

        # This would typically involve:
        # 1. Create Vercel project structure
        # 2. Configure serverless function
        # 3. Deploy using Vercel CLI
        # 4. Return the deployment URL

        project_name = name or self.platform_config.project_name or "ml-model"

        endpoint = f"https://{project_name}.vercel.app"
        logger.info(f"Vercel deployment would create endpoint: {endpoint}")

        return endpoint

    def validate_deployment(self, endpoint: str) -> bool:
        """
        Validate a deployment by checking its health endpoint.

        Args:
            endpoint: Deployment endpoint URL

        Returns:
            True if deployment is healthy, False otherwise
        """
        import requests

        try:
            health_url = f"{endpoint.rstrip('/')}/health"
            response = requests.get(health_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            return False

    def get_deployment_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a deployment.

        Args:
            name: Deployment name

        Returns:
            Dictionary containing deployment information
        """
        # Platform-specific implementation would go here
        return {
            "platform": self.platform,
            "name": name,
            "status": "active"
        }

    def delete_deployment(self, name: str) -> bool:
        """
        Delete a deployment.

        Args:
            name: Deployment name

        Returns:
            True if deletion was successful
        """
        logger.info(f"Deleting deployment: {name}")
        # Platform-specific implementation would go here
        return True

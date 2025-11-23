"""ML Model Deployment Toolkit - Production-ready ML model deployment."""

__version__ = "1.0.0"
__author__ = "UNC Google Developer Student Club"

from src.core.base_model import BaseModel
from src.core.deployer import ModelDeployer

__all__ = ["BaseModel", "ModelDeployer", "__version__"]

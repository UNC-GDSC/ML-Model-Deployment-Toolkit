"""Configuration management for ML deployment."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AWSConfig(BaseModel):
    """AWS-specific configuration."""

    region: str = Field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    lambda_function_name: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    lambda_memory: int = Field(default_factory=lambda: int(os.getenv("AWS_LAMBDA_MEMORY", "1024")))
    lambda_timeout: int = Field(default_factory=lambda: int(os.getenv("AWS_LAMBDA_TIMEOUT", "30")))
    s3_bucket: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_S3_BUCKET"))
    api_gateway_stage: str = Field(default_factory=lambda: os.getenv("AWS_API_GATEWAY_STAGE", "prod"))


class GCPConfig(BaseModel):
    """GCP-specific configuration."""

    project_id: Optional[str] = Field(default_factory=lambda: os.getenv("GCP_PROJECT_ID"))
    region: str = Field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    service_name: Optional[str] = Field(default_factory=lambda: os.getenv("GCP_SERVICE_NAME"))
    memory: str = Field(default_factory=lambda: os.getenv("GCP_MEMORY", "2Gi"))
    cpu: str = Field(default_factory=lambda: os.getenv("GCP_CPU", "1"))
    max_instances: int = Field(default_factory=lambda: int(os.getenv("GCP_MAX_INSTANCES", "100")))
    min_instances: int = Field(default_factory=lambda: int(os.getenv("GCP_MIN_INSTANCES", "0")))
    timeout: int = Field(default_factory=lambda: int(os.getenv("GCP_TIMEOUT", "300")))


class VercelConfig(BaseModel):
    """Vercel-specific configuration."""

    project_name: Optional[str] = Field(default_factory=lambda: os.getenv("VERCEL_PROJECT_NAME"))
    org_id: Optional[str] = Field(default_factory=lambda: os.getenv("VERCEL_ORG_ID"))
    token: Optional[str] = Field(default_factory=lambda: os.getenv("VERCEL_TOKEN"))
    region: str = Field(default_factory=lambda: os.getenv("VERCEL_REGION", "iad1"))


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key_enabled: bool = Field(default_factory=lambda: os.getenv("API_KEY_ENABLED", "true").lower() == "true")
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("API_KEY"))
    rate_limit_enabled: bool = Field(default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true")
    rate_limit_requests: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60")))
    cors_enabled: bool = Field(default_factory=lambda: os.getenv("CORS_ENABLED", "true").lower() == "true")
    cors_origins: str = Field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    metrics_enabled: bool = Field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    prometheus_port: int = Field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "8000")))
    structured_logging: bool = Field(default_factory=lambda: os.getenv("STRUCTURED_LOGGING", "true").lower() == "true")


class ModelConfig(BaseModel):
    """Model-specific configuration."""

    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "/models/model.pkl"))
    model_version: str = Field(default_factory=lambda: os.getenv("MODEL_VERSION", "1.0.0"))
    model_type: str = Field(default_factory=lambda: os.getenv("MODEL_TYPE", "sklearn"))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    max_prediction_time: int = Field(default_factory=lambda: int(os.getenv("MAX_PREDICTION_TIME", "10")))


class Config(BaseModel):
    """Main configuration class."""

    # Environment
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Platform configs
    aws: AWSConfig = Field(default_factory=AWSConfig)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    vercel: VercelConfig = Field(default_factory=VercelConfig)

    # Other configs
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    def get_platform_config(self, platform: str) -> BaseModel:
        """
        Get platform-specific configuration.

        Args:
            platform: Platform name ('aws', 'gcp', or 'vercel')

        Returns:
            Platform-specific configuration object
        """
        platform_map = {
            'aws': self.aws,
            'aws-lambda': self.aws,
            'gcp': self.gcp,
            'gcp-cloud-run': self.gcp,
            'vercel': self.vercel
        }

        if platform not in platform_map:
            raise ValueError(f"Unknown platform: {platform}")

        return platform_map[platform]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()


# Global config instance
config = Config()

"""
OpenAPI/Swagger documentation generator for ML model APIs.

Automatically generates API documentation from model metadata.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class OpenAPIGenerator:
    """
    Generate OpenAPI 3.0 specification for ML model APIs.

    Creates standardized API documentation including:
    - Endpoints
    - Request/response schemas
    - Authentication
    - Examples
    """

    def __init__(
        self,
        title: str = "ML Model API",
        version: str = "1.0.0",
        description: str = "Machine Learning Model Inference API"
    ):
        """
        Initialize OpenAPI generator.

        Args:
            title: API title
            version: API version
            description: API description
        """
        self.title = title
        self.version = version
        self.description = description

    def generate(
        self,
        model_info: Dict[str, Any],
        base_url: str = "https://api.example.com",
        auth_type: str = "bearer"
    ) -> Dict[str, Any]:
        """
        Generate OpenAPI specification.

        Args:
            model_info: Model metadata
            base_url: Base URL for API
            auth_type: Authentication type (bearer, apikey, none)

        Returns:
            OpenAPI 3.0 specification dictionary
        """
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
                "contact": {
                    "name": "API Support",
                    "email": "support@example.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": base_url,
                    "description": "Production server"
                }
            ],
            "paths": self._generate_paths(model_info),
            "components": self._generate_components(model_info, auth_type),
            "tags": [
                {
                    "name": "prediction",
                    "description": "Model prediction endpoints"
                },
                {
                    "name": "health",
                    "description": "Health check endpoints"
                },
                {
                    "name": "model",
                    "description": "Model information endpoints"
                }
            ]
        }

        return spec

    def _generate_paths(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API paths."""
        paths = {
            "/health": {
                "get": {
                    "tags": ["health"],
                    "summary": "Health check",
                    "description": "Check if the API is running",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    },
                                    "example": {
                                        "status": "healthy",
                                        "timestamp": "2024-01-15T10:30:00Z",
                                        "model_loaded": True
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "tags": ["prediction"],
                    "summary": "Make prediction",
                    "description": "Get model prediction for input features",
                    "operationId": "predict",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictionRequest"
                                },
                                "examples": {
                                    "single": {
                                        "summary": "Single prediction",
                                        "value": {
                                            "features": [5.1, 3.5, 1.4, 0.2]
                                        }
                                    },
                                    "batch": {
                                        "summary": "Batch prediction",
                                        "value": {
                                            "instances": [
                                                [5.1, 3.5, 1.4, 0.2],
                                                [6.2, 3.4, 5.4, 2.3]
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PredictionResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/info": {
                "get": {
                    "tags": ["model"],
                    "summary": "Get model information",
                    "description": "Retrieve model metadata and statistics",
                    "operationId": "getModelInfo",
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelInfoResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "tags": ["health"],
                    "summary": "Get metrics",
                    "description": "Retrieve Prometheus-compatible metrics",
                    "operationId": "getMetrics",
                    "responses": {
                        "200": {
                            "description": "Metrics in Prometheus format",
                            "content": {
                                "text/plain": {
                                    "schema": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return paths

    def _generate_components(
        self,
        model_info: Dict[str, Any],
        auth_type: str
    ) -> Dict[str, Any]:
        """Generate API components (schemas, security)."""
        components = {
            "schemas": {
                "PredictionRequest": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Input features for single prediction"
                        },
                        "instances": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "description": "Multiple instances for batch prediction"
                        }
                    },
                    "oneOf": [
                        {"required": ["features"]},
                        {"required": ["instances"]}
                    ]
                },
                "PredictionResponse": {
                    "type": "object",
                    "properties": {
                        "prediction": {
                            "oneOf": [
                                {"type": "number"},
                                {"type": "array", "items": {"type": "number"}},
                                {"type": "object"}
                            ],
                            "description": "Model prediction"
                        },
                        "probabilities": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Class probabilities (classification only)"
                        },
                        "latency_ms": {
                            "type": "number",
                            "description": "Inference latency in milliseconds"
                        },
                        "model_version": {
                            "type": "string",
                            "description": "Model version used"
                        }
                    },
                    "required": ["prediction"]
                },
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "unhealthy"]
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "model_loaded": {
                            "type": "boolean"
                        }
                    },
                    "required": ["status"]
                },
                "ModelInfoResponse": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "model_version": {"type": "string"},
                        "model_type": {"type": "string"},
                        "framework": {"type": "string"},
                        "input_shape": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "output_shape": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message"
                        },
                        "code": {
                            "type": "string",
                            "description": "Error code"
                        },
                        "details": {
                            "type": "object",
                            "description": "Additional error details"
                        }
                    },
                    "required": ["error"]
                }
            },
            "securitySchemes": {}
        }

        # Add authentication based on type
        if auth_type == "bearer":
            components["securitySchemes"]["BearerAuth"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        elif auth_type == "apikey":
            components["securitySchemes"]["ApiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }

        return components

    def save_to_file(self, spec: Dict[str, Any], filepath: str):
        """Save OpenAPI spec to file."""
        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2)
        logger.info(f"OpenAPI specification saved to {filepath}")

    def generate_fastapi_code(self, spec: Dict[str, Any]) -> str:
        """Generate FastAPI code from OpenAPI spec."""
        code = '''"""
Auto-generated FastAPI application from OpenAPI specification.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Union, Optional
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="{title}",
    version="{version}",
    description="{description}"
)

# Security
security = HTTPBearer()

# Pydantic models
class PredictionRequest(BaseModel):
    features: Optional[List[float]] = None
    instances: Optional[List[List[float]]] = None

class PredictionResponse(BaseModel):
    prediction: Union[float, List[float], dict]
    latency_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

# Load model (implement this)
def load_model():
    # TODO: Implement model loading
    pass

model = load_model()

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    from datetime import datetime
    return {{
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Validate token (implement this)
    # validate_token(credentials.credentials)

    # Make prediction
    import time
    start = time.time()

    if request.features:
        features = np.array(request.features).reshape(1, -1)
    elif request.instances:
        features = np.array(request.instances)
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    prediction = model.predict(features)
    latency = (time.time() - start) * 1000

    return {{
        "prediction": prediction.tolist(),
        "latency_ms": latency,
        "model_version": "1.0.0"
    }}

@app.get("/info")
async def get_info():
    return {{
        "model_name": "{title}",
        "model_version": "{version}",
        "framework": "scikit-learn"
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''.format(
            title=spec['info']['title'],
            version=spec['info']['version'],
            description=spec['info']['description']
        )

        return code


def generate_docs_for_model(
    model_path: str,
    output_dir: str = "./docs"
):
    """
    Generate complete API documentation for a model.

    Args:
        model_path: Path to model file
        output_dir: Output directory for documentation
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Model info (in production, extract from actual model)
    model_info = {
        "model_type": "sklearn",
        "framework": "scikit-learn",
        "input_shape": [None, 4],
        "output_shape": [None, 3]
    }

    # Generate OpenAPI spec
    generator = OpenAPIGenerator(
        title="ML Model API",
        version="1.0.0",
        description="Production ML Model Inference API"
    )

    spec = generator.generate(model_info)

    # Save OpenAPI spec
    spec_path = os.path.join(output_dir, "openapi.json")
    generator.save_to_file(spec, spec_path)

    # Generate FastAPI code
    fastapi_code = generator.generate_fastapi_code(spec)
    code_path = os.path.join(output_dir, "app.py")

    with open(code_path, 'w') as f:
        f.write(fastapi_code)

    logger.info(f"API documentation generated in {output_dir}")
    logger.info(f"  - OpenAPI spec: {spec_path}")
    logger.info(f"  - FastAPI code: {code_path}")

    return spec_path, code_path

"""Model versioning and management system."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import hashlib

from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_type: str
    model_path: str
    created_at: str
    checksum: str
    metrics: Dict[str, float]
    tags: List[str]
    description: str
    is_active: bool = False
    deployment_status: str = "inactive"  # inactive, staging, production, deprecated


class ModelRegistry:
    """
    Model registry for version management.

    Tracks multiple model versions and manages deployments.
    """

    def __init__(self, registry_path: str = ".model_registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.versions: Dict[str, ModelVersion] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                self.versions = {
                    version: ModelVersion(**info)
                    for version, info in data.items()
                }

                logger.info(f"Loaded {len(self.versions)} model versions from registry")

            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.versions = {}
        else:
            logger.info("No existing registry found, creating new one")

    def save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {
                version: asdict(info)
                for version, info in self.versions.items()
            }

            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.versions)} model versions to registry")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(
        self,
        version: str,
        model_type: str,
        model_path: str,
        metrics: Dict[str, float] = None,
        tags: List[str] = None,
        description: str = ""
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            version: Version string (e.g., "1.0.0")
            model_type: Type of model (sklearn, tensorflow, etc.)
            model_path: Path to model file
            metrics: Model performance metrics
            tags: Tags for categorization
            description: Description of the model

        Returns:
            ModelVersion object
        """
        # Calculate checksum
        checksum = self._calculate_checksum(model_path)

        # Create version object
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            model_path=model_path,
            created_at=datetime.utcnow().isoformat(),
            checksum=checksum,
            metrics=metrics or {},
            tags=tags or [],
            description=description
        )

        self.versions[version] = model_version
        self.save_registry()

        logger.info(f"Registered model version {version}")

        return model_version

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """
        Get a specific model version.

        Args:
            version: Version string

        Returns:
            ModelVersion object or None
        """
        return self.versions.get(version)

    def list_versions(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        deployment_status: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        List model versions with optional filtering.

        Args:
            model_type: Filter by model type
            tags: Filter by tags (any match)
            deployment_status: Filter by deployment status

        Returns:
            List of ModelVersion objects
        """
        versions = list(self.versions.values())

        if model_type:
            versions = [v for v in versions if v.model_type == model_type]

        if tags:
            versions = [
                v for v in versions
                if any(tag in v.tags for tag in tags)
            ]

        if deployment_status:
            versions = [
                v for v in versions
                if v.deployment_status == deployment_status
            ]

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def get_active_version(self) -> Optional[ModelVersion]:
        """
        Get the currently active model version.

        Returns:
            Active ModelVersion or None
        """
        active = [v for v in self.versions.values() if v.is_active]
        return active[0] if active else None

    def set_active_version(self, version: str) -> None:
        """
        Set a version as active.

        Args:
            version: Version string to activate
        """
        # Deactivate all versions
        for v in self.versions.values():
            v.is_active = False

        # Activate specified version
        if version in self.versions:
            self.versions[version].is_active = True
            logger.info(f"Set version {version} as active")
            self.save_registry()
        else:
            raise ValueError(f"Version {version} not found in registry")

    def promote_version(self, version: str, status: str) -> None:
        """
        Promote a version to a deployment status.

        Args:
            version: Version string
            status: Deployment status (staging, production, deprecated)
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found in registry")

        valid_statuses = ["inactive", "staging", "production", "deprecated"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of {valid_statuses}")

        self.versions[version].deployment_status = status
        logger.info(f"Promoted version {version} to {status}")
        self.save_registry()

    def compare_versions(
        self,
        version1: str,
        version2: str,
        metric: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions by a metric.

        Args:
            version1: First version string
            version2: Second version string
            metric: Metric to compare

        Returns:
            Comparison results
        """
        v1 = self.versions.get(version1)
        v2 = self.versions.get(version2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        v1_metric = v1.metrics.get(metric)
        v2_metric = v2.metrics.get(metric)

        if v1_metric is None or v2_metric is None:
            raise ValueError(f"Metric {metric} not found in one or both versions")

        return {
            'version1': version1,
            'version2': version2,
            'metric': metric,
            'value1': v1_metric,
            'value2': v2_metric,
            'difference': v2_metric - v1_metric,
            'percent_change': ((v2_metric - v1_metric) / v1_metric) * 100,
            'better': version2 if v2_metric > v1_metric else version1
        }

    def delete_version(self, version: str) -> None:
        """
        Delete a model version from registry.

        Args:
            version: Version string to delete
        """
        if version in self.versions:
            del self.versions[version]
            logger.info(f"Deleted version {version} from registry")
            self.save_registry()
        else:
            raise ValueError(f"Version {version} not found in registry")

    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of checksum
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def export_metadata(self, output_path: str) -> None:
        """
        Export registry metadata to a file.

        Args:
            output_path: Path to output file
        """
        data = {
            version: asdict(info)
            for version, info in self.versions.items()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported registry metadata to {output_path}")

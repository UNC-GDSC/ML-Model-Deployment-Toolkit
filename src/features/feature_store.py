"""Feature store integration for ML models."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Simple feature store for managing ML features.

    Provides feature serving and versioning.
    """

    def __init__(self, storage_backend: str = "memory"):
        """
        Initialize feature store.

        Args:
            storage_backend: Storage backend (memory, redis, database)
        """
        self.storage_backend = storage_backend
        self.features = {}  # entity_id -> features
        self.feature_schemas = {}  # feature_name -> schema
        self.feature_metadata = {}  # feature_name -> metadata

        logger.info(f"Initialized feature store with {storage_backend} backend")

    def register_feature(
        self,
        feature_name: str,
        feature_type: str,
        description: str = "",
        tags: List[str] = None
    ):
        """
        Register a feature.

        Args:
            feature_name: Name of the feature
            feature_type: Data type of the feature
            description: Feature description
            tags: Feature tags
        """
        self.feature_schemas[feature_name] = {
            'name': feature_name,
            'type': feature_type,
            'description': description,
            'tags': tags or [],
            'registered_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Registered feature: {feature_name}")

    def store_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Store features for an entity.

        Args:
            entity_id: Entity identifier
            features: Feature dictionary
            timestamp: Feature timestamp
        """
        if entity_id not in self.features:
            self.features[entity_id] = []

        feature_record = {
            'features': features,
            'timestamp': (timestamp or datetime.utcnow()).isoformat(),
            'version': len(self.features[entity_id])
        }

        self.features[entity_id].append(feature_record)
        logger.debug(f"Stored features for entity: {entity_id}")

    def get_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Retrieve features for an entity.

        Args:
            entity_id: Entity identifier
            feature_names: Specific features to retrieve (None = all)
            timestamp: Point-in-time for features

        Returns:
            Feature dictionary
        """
        if entity_id not in self.features:
            logger.warning(f"No features found for entity: {entity_id}")
            return {}

        # Get latest features (or point-in-time if specified)
        feature_records = self.features[entity_id]

        if timestamp:
            # Find features at specific timestamp
            valid_records = [
                r for r in feature_records
                if datetime.fromisoformat(r['timestamp']) <= timestamp
            ]
            if not valid_records:
                return {}
            feature_record = max(valid_records, key=lambda x: x['timestamp'])
        else:
            feature_record = feature_records[-1]  # Latest

        features = feature_record['features']

        # Filter by feature names if specified
        if feature_names:
            features = {k: v for k, v in features.items() if k in feature_names}

        return features

    def create_feature_group(
        self,
        group_name: str,
        feature_names: List[str],
        description: str = ""
    ):
        """
        Create a feature group.

        Args:
            group_name: Name of the feature group
            feature_names: Features in the group
            description: Group description
        """
        self.feature_metadata[group_name] = {
            'type': 'group',
            'features': feature_names,
            'description': description,
            'created_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Created feature group: {group_name}")

    def get_feature_group(
        self,
        entity_id: str,
        group_name: str
    ) -> Dict[str, Any]:
        """
        Retrieve all features in a group.

        Args:
            entity_id: Entity identifier
            group_name: Feature group name

        Returns:
            Feature dictionary
        """
        if group_name not in self.feature_metadata:
            raise ValueError(f"Unknown feature group: {group_name}")

        group = self.feature_metadata[group_name]
        return self.get_features(entity_id, group['features'])

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch retrieve online features.

        Args:
            entity_ids: List of entity identifiers
            feature_names: Features to retrieve

        Returns:
            List of feature dictionaries
        """
        results = []

        for entity_id in entity_ids:
            features = self.get_features(entity_id, feature_names)
            results.append({
                'entity_id': entity_id,
                'features': features
            })

        return results

    def compute_derived_features(
        self,
        entity_id: str,
        transformations: Dict[str, callable]
    ) -> Dict[str, Any]:
        """
        Compute derived features.

        Args:
            entity_id: Entity identifier
            transformations: Dictionary of feature_name -> transformation_function

        Returns:
            Derived features
        """
        # Get base features
        base_features = self.get_features(entity_id)

        # Compute derived features
        derived = {}
        for feature_name, transform_func in transformations.items():
            try:
                derived[feature_name] = transform_func(base_features)
            except Exception as e:
                logger.error(f"Failed to compute derived feature {feature_name}: {e}")

        return derived


class FeatureTransformer:
    """
    Transform features for model input.
    """

    def __init__(self, feature_config: Dict[str, Any]):
        """
        Initialize feature transformer.

        Args:
            feature_config: Feature transformation configuration
        """
        self.feature_config = feature_config

    def transform(self, features: Dict[str, Any]) -> Any:
        """
        Transform features to model input format.

        Args:
            features: Raw feature dictionary

        Returns:
            Transformed features
        """
        import numpy as np

        # Extract features in configured order
        feature_values = []

        for feature_name in self.feature_config.get('feature_order', features.keys()):
            value = features.get(feature_name)

            # Apply transformations
            if feature_name in self.feature_config.get('transformations', {}):
                transform = self.feature_config['transformations'][feature_name]

                if transform == 'log':
                    value = np.log1p(value)
                elif transform == 'sqrt':
                    value = np.sqrt(value)
                elif transform == 'normalize':
                    # Would need statistics for proper normalization
                    pass

            feature_values.append(value)

        return np.array(feature_values)

"""Redis-based caching for predictions."""

import logging
import json
import hashlib
from typing import Any, Optional
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based cache for ML predictions.

    Provides distributed caching across multiple instances.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 300
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            ttl_seconds: Time-to-live for cache entries
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not installed. "
                "Install it with: pip install redis"
            )

        self.ttl_seconds = ttl_seconds

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _generate_key(self, data: Any, prefix: str = "prediction") -> str:
        """
        Generate cache key from data.

        Args:
            data: Data to hash
            prefix: Key prefix

        Returns:
            Cache key
        """
        # Convert data to stable JSON string
        json_str = json.dumps(data, sort_keys=True)

        # Hash the JSON string
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()

        return f"{prefix}:{hash_value}"

    def get(self, features: Any) -> Optional[dict]:
        """
        Get cached prediction.

        Args:
            features: Input features

        Returns:
            Cached prediction or None
        """
        if self.client is None:
            return None

        try:
            key = self._generate_key(features)
            value = self.client.get(key)

            if value:
                logger.debug(f"Cache hit for key: {key[:32]}...")
                return json.loads(value)

            logger.debug(f"Cache miss for key: {key[:32]}...")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, features: Any, prediction: dict) -> bool:
        """
        Cache a prediction.

        Args:
            features: Input features
            prediction: Prediction result

        Returns:
            True if successful
        """
        if self.client is None:
            return False

        try:
            key = self._generate_key(features)
            value = json.dumps(prediction)

            self.client.setex(
                key,
                timedelta(seconds=self.ttl_seconds),
                value
            )

            logger.debug(f"Cached prediction for key: {key[:32]}...")
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, features: Any) -> bool:
        """
        Delete cached prediction.

        Args:
            features: Input features

        Returns:
            True if successful
        """
        if self.client is None:
            return False

        try:
            key = self._generate_key(features)
            self.client.delete(key)
            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear_all(self) -> bool:
        """
        Clear all cached predictions.

        Returns:
            True if successful
        """
        if self.client is None:
            return False

        try:
            self.client.flushdb()
            logger.info("Cleared all cache entries")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        if self.client is None:
            return {}

        try:
            info = self.client.info()

            return {
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) /
                    max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
                ),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_keys': self.client.dbsize()
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def set_batch(self, batch_data: list) -> bool:
        """
        Cache multiple predictions at once.

        Args:
            batch_data: List of (features, prediction) tuples

        Returns:
            True if successful
        """
        if self.client is None:
            return False

        try:
            pipe = self.client.pipeline()

            for features, prediction in batch_data:
                key = self._generate_key(features)
                value = json.dumps(prediction)
                pipe.setex(key, timedelta(seconds=self.ttl_seconds), value)

            pipe.execute()
            logger.info(f"Cached {len(batch_data)} predictions")
            return True

        except Exception as e:
            logger.error(f"Batch cache error: {e}")
            return False

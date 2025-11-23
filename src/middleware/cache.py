"""Caching middleware for predictions."""

import json
import hashlib
import logging
from typing import Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict

from fastapi import Request

logger = logging.getLogger(__name__)


class CacheMiddleware:
    """
    LRU cache for prediction results.

    Caches prediction results to improve response time for repeated queries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300
    ):
        """
        Initialize cache middleware.

        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _generate_cache_key(self, data: Any) -> str:
        """
        Generate cache key from request data.

        Args:
            data: Request data

        Returns:
            Cache key
        """
        # Convert data to stable JSON string
        json_str = json.dumps(data, sort_keys=True)

        # Hash the JSON string
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check if entry is expired
        if datetime.utcnow() > entry['expires']:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)

        self.hits += 1
        logger.debug(f"Cache hit for key: {key[:16]}...")

        return entry['value']

    def set(self, key: str, value: dict) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")

        # Add new entry
        self.cache[key] = {
            'value': value,
            'expires': datetime.utcnow() + timedelta(seconds=self.ttl_seconds),
            'created': datetime.utcnow()
        }

        logger.debug(f"Cached value for key: {key[:16]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expires']
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }

    async def __call__(self, request: Request, call_next):
        """
        Process request with caching.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (from cache or handler)
        """
        # Only cache GET requests to /predict
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)

        # Get request body
        body = await request.body()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(data)

        # Check cache
        cached_value = self.get(cache_key)

        if cached_value is not None:
            # Return cached response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=cached_value,
                headers={'X-Cache': 'HIT'}
            )

        # Call next handler
        response = await call_next(request)

        # Cache the response if successful
        if response.status_code == 200:
            # This is a simplified version
            # In production, you'd need to properly handle streaming responses
            pass

        # Add cache miss header
        response.headers['X-Cache'] = 'MISS'

        return response

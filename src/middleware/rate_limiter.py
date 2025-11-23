"""Rate limiting middleware."""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits the number of requests per time window.
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Time window in seconds
            burst_size: Maximum burst size (defaults to requests_per_window)
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst_size = burst_size or requests_per_window

        # Client tracking
        self.clients: Dict[str, dict] = defaultdict(
            lambda: {
                'tokens': self.burst_size,
                'last_update': time.time()
            }
        )

    def _get_client_key(self, request: Request) -> str:
        """
        Get unique client identifier.

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Try to get API key from user state
        if hasattr(request.state, 'user'):
            user = request.state.user
            if isinstance(user, dict) and 'sub' in user:
                return f"user:{user['sub']}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _refill_tokens(self, client_data: dict) -> None:
        """
        Refill tokens based on elapsed time.

        Args:
            client_data: Client data dictionary
        """
        now = time.time()
        elapsed = now - client_data['last_update']

        # Calculate tokens to add
        tokens_to_add = (
            elapsed * self.requests_per_window / self.window_seconds
        )

        # Update tokens (capped at burst size)
        client_data['tokens'] = min(
            self.burst_size,
            client_data['tokens'] + tokens_to_add
        )
        client_data['last_update'] = now

    def is_allowed(self, client_key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Args:
            client_key: Client identifier

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_data = self.clients[client_key]

        # Refill tokens
        self._refill_tokens(client_data)

        # Check if request is allowed
        if client_data['tokens'] >= 1.0:
            client_data['tokens'] -= 1.0
            allowed = True
        else:
            allowed = False

        # Calculate rate limit info
        info = {
            'limit': self.requests_per_window,
            'remaining': int(client_data['tokens']),
            'reset': int(client_data['last_update'] + self.window_seconds)
        }

        return allowed, info

    async def __call__(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from next handler

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip rate limiting for health check
        if request.url.path in ["/health", "/"]:
            return await call_next(request)

        client_key = self._get_client_key(request)
        allowed, info = self.is_allowed(client_key)

        # Add rate limit headers
        headers = {
            'X-RateLimit-Limit': str(info['limit']),
            'X-RateLimit-Remaining': str(info['remaining']),
            'X-RateLimit-Reset': str(info['reset'])
        }

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=headers
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    def cleanup_old_clients(self, max_age_seconds: int = 3600):
        """
        Remove old client entries.

        Args:
            max_age_seconds: Maximum age of client entries
        """
        now = time.time()
        cutoff = now - max_age_seconds

        # Remove old entries
        old_clients = [
            key for key, data in self.clients.items()
            if data['last_update'] < cutoff
        ]

        for key in old_clients:
            del self.clients[key]

        if old_clients:
            logger.info(f"Cleaned up {len(old_clients)} old client entries")

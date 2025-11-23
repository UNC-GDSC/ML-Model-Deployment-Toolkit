"""Authentication middleware."""

import logging
from typing import Optional, Callable
from datetime import datetime, timedelta
import hashlib
import secrets

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

security = HTTPBearer()


class AuthMiddleware:
    """
    Authentication middleware for API endpoints.

    Supports API key and JWT token authentication.
    """

    def __init__(
        self,
        api_keys: Optional[list] = None,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256"
    ):
        """
        Initialize authentication middleware.

        Args:
            api_keys: List of valid API keys
            jwt_secret: Secret key for JWT token validation
            jwt_algorithm: JWT algorithm
        """
        self.api_keys = set(api_keys or [])
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify API key.

        Args:
            api_key: API key to verify

        Returns:
            True if valid, False otherwise
        """
        # Use constant-time comparison to prevent timing attacks
        return any(
            secrets.compare_digest(api_key, valid_key)
            for valid_key in self.api_keys
        )

    def verify_jwt_token(self, token: str) -> dict:
        """
        Verify JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid
        """
        if not self.jwt_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT authentication not configured"
            )

        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            return payload
        except JWTError as e:
            logger.error(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )

    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ):
        """
        Process request with authentication.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from next handler

        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication for health check
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header"
            )

        # Check for Bearer token (JWT)
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # Try JWT first
            if self.jwt_secret:
                try:
                    payload = self.verify_jwt_token(token)
                    request.state.user = payload
                    return await call_next(request)
                except HTTPException:
                    pass  # Fall through to API key check

            # Try API key
            if self.verify_api_key(token):
                request.state.user = {"api_key": True}
                return await call_next(request)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    @staticmethod
    def create_access_token(
        data: dict,
        secret_key: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new JWT access token.

        Args:
            data: Data to encode in token
            secret_key: Secret key for signing
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)

        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(
            to_encode,
            secret_key,
            algorithm="HS256"
        )

        return encoded_jwt

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """
        Hash an API key for secure storage.

        Args:
            api_key: API key to hash

        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def generate_api_key(prefix: str = "ml_") -> str:
        """
        Generate a new API key.

        Args:
            prefix: Prefix for the API key

        Returns:
            Generated API key
        """
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}{random_part}"

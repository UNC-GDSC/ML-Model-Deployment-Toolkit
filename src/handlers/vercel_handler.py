"""Vercel serverless function handler."""

import json
import logging
from typing import Any, Dict
from urllib.parse import parse_qs

from src.handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class VercelHandler(BaseHandler):
    """Handler for Vercel serverless functions."""

    def handle_request(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Handle Vercel serverless function request.

        Args:
            event: Vercel request object
            context: Vercel context object (unused)

        Returns:
            Vercel response object
        """
        try:
            # Extract request information
            method = event.get('method', 'GET')
            path = event.get('path', '/')
            body = event.get('body', {})

            logger.info(f"Handling {method} request to {path}")

            # Route to appropriate handler
            if path == '/' or path == '/health':
                return self._health_response()
            elif path == '/predict' or path == '/api/predict':
                if method == 'POST':
                    return self._predict_response(body)
                else:
                    return self._error_response('Method not allowed', 405)
            elif path == '/info' or path == '/api/info':
                return self._info_response()
            else:
                return self._error_response(f'Path not found: {path}', 404)

        except Exception as e:
            logger.error(f"Request handling failed: {e}", exc_info=True)
            return self._error_response(str(e), 500)

    def _predict_response(self, body: Any) -> Dict[str, Any]:
        """Handle prediction request."""
        try:
            # Parse body if string
            if isinstance(body, str):
                body = json.loads(body)

            # Validate request
            request = self.validate_request(body)

            # Make prediction
            result = self.process_prediction(
                request.features,
                request.return_probabilities
            )

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }

        except ValueError as e:
            return self._error_response(f"Validation error: {e}", 400)
        except Exception as e:
            return self._error_response(f"Prediction error: {e}", 500)

    def _health_response(self) -> Dict[str, Any]:
        """Handle health check request."""
        health = self.health_check()
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(health)
        }

    def _info_response(self) -> Dict[str, Any]:
        """Handle model info request."""
        info = self.model.get_model_info()
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(info)
        }

    def _error_response(self, message: str, status_code: int) -> Dict[str, Any]:
        """Format error response."""
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': message,
                'status_code': status_code
            })
        }


# Vercel serverless function entry point format
def create_vercel_handler(model):
    """
    Create a Vercel-compatible handler function.

    Args:
        model: Loaded BaseModel instance

    Returns:
        Handler function for Vercel
    """
    handler = VercelHandler(model)

    def vercel_handler(request):
        """Vercel handler function."""
        event = {
            'method': request.get('method', 'GET'),
            'path': request.get('path', '/'),
            'body': request.get('body'),
            'headers': request.get('headers', {}),
            'query': request.get('query', {})
        }

        response = handler.handle_request(event)

        return {
            'statusCode': response.get('statusCode', 200),
            'headers': response.get('headers', {}),
            'body': response.get('body', '{}')
        }

    return vercel_handler

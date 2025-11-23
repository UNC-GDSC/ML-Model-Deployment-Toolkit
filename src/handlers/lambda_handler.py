"""AWS Lambda request handler."""

import json
import logging
from typing import Any, Dict

from src.handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LambdaHandler(BaseHandler):
    """Handler for AWS Lambda events."""

    def handle_request(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Handle AWS Lambda event.

        Args:
            event: Lambda event object
            context: Lambda context object

        Returns:
            Lambda response object with statusCode, headers, and body
        """
        try:
            # Extract path from event
            path = event.get('path', '/')
            http_method = event.get('httpMethod', 'GET')

            logger.info(f"Handling {http_method} request to {path}")

            # Route to appropriate handler
            if path == '/health' or path == '/':
                return self._health_response()
            elif path == '/predict' and http_method == 'POST':
                return self._predict_response(event)
            elif path == '/info':
                return self._info_response()
            else:
                return self._error_response(
                    f"Path not found: {path}",
                    404
                )

        except Exception as e:
            logger.error(f"Request handling failed: {e}", exc_info=True)
            return self._error_response(str(e), 500)

    def _predict_response(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request."""
        try:
            # Parse body
            body = event.get('body', '{}')
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

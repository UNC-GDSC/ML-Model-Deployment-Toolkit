"""Vercel serverless function for health checks."""

import json
from datetime import datetime


def handler(request):
    """
    Health check endpoint.

    Args:
        request: Vercel request object

    Returns:
        Health status response
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'ml-model-vercel'
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(health_status)
    }

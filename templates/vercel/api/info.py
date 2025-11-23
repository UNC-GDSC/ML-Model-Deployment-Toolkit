"""Vercel serverless function for model info."""

import os
import json


def handler(request):
    """
    Model information endpoint.

    Args:
        request: Vercel request object

    Returns:
        Model information response
    """
    model_info = {
        'model_version': os.getenv('MODEL_VERSION', '1.0.0'),
        'model_type': os.getenv('MODEL_TYPE', 'sklearn'),
        'platform': 'vercel',
        'runtime': 'python3.9'
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(model_info)
    }

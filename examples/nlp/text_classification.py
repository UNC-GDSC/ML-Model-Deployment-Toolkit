"""
Text Classification Example using Transformers

This example demonstrates how to deploy a text classification model
for sentiment analysis and topic classification.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.base_model import BaseModel


class TextClassificationModel(BaseModel):
    """
    Text classification model using pre-trained transformers.

    Supports sentiment analysis, topic classification, and more.
    """

    def __init__(
        self,
        model_path: str = "distilbert-base-uncased-finetuned-sst-2-english",
        model_version: str = "1.0.0",
        task: str = "sentiment-analysis"
    ):
        """
        Initialize text classification model.

        Args:
            model_path: HuggingFace model name or local path
            model_version: Model version
            task: Classification task (sentiment-analysis, topic-classification, etc.)
        """
        super().__init__(model_path, model_version)
        self.task = task
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load pre-trained transformer model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

        except ImportError:
            print("Warning: transformers library not installed")
            print("Using mock predictions for demonstration")
            self.model = None

    def preprocess(self, features):
        """
        Preprocess text for model input.

        Args:
            features: Text input (string or dict)

        Returns:
            Tokenized input tensors
        """
        # Extract text from input
        if isinstance(features, str):
            text = features
        elif isinstance(features, dict):
            text = features.get('text', features.get('input', ''))
        elif isinstance(features, list):
            text = features
        else:
            text = str(features)

        if self.model is None:
            return text

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def predict(self, features):
        """
        Classify text.

        Args:
            features: Preprocessed inputs

        Returns:
            Classification logits
        """
        if self.model is None:
            # Mock predictions
            return self._mock_predictions(features)

        import torch

        with torch.no_grad():
            outputs = self.model(**features)
            logits = outputs.logits

        return logits

    def postprocess(self, predictions):
        """
        Postprocess model outputs.

        Args:
            predictions: Raw logits

        Returns:
            Classification results with probabilities
        """
        if isinstance(predictions, dict):
            # Mock predictions
            return predictions

        import torch

        # Apply softmax
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get label mappings
        label_map = self.model.config.id2label

        # Create results
        results = {
            'label': label_map[np.argmax(probs)],
            'confidence': float(np.max(probs)),
            'all_scores': [
                {
                    'label': label_map[i],
                    'score': float(prob)
                }
                for i, prob in enumerate(probs)
            ]
        }

        return results

    def _mock_predictions(self, text):
        """Generate mock predictions for demonstration."""
        if 'good' in text.lower() or 'great' in text.lower() or 'excellent' in text.lower():
            sentiment = 'POSITIVE'
            score = 0.95
        elif 'bad' in text.lower() or 'terrible' in text.lower() or 'awful' in text.lower():
            sentiment = 'NEGATIVE'
            score = 0.92
        else:
            sentiment = 'NEUTRAL'
            score = 0.60

        return {
            'label': sentiment,
            'confidence': score,
            'all_scores': [
                {'label': 'POSITIVE', 'score': score if sentiment == 'POSITIVE' else 1 - score},
                {'label': 'NEGATIVE', 'score': score if sentiment == 'NEGATIVE' else 1 - score}
            ]
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of text strings

        Returns:
            List of classification results
        """
        results = []

        for text in texts:
            result = self.predict_with_preprocessing(text)
            results.append(result['prediction'])

        return results


def example_usage():
    """Example usage of text classification model."""

    print("Text Classification Example")
    print("=" * 50)
    print()

    # Initialize model
    model = TextClassificationModel()
    model.load()

    # Example texts
    texts = [
        "This product is amazing! I love it!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
        "Best purchase I've ever made!",
        "Complete waste of money."
    ]

    print("Sentiment Analysis Results:")
    print("-" * 50)

    for text in texts:
        result = model.predict_with_preprocessing(text)
        pred = result['prediction']

        print(f"Text: {text}")
        print(f"Sentiment: {pred['label']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print()

    # Batch processing
    print("Batch Processing:")
    print("-" * 50)

    batch_results = model.classify_batch(texts)

    print(f"Processed {len(batch_results)} texts")
    positive_count = sum(1 for r in batch_results if r['label'] == 'POSITIVE')
    negative_count = sum(1 for r in batch_results if r['label'] == 'NEGATIVE')

    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")
    print()

    # API Integration Example
    print("API Integration:")
    print("-" * 50)
    print("Example request to deployed endpoint:")
    print()
    print("POST /predict")
    print("Content-Type: application/json")
    print()
    print('''{
    "text": "This product is amazing!"
}''')
    print()
    print("Response:")
    print()
    print('''{
    "prediction": {
        "label": "POSITIVE",
        "confidence": 0.95,
        "all_scores": [...]
    },
    "latency_ms": 42.3
}''')
    print()

    # Deployment instructions
    print("Deployment Instructions:")
    print("-" * 50)
    print("1. For GCP Cloud Run:")
    print("   cd templates/gcp-cloud-run")
    print("   gcloud run deploy text-classifier --source .")
    print()
    print("2. For Vercel:")
    print("   cd templates/vercel")
    print("   vercel deploy")
    print()
    print("3. For AWS SageMaker:")
    print("   cd templates/aws-sagemaker")
    print("   ./deploy.sh text-classifier prod us-east-1 ml.m5.large")
    print()


if __name__ == '__main__':
    example_usage()

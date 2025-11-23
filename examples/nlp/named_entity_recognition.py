"""
Named Entity Recognition (NER) Example

This example demonstrates how to deploy an NER model to extract
entities like persons, organizations, and locations from text.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.base_model import BaseModel


class NERModel(BaseModel):
    """
    Named Entity Recognition model using transformers.

    Extracts entities such as:
    - PER (Person)
    - ORG (Organization)
    - LOC (Location)
    - MISC (Miscellaneous)
    """

    def __init__(
        self,
        model_path: str = "dslim/bert-base-NER",
        model_version: str = "1.0.0"
    ):
        """
        Initialize NER model.

        Args:
            model_path: HuggingFace model name or local path
            model_version: Model version
        """
        super().__init__(model_path, model_version)
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def load(self) -> None:
        """Load pre-trained NER model."""
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "ner",
                model=self.model_path,
                aggregation_strategy="simple"
            )

        except ImportError:
            print("Warning: transformers library not installed")
            print("Using mock NER for demonstration")
            self.pipeline = None

    def preprocess(self, features):
        """
        Preprocess text for NER.

        Args:
            features: Text input

        Returns:
            Text string
        """
        if isinstance(features, str):
            return features
        elif isinstance(features, dict):
            return features.get('text', features.get('input', ''))
        else:
            return str(features)

    def predict(self, features):
        """
        Extract named entities from text.

        Args:
            features: Preprocessed text

        Returns:
            Entity predictions
        """
        if self.pipeline is None:
            return self._mock_predictions(features)

        results = self.pipeline(features)
        return results

    def postprocess(self, predictions):
        """
        Postprocess NER results.

        Args:
            predictions: Raw entity predictions

        Returns:
            Formatted entity list
        """
        entities = []

        for entity in predictions:
            entities.append({
                'text': entity.get('word', entity.get('entity_group', '')),
                'label': entity.get('entity_group', entity.get('entity', '')),
                'score': float(entity.get('score', 0.0)),
                'start': entity.get('start', 0),
                'end': entity.get('end', 0)
            })

        return entities

    def _mock_predictions(self, text):
        """Generate mock NER predictions."""
        entities = []

        # Simple keyword matching for demonstration
        keywords = {
            'Apple': 'ORG',
            'Microsoft': 'ORG',
            'Google': 'ORG',
            'New York': 'LOC',
            'London': 'LOC',
            'John': 'PER',
            'Mary': 'PER'
        }

        for keyword, label in keywords.items():
            if keyword in text:
                start = text.find(keyword)
                entities.append({
                    'word': keyword,
                    'entity_group': label,
                    'score': 0.95,
                    'start': start,
                    'end': start + len(keyword)
                })

        return entities

    def extract_entities_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type.

        Args:
            text: Input text

        Returns:
            Dictionary of entity type -> list of entities
        """
        result = self.predict_with_preprocessing(text)
        entities = result['prediction']

        grouped = {}
        for entity in entities:
            label = entity['label']
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(entity['text'])

        return grouped


def example_usage():
    """Example usage of NER model."""

    print("Named Entity Recognition Example")
    print("=" * 50)
    print()

    # Initialize model
    model = NERModel()
    model.load()

    # Example texts
    texts = [
        "Apple Inc. is planning to open a new store in New York next month.",
        "John Smith works at Microsoft in London.",
        "The meeting between Google and Amazon executives will take place in San Francisco.",
    ]

    for i, text in enumerate(texts, 1):
        print(f"Example {i}:")
        print(f"Text: {text}")
        print()

        # Extract entities
        result = model.predict_with_preprocessing(text)
        entities = result['prediction']

        if entities:
            print("Entities found:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']}) - confidence: {entity['score']:.4f}")
        else:
            print("No entities found")

        print(f"Latency: {result['latency_ms']:.2f}ms")
        print()
        print("-" * 50)
        print()

    # Grouped entities
    print("Grouped Entities Example:")
    print("-" * 50)

    text = "John Smith from Apple met with Mary Johnson from Microsoft in London."
    grouped = model.extract_entities_by_type(text)

    print(f"Text: {text}")
    print()

    for entity_type, entities in grouped.items():
        print(f"{entity_type}: {', '.join(entities)}")

    print()

    # Real-time API usage
    print("Real-time API Usage:")
    print("-" * 50)
    print('''
# Python client example
import requests

response = requests.post(
    "https://your-endpoint.com/predict",
    json={"text": "Apple Inc. is based in Cupertino."}
)

entities = response.json()['prediction']
for entity in entities:
    print(f"{entity['text']}: {entity['label']}")
''')
    print()

    # Use cases
    print("Common Use Cases:")
    print("-" * 50)
    print("1. Information Extraction: Extract structured data from documents")
    print("2. Content Organization: Categorize and tag articles automatically")
    print("3. Privacy Compliance: Identify and redact PII (Personal Identifiable Information)")
    print("4. Knowledge Graphs: Build entity relationships from text")
    print("5. Search Enhancement: Improve search with entity-based indexing")
    print()

    # Deployment
    print("Deployment Options:")
    print("-" * 50)
    print("1. High-throughput (Kubernetes):")
    print("   - Deploy with HPA for auto-scaling")
    print("   - Use GPU instances for better performance")
    print()
    print("2. Low-latency (AWS SageMaker):")
    print("   - Use ml.g4dn.xlarge for GPU acceleration")
    print("   - Enable auto-scaling based on invocations")
    print()
    print("3. Cost-optimized (Azure Functions):")
    print("   - Serverless for intermittent workloads")
    print("   - Pay only for actual usage")
    print()


if __name__ == '__main__':
    example_usage()

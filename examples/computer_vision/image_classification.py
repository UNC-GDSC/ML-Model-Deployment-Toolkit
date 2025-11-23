"""
Image Classification Example using ResNet50

This example demonstrates how to deploy an image classification model
using the ML Model Deployment Toolkit.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.base_model import BaseModel


class ImageClassificationModel(BaseModel):
    """
    Image classification model using pre-trained ResNet50.

    This model can classify images into 1000 ImageNet categories.
    """

    def __init__(self, model_path: str = None, model_version: str = "1.0.0"):
        """
        Initialize image classification model.

        Args:
            model_path: Path to saved model (if fine-tuned)
            model_version: Model version
        """
        super().__init__(model_path or "resnet50", model_version)
        self.model = None
        self.preprocess_fn = None

    def load(self) -> None:
        """Load pre-trained ResNet50 model."""
        try:
            # Try TensorFlow/Keras first
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input

            self.model = ResNet50(weights='imagenet')
            self.preprocess_fn = preprocess_input
            self.framework = 'tensorflow'

        except ImportError:
            # Fall back to PyTorch
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self.model = models.resnet50(pretrained=True)
            self.model.eval()

            self.preprocess_fn = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.framework = 'pytorch'

    def preprocess(self, features):
        """
        Preprocess image for model input.

        Args:
            features: Image data (file path, bytes, or array)

        Returns:
            Preprocessed image tensor
        """
        from PIL import Image
        import io

        # Handle different input types
        if isinstance(features, str):
            # File path
            img = Image.open(features)
        elif isinstance(features, bytes):
            # Raw bytes
            img = Image.open(io.BytesIO(features))
        elif isinstance(features, dict) and 'image' in features:
            # Dictionary with image key
            if isinstance(features['image'], str):
                img = Image.open(features['image'])
            elif isinstance(features['image'], bytes):
                img = Image.open(io.BytesIO(features['image']))
            else:
                img = features['image']
        else:
            # Assume PIL Image
            img = features

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.framework == 'tensorflow':
            # Resize and convert to array
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return self.preprocess_fn(img_array)
        else:
            # PyTorch preprocessing
            return self.preprocess_fn(img).unsqueeze(0)

    def predict(self, features) -> np.ndarray:
        """
        Make prediction on preprocessed image.

        Args:
            features: Preprocessed image tensor

        Returns:
            Prediction probabilities
        """
        if self.framework == 'tensorflow':
            predictions = self.model.predict(features)
        else:
            import torch
            with torch.no_grad():
                predictions = self.model(features)
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                predictions = predictions.numpy()

        return predictions

    def postprocess(self, predictions):
        """
        Postprocess predictions to get top-k classes.

        Args:
            predictions: Raw model predictions

        Returns:
            Top-5 predicted classes with probabilities
        """
        # Get top 5 predictions
        top_k = 5
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_probs = predictions[0][top_indices]

        # Load class names
        class_names = self._load_imagenet_classes()

        results = [
            {
                'class_id': int(idx),
                'class_name': class_names.get(idx, f'class_{idx}'),
                'probability': float(prob)
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        return results

    def _load_imagenet_classes(self):
        """Load ImageNet class names."""
        # Simplified class names (in production, load from file)
        return {
            i: f"imagenet_class_{i}"
            for i in range(1000)
        }


def example_usage():
    """Example usage of image classification model."""

    # Initialize model
    model = ImageClassificationModel()
    model.load()

    # Example 1: Classify from file path
    print("Example 1: Classify image from file")
    print("-" * 50)

    # Create a sample image (in production, use real image)
    from PIL import Image
    sample_img = Image.new('RGB', (224, 224), color='red')
    sample_img.save('/tmp/sample_image.jpg')

    result = model.predict_with_preprocessing('/tmp/sample_image.jpg')
    print(f"Top predictions:")
    for pred in result['prediction'][:3]:
        print(f"  {pred['class_name']}: {pred['probability']:.4f}")
    print(f"Inference time: {result['latency_ms']:.2f}ms")
    print()

    # Example 2: Classify from bytes
    print("Example 2: Classify image from bytes")
    print("-" * 50)

    with open('/tmp/sample_image.jpg', 'rb') as f:
        img_bytes = f.read()

    result = model.predict_with_preprocessing(img_bytes)
    print(f"Prediction: {result['prediction'][0]['class_name']}")
    print(f"Confidence: {result['prediction'][0]['probability']:.4f}")
    print()

    # Example 3: Deploy to AWS Lambda
    print("Example 3: Deploy to AWS Lambda")
    print("-" * 50)
    print("To deploy this model to AWS Lambda:")
    print("1. Save the model:")
    print("   model.save('models/image_classifier.h5')")
    print("2. Use the deployment script:")
    print("   ml-deploy aws-lambda --model-path models/image_classifier.h5 --name image-classifier")
    print()

    # Cleanup
    os.remove('/tmp/sample_image.jpg')


if __name__ == '__main__':
    example_usage()

"""
Object Detection Example using YOLO

This example demonstrates how to deploy an object detection model
for real-time inference.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.base_model import BaseModel


class ObjectDetectionModel(BaseModel):
    """
    Object detection model using YOLO architecture.

    Detects multiple objects in images with bounding boxes.
    """

    def __init__(self, model_path: str = "yolov5s", model_version: str = "1.0.0"):
        """
        Initialize object detection model.

        Args:
            model_path: Path to model or model name
            model_version: Model version
        """
        super().__init__(model_path, model_version)
        self.model = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def load(self) -> None:
        """Load YOLO model."""
        try:
            # Try YOLOv5 from torch hub
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold

        except Exception as e:
            print(f"Warning: Could not load YOLOv5: {e}")
            print("Using mock detection for demonstration")
            self.model = None

    def preprocess(self, features):
        """
        Preprocess image for object detection.

        Args:
            features: Image data

        Returns:
            Preprocessed image
        """
        from PIL import Image
        import io

        # Handle different input types
        if isinstance(features, str):
            img = Image.open(features)
        elif isinstance(features, bytes):
            img = Image.open(io.BytesIO(features))
        elif isinstance(features, dict) and 'image' in features:
            if isinstance(features['image'], str):
                img = Image.open(features['image'])
            else:
                img = features['image']
        else:
            img = features

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def predict(self, features):
        """
        Detect objects in image.

        Args:
            features: Preprocessed image

        Returns:
            Detection results
        """
        if self.model is None:
            # Mock predictions for demonstration
            return self._mock_predictions()

        # Run inference
        results = self.model(features)

        return results

    def postprocess(self, predictions):
        """
        Postprocess detection results.

        Args:
            predictions: Raw model predictions

        Returns:
            List of detected objects with bounding boxes
        """
        if isinstance(predictions, list):
            # Mock predictions
            return predictions

        # Parse YOLO results
        detections = []

        # Get pandas DataFrame from results
        results_df = predictions.pandas().xyxy[0]

        for _, row in results_df.iterrows():
            detection = {
                'class_name': row['name'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'xmin': float(row['xmin']),
                    'ymin': float(row['ymin']),
                    'xmax': float(row['xmax']),
                    'ymax': float(row['ymax'])
                }
            }
            detections.append(detection)

        return detections

    def _mock_predictions(self):
        """Generate mock predictions for demonstration."""
        return [
            {
                'class_name': 'person',
                'confidence': 0.92,
                'bbox': {'xmin': 100, 'ymin': 50, 'xmax': 300, 'ymax': 400}
            },
            {
                'class_name': 'car',
                'confidence': 0.85,
                'bbox': {'xmin': 400, 'ymin': 200, 'xmax': 600, 'ymax': 450}
            }
        ]

    def draw_detections(self, image_path: str, detections: List[Dict], output_path: str):
        """
        Draw bounding boxes on image.

        Args:
            image_path: Path to input image
            detections: List of detections
            output_path: Path to save annotated image
        """
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            # Draw rectangle
            draw.rectangle(
                [(bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax'])],
                outline='red',
                width=3
            )

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((bbox['xmin'], bbox['ymin'] - 10), label, fill='red')

        img.save(output_path)


def example_usage():
    """Example usage of object detection model."""
    from PIL import Image
    import os

    print("Object Detection Example")
    print("=" * 50)
    print()

    # Initialize model
    model = ObjectDetectionModel()
    model.load()

    # Create sample image
    sample_img = Image.new('RGB', (640, 480), color='blue')
    sample_img.save('/tmp/detect_sample.jpg')

    # Run detection
    print("Running object detection...")
    result = model.predict_with_preprocessing('/tmp/detect_sample.jpg')

    print(f"\nDetected {len(result['prediction'])} objects:")
    print("-" * 50)

    for i, det in enumerate(result['prediction'], 1):
        print(f"{i}. {det['class_name']}")
        print(f"   Confidence: {det['confidence']:.2f}")
        print(f"   Bounding box: {det['bbox']}")
        print()

    print(f"Inference time: {result['latency_ms']:.2f}ms")
    print()

    # Draw detections
    print("Saving annotated image...")
    model.draw_detections(
        '/tmp/detect_sample.jpg',
        result['prediction'],
        '/tmp/detect_output.jpg'
    )
    print("Saved to: /tmp/detect_output.jpg")
    print()

    # Deployment example
    print("Deployment Instructions:")
    print("-" * 50)
    print("1. For AWS Lambda (with container):")
    print("   cd templates/aws-lambda")
    print("   docker build -t object-detector .")
    print("   # Push to ECR and deploy")
    print()
    print("2. For Kubernetes:")
    print("   cd templates/kubernetes")
    print("   kubectl apply -f deployment.yaml")
    print()
    print("3. For AWS SageMaker:")
    print("   cd templates/aws-sagemaker")
    print("   ./deploy.sh object-detector prod")
    print()

    # Cleanup
    os.remove('/tmp/detect_sample.jpg')
    if os.path.exists('/tmp/detect_output.jpg'):
        os.remove('/tmp/detect_output.jpg')


if __name__ == '__main__':
    example_usage()

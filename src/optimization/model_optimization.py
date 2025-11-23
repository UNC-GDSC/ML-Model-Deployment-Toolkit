"""Model optimization utilities for production deployment."""

import logging
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Quantize models to reduce size and improve inference speed.

    Supports different quantization methods for various frameworks.
    """

    def __init__(self, model: Any, model_type: str):
        """
        Initialize model quantizer.

        Args:
            model: Model to quantize
            model_type: Type of model (tensorflow, pytorch, onnx)
        """
        self.model = model
        self.model_type = model_type

    def quantize_tensorflow(
        self,
        output_path: str,
        optimization: str = "DEFAULT"
    ) -> str:
        """
        Quantize TensorFlow model.

        Args:
            output_path: Path to save quantized model
            optimization: Optimization mode (DEFAULT, OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY)

        Returns:
            Path to quantized model
        """
        try:
            import tensorflow as tf

            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Set optimization
            if optimization == "OPTIMIZE_FOR_SIZE":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif optimization == "OPTIMIZE_FOR_LATENCY":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert
            tflite_model = converter.convert()

            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"Quantized TensorFlow model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TensorFlow quantization failed: {e}")
            raise

    def quantize_pytorch(
        self,
        output_path: str,
        quantization_type: str = "dynamic"
    ) -> str:
        """
        Quantize PyTorch model.

        Args:
            output_path: Path to save quantized model
            quantization_type: Type of quantization (dynamic, static, qat)

        Returns:
            Path to quantized model
        """
        try:
            import torch
            from torch.quantization import quantize_dynamic, get_default_qconfig

            if quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            elif quantization_type == "static":
                # Static quantization requires calibration
                self.model.qconfig = get_default_qconfig('fbgemm')
                torch.quantization.prepare(self.model, inplace=True)
                # Need to run calibration here with representative data
                quantized_model = torch.quantization.convert(self.model, inplace=True)
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")

            # Save
            torch.save(quantized_model.state_dict(), output_path)

            logger.info(f"Quantized PyTorch model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"PyTorch quantization failed: {e}")
            raise

    def quantize_onnx(
        self,
        model_path: str,
        output_path: str,
        quantization_mode: str = "IntegerOps"
    ) -> str:
        """
        Quantize ONNX model.

        Args:
            model_path: Path to ONNX model
            output_path: Path to save quantized model
            quantization_mode: Quantization mode

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_path,
                output_path,
                weight_type=QuantType.QUInt8
            )

            logger.info(f"Quantized ONNX model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            raise


class ModelPruner:
    """
    Prune models to reduce size and computation.
    """

    def __init__(self, model: Any, model_type: str):
        """
        Initialize model pruner.

        Args:
            model: Model to prune
            model_type: Type of model
        """
        self.model = model
        self.model_type = model_type

    def prune_tensorflow(
        self,
        target_sparsity: float = 0.5,
        output_path: str = None
    ):
        """
        Prune TensorFlow model.

        Args:
            target_sparsity: Target sparsity (0.0 to 1.0)
            output_path: Path to save pruned model

        Returns:
            Pruned model
        """
        try:
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot

            # Define pruning schedule
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }

            # Apply pruning
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                self.model,
                **pruning_params
            )

            if output_path:
                model_for_pruning.save(output_path)
                logger.info(f"Pruned model saved to {output_path}")

            return model_for_pruning

        except Exception as e:
            logger.error(f"TensorFlow pruning failed: {e}")
            raise

    def prune_pytorch(
        self,
        amount: float = 0.5,
        output_path: str = None
    ):
        """
        Prune PyTorch model.

        Args:
            amount: Amount to prune (0.0 to 1.0)
            output_path: Path to save pruned model

        Returns:
            Pruned model
        """
        try:
            import torch
            import torch.nn.utils.prune as prune

            # Prune all linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')

            if output_path:
                torch.save(self.model.state_dict(), output_path)
                logger.info(f"Pruned model saved to {output_path}")

            return self.model

        except Exception as e:
            logger.error(f"PyTorch pruning failed: {e}")
            raise


class ModelConverter:
    """
    Convert models between different formats for optimization.
    """

    @staticmethod
    def sklearn_to_onnx(
        sklearn_model,
        initial_types,
        output_path: str
    ) -> str:
        """
        Convert sklearn model to ONNX.

        Args:
            sklearn_model: Scikit-learn model
            initial_types: Initial types for ONNX
            output_path: Path to save ONNX model

        Returns:
            Path to ONNX model
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Convert
            onnx_model = convert_sklearn(sklearn_model, initial_types=initial_types)

            # Save
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"Converted sklearn model to ONNX: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"sklearn to ONNX conversion failed: {e}")
            raise

    @staticmethod
    def pytorch_to_onnx(
        pytorch_model,
        dummy_input,
        output_path: str,
        input_names: list = None,
        output_names: list = None
    ) -> str:
        """
        Convert PyTorch model to ONNX.

        Args:
            pytorch_model: PyTorch model
            dummy_input: Example input tensor
            output_path: Path to save ONNX model
            input_names: Names of inputs
            output_names: Names of outputs

        Returns:
            Path to ONNX model
        """
        try:
            import torch

            # Export
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                input_names=input_names or ['input'],
                output_names=output_names or ['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            logger.info(f"Converted PyTorch model to ONNX: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"PyTorch to ONNX conversion failed: {e}")
            raise

    @staticmethod
    def tensorflow_to_onnx(
        tf_model,
        output_path: str
    ) -> str:
        """
        Convert TensorFlow model to ONNX.

        Args:
            tf_model: TensorFlow model
            output_path: Path to save ONNX model

        Returns:
            Path to ONNX model
        """
        try:
            import tf2onnx
            import tensorflow as tf

            # Convert
            spec = (tf.TensorSpec(tf_model.input_shape, tf.float32, name="input"),)
            output_path_with_ext = output_path if output_path.endswith('.onnx') else f"{output_path}.onnx"

            model_proto, _ = tf2onnx.convert.from_keras(
                tf_model,
                input_signature=spec,
                output_path=output_path_with_ext
            )

            logger.info(f"Converted TensorFlow model to ONNX: {output_path_with_ext}")
            return output_path_with_ext

        except Exception as e:
            logger.error(f"TensorFlow to ONNX conversion failed: {e}")
            raise


def estimate_model_size(model_path: str) -> dict:
    """
    Estimate model size and memory requirements.

    Args:
        model_path: Path to model file

    Returns:
        Dictionary with size estimates
    """
    import os

    file_size = os.path.getsize(model_path)

    return {
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'file_size_gb': file_size / (1024 * 1024 * 1024),
        'estimated_memory_mb': file_size / (1024 * 1024) * 1.5,  # Rough estimate
        'recommendations': {
            'vercel_compatible': file_size < 50 * 1024 * 1024,  # 50MB
            'aws_lambda_compatible': file_size < 250 * 1024 * 1024,  # 250MB
            'optimization_suggested': file_size > 100 * 1024 * 1024  # 100MB
        }
    }

"""Batch prediction utilities."""

import logging
from typing import List, Any, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Efficient batch prediction handler.

    Processes multiple predictions in batches for improved throughput.
    """

    def __init__(
        self,
        model: BaseModel,
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """
        Initialize batch predictor.

        Args:
            model: Model instance
            batch_size: Size of each batch
            max_workers: Maximum number of worker threads
        """
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers

    def predict_batch(
        self,
        features_list: List[Any],
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of inputs.

        Args:
            features_list: List of feature arrays
            return_probabilities: Whether to return probabilities

        Returns:
            List of prediction results
        """
        if not features_list:
            return []

        # Process in batches
        results = []

        for i in range(0, len(features_list), self.batch_size):
            batch = features_list[i:i + self.batch_size]

            # Convert to numpy array
            batch_array = np.array(batch)

            # Make prediction
            result = self.model.predict_with_preprocessing(
                batch_array,
                return_probabilities
            )

            # Split result back into individual predictions
            predictions = result['prediction']
            if not isinstance(predictions, list):
                predictions = [predictions]

            probabilities = result.get('probabilities', [None] * len(predictions))
            if not isinstance(probabilities, list):
                probabilities = [probabilities]

            for j, (pred, prob) in enumerate(zip(predictions, probabilities)):
                item_result = {
                    'prediction': pred,
                    'model_version': result['model_version'],
                    'timestamp': result['timestamp']
                }

                if prob is not None:
                    item_result['probabilities'] = prob

                results.append(item_result)

        return results

    def predict_parallel(
        self,
        features_list: List[Any],
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions using parallel processing.

        Args:
            features_list: List of feature arrays
            return_probabilities: Whether to return probabilities

        Returns:
            List of prediction results
        """
        if not features_list:
            return []

        # Split into chunks
        chunks = [
            features_list[i:i + self.batch_size]
            for i in range(0, len(features_list), self.batch_size)
        ]

        results = []

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_chunk,
                    chunk,
                    return_probabilities
                ): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results in order
            chunk_results = [None] * len(chunks)

            for future in as_completed(futures):
                chunk_index = futures[future]
                try:
                    chunk_results[chunk_index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    chunk_results[chunk_index] = []

            # Flatten results
            for chunk_result in chunk_results:
                if chunk_result:
                    results.extend(chunk_result)

        return results

    def _process_chunk(
        self,
        chunk: List[Any],
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """
        Process a single chunk.

        Args:
            chunk: Chunk of feature arrays
            return_probabilities: Whether to return probabilities

        Returns:
            List of prediction results
        """
        return self.predict_batch(chunk, return_probabilities)

    def get_optimal_batch_size(
        self,
        sample_features: Any,
        target_latency_ms: float = 100,
        max_batch_size: int = 128
    ) -> int:
        """
        Determine optimal batch size for target latency.

        Args:
            sample_features: Sample feature array
            target_latency_ms: Target latency in milliseconds
            max_batch_size: Maximum batch size to test

        Returns:
            Optimal batch size
        """
        import time

        best_batch_size = 1
        best_throughput = 0

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > max_batch_size:
                break

            # Create batch
            batch = [sample_features] * batch_size

            # Time prediction
            start = time.time()
            self.predict_batch(batch)
            elapsed = (time.time() - start) * 1000  # ms

            # Calculate throughput
            throughput = batch_size / elapsed

            logger.info(
                f"Batch size {batch_size}: "
                f"{elapsed:.2f}ms total, "
                f"{elapsed/batch_size:.2f}ms per item, "
                f"{throughput:.2f} items/ms"
            )

            # Check if within target latency
            if elapsed / batch_size <= target_latency_ms:
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
            else:
                break

        logger.info(f"Optimal batch size: {best_batch_size}")
        return best_batch_size

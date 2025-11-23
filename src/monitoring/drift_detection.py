"""Data drift detection for deployed models."""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect data drift in production using statistical tests.

    Monitors for changes in input feature distributions.
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference dataset (training data)
            feature_names: Names of features
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(reference_data.shape[1])
        ]
        self.threshold = threshold

        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)

        # Drift history
        self.drift_history: List[Dict[str, Any]] = []

    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each feature.

        Args:
            data: Input data

        Returns:
            Dictionary of feature statistics
        """
        stats_dict = {}

        for i, name in enumerate(self.feature_names):
            feature_data = data[:, i]

            stats_dict[name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q75': float(np.percentile(feature_data, 75))
            }

        return stats_dict

    def detect_drift(
        self,
        current_data: np.ndarray,
        method: str = "ks"
    ) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current production data
            method: Statistical test method ('ks' or 'chi2')

        Returns:
            Dictionary containing drift detection results
        """
        drift_detected = {}
        p_values = {}

        for i, name in enumerate(self.feature_names):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]

            if method == "ks":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
            elif method == "chi2":
                # Chi-square test (for categorical features)
                # Bin the data
                ref_hist, bins = np.histogram(ref_feature, bins=10)
                curr_hist, _ = np.histogram(curr_feature, bins=bins)

                statistic, p_value = stats.chisquare(curr_hist, ref_hist)
            else:
                raise ValueError(f"Unknown method: {method}")

            p_values[name] = float(p_value)
            drift_detected[name] = p_value < self.threshold

        # Calculate current statistics
        current_stats = self._calculate_statistics(current_data)

        # Overall drift status
        any_drift = any(drift_detected.values())
        num_drifted = sum(drift_detected.values())

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': any_drift,
            'num_features_drifted': num_drifted,
            'total_features': len(self.feature_names),
            'drift_percentage': (num_drifted / len(self.feature_names)) * 100,
            'p_values': p_values,
            'drift_by_feature': drift_detected,
            'reference_stats': self.reference_stats,
            'current_stats': current_stats,
            'method': method
        }

        # Add to history
        self.drift_history.append(result)

        if any_drift:
            drifted_features = [name for name, drifted in drift_detected.items() if drifted]
            logger.warning(
                f"Data drift detected in {num_drifted} features: {drifted_features}"
            )
        else:
            logger.info("No data drift detected")

        return result

    def get_drift_report(self) -> Dict[str, Any]:
        """
        Get comprehensive drift report.

        Returns:
            Dictionary containing drift analysis
        """
        if not self.drift_history:
            return {'message': 'No drift checks performed yet'}

        latest = self.drift_history[-1]
        total_checks = len(self.drift_history)
        drift_count = sum(1 for h in self.drift_history if h['drift_detected'])

        # Feature-level drift frequency
        feature_drift_count = {name: 0 for name in self.feature_names}
        for history in self.drift_history:
            for name, drifted in history['drift_by_feature'].items():
                if drifted:
                    feature_drift_count[name] += 1

        return {
            'total_checks': total_checks,
            'drift_count': drift_count,
            'drift_rate': (drift_count / total_checks) * 100,
            'latest_check': latest,
            'feature_drift_frequency': {
                name: (count / total_checks) * 100
                for name, count in feature_drift_count.items()
            },
            'most_drifted_features': sorted(
                feature_drift_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def calculate_psi(
        self,
        current_data: np.ndarray,
        bins: int = 10
    ) -> Dict[str, float]:
        """
        Calculate Population Stability Index (PSI) for each feature.

        PSI is a measure of how much a distribution has shifted.
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change

        Args:
            current_data: Current production data
            bins: Number of bins for discretization

        Returns:
            Dictionary of PSI values per feature
        """
        psi_values = {}

        for i, name in enumerate(self.feature_names):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]

            # Create bins based on reference data
            _, bin_edges = np.histogram(ref_feature, bins=bins)

            # Calculate distributions
            ref_dist, _ = np.histogram(ref_feature, bins=bin_edges)
            curr_dist, _ = np.histogram(curr_feature, bins=bin_edges)

            # Normalize to get proportions
            ref_prop = ref_dist / len(ref_feature)
            curr_prop = curr_dist / len(curr_feature)

            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            psi_values[name] = float(psi)

        return psi_values

    def monitor_prediction_drift(
        self,
        predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Monitor drift in model predictions.

        Args:
            predictions: Current predictions
            reference_predictions: Reference predictions (optional)

        Returns:
            Prediction drift analysis
        """
        if reference_predictions is None:
            # Use reference data to generate predictions
            logger.warning("No reference predictions provided")
            return {}

        # Statistical test
        statistic, p_value = stats.ks_2samp(reference_predictions, predictions)

        drift_detected = p_value < self.threshold

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_drift_detected': drift_detected,
            'p_value': float(p_value),
            'statistic': float(statistic),
            'reference_mean': float(np.mean(reference_predictions)),
            'current_mean': float(np.mean(predictions)),
            'reference_std': float(np.std(reference_predictions)),
            'current_std': float(np.std(predictions))
        }

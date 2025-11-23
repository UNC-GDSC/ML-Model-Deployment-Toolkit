"""A/B testing framework for model deployment."""

import logging
import random
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

from src.core.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    variants: Dict[str, float]  # variant_name: traffic_percentage
    created_at: str
    status: str = "active"  # active, paused, completed
    metrics: Dict[str, Dict[str, Any]] = None  # variant -> metrics


class ABTestManager:
    """
    A/B testing manager for model deployment.

    Enables traffic splitting between multiple model versions.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        config_path: str = ".ab_tests.json"
    ):
        """
        Initialize A/B test manager.

        Args:
            models: Dictionary of model_name -> BaseModel instances
            config_path: Path to configuration file
        """
        self.models = models
        self.config_path = Path(config_path)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.load_experiments()

        # Statistics
        self.requests_per_variant: Dict[str, int] = {
            name: 0 for name in models.keys()
        }
        self.errors_per_variant: Dict[str, int] = {
            name: 0 for name in models.keys()
        }

    def load_experiments(self) -> None:
        """Load experiments from disk."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)

                self.experiments = {
                    name: ExperimentConfig(**config)
                    for name, config in data.items()
                }

                logger.info(f"Loaded {len(self.experiments)} experiments")

            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")
                self.experiments = {}

    def save_experiments(self) -> None:
        """Save experiments to disk."""
        try:
            data = {
                name: asdict(config)
                for name, config in self.experiments.items()
            }

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.experiments)} experiments")

        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")

    def create_experiment(
        self,
        name: str,
        variants: Dict[str, float],
        description: str = ""
    ) -> ExperimentConfig:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            variants: Dictionary of variant_name -> traffic_percentage
            description: Experiment description

        Returns:
            ExperimentConfig object

        Raises:
            ValueError: If variants don't sum to 100 or reference unknown models
        """
        # Validate traffic percentages
        total_traffic = sum(variants.values())
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(
                f"Traffic percentages must sum to 100, got {total_traffic}"
            )

        # Validate model names
        unknown_models = set(variants.keys()) - set(self.models.keys())
        if unknown_models:
            raise ValueError(f"Unknown models: {unknown_models}")

        experiment = ExperimentConfig(
            name=name,
            description=description,
            variants=variants,
            created_at=datetime.utcnow().isoformat(),
            metrics={}
        )

        self.experiments[name] = experiment
        self.save_experiments()

        logger.info(f"Created experiment '{name}' with variants: {variants}")

        return experiment

    def select_variant(self, experiment_name: str) -> str:
        """
        Select a variant based on traffic allocation.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Selected variant name

        Raises:
            ValueError: If experiment not found or not active
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment = self.experiments[experiment_name]

        if experiment.status != "active":
            raise ValueError(f"Experiment '{experiment_name}' is not active")

        # Weighted random selection
        variants = list(experiment.variants.keys())
        weights = list(experiment.variants.values())

        variant = random.choices(variants, weights=weights, k=1)[0]

        logger.debug(f"Selected variant '{variant}' for experiment '{experiment_name}'")

        return variant

    def predict_with_ab_test(
        self,
        experiment_name: str,
        features: Any,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction using A/B test variant selection.

        Args:
            experiment_name: Name of the experiment
            features: Input features
            return_probabilities: Whether to return probabilities

        Returns:
            Prediction results with variant information
        """
        # Select variant
        variant = self.select_variant(experiment_name)

        # Get model
        model = self.models[variant]

        # Track request
        self.requests_per_variant[variant] += 1

        # Make prediction
        try:
            result = model.predict_with_preprocessing(
                features,
                return_probabilities
            )

            result['variant'] = variant
            result['experiment'] = experiment_name

            return result

        except Exception as e:
            self.errors_per_variant[variant] += 1
            logger.error(f"Prediction failed for variant '{variant}': {e}")
            raise

    def record_metric(
        self,
        experiment_name: str,
        variant: str,
        metric_name: str,
        value: float
    ) -> None:
        """
        Record a metric for a variant.

        Args:
            experiment_name: Name of the experiment
            variant: Variant name
            metric_name: Metric name
            value: Metric value
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment = self.experiments[experiment_name]

        if experiment.metrics is None:
            experiment.metrics = {}

        if variant not in experiment.metrics:
            experiment.metrics[variant] = {}

        if metric_name not in experiment.metrics[variant]:
            experiment.metrics[variant][metric_name] = {
                'values': [],
                'count': 0,
                'sum': 0.0,
                'mean': 0.0
            }

        metric = experiment.metrics[variant][metric_name]
        metric['values'].append(value)
        metric['count'] += 1
        metric['sum'] += value
        metric['mean'] = metric['sum'] / metric['count']

        self.save_experiments()

    def get_experiment_stats(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get statistics for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary of statistics
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment = self.experiments[experiment_name]

        stats = {
            'experiment': experiment_name,
            'status': experiment.status,
            'created_at': experiment.created_at,
            'variants': {}
        }

        for variant, traffic in experiment.variants.items():
            stats['variants'][variant] = {
                'traffic_allocation': traffic,
                'requests': self.requests_per_variant.get(variant, 0),
                'errors': self.errors_per_variant.get(variant, 0),
                'error_rate': (
                    self.errors_per_variant.get(variant, 0) /
                    max(self.requests_per_variant.get(variant, 1), 1)
                ),
                'metrics': experiment.metrics.get(variant, {}) if experiment.metrics else {}
            }

        return stats

    def stop_experiment(self, experiment_name: str) -> None:
        """
        Stop an experiment.

        Args:
            experiment_name: Name of the experiment
        """
        if experiment_name in self.experiments:
            self.experiments[experiment_name].status = "completed"
            self.save_experiments()
            logger.info(f"Stopped experiment '{experiment_name}'")
        else:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def pause_experiment(self, experiment_name: str) -> None:
        """
        Pause an experiment.

        Args:
            experiment_name: Name of the experiment
        """
        if experiment_name in self.experiments:
            self.experiments[experiment_name].status = "paused"
            self.save_experiments()
            logger.info(f"Paused experiment '{experiment_name}'")
        else:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def resume_experiment(self, experiment_name: str) -> None:
        """
        Resume a paused experiment.

        Args:
            experiment_name: Name of the experiment
        """
        if experiment_name in self.experiments:
            self.experiments[experiment_name].status = "active"
            self.save_experiments()
            logger.info(f"Resumed experiment '{experiment_name}'")
        else:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def get_winner(
        self,
        experiment_name: str,
        metric_name: str,
        higher_is_better: bool = True
    ) -> str:
        """
        Determine the winning variant based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to compare
            higher_is_better: Whether higher values are better

        Returns:
            Name of the winning variant
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment = self.experiments[experiment_name]

        if not experiment.metrics:
            raise ValueError(f"No metrics recorded for experiment '{experiment_name}'")

        # Get metric values for each variant
        variant_metrics = {}
        for variant in experiment.variants.keys():
            if variant in experiment.metrics:
                if metric_name in experiment.metrics[variant]:
                    variant_metrics[variant] = experiment.metrics[variant][metric_name]['mean']

        if not variant_metrics:
            raise ValueError(f"Metric '{metric_name}' not found for any variant")

        # Find winner
        if higher_is_better:
            winner = max(variant_metrics.items(), key=lambda x: x[1])
        else:
            winner = min(variant_metrics.items(), key=lambda x: x[1])

        logger.info(
            f"Winner for experiment '{experiment_name}': "
            f"{winner[0]} with {metric_name}={winner[1]:.4f}"
        )

        return winner[0]

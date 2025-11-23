"""Metrics collection and monitoring."""

import time
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collect and expose metrics for monitoring.

    Supports Prometheus metrics if available.
    """

    def __init__(self, enable_prometheus: bool = True):
        """
        Initialize metrics collector.

        Args:
            enable_prometheus: Whether to use Prometheus metrics
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Simple metrics storage
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

        # Prometheus metrics
        if self.enable_prometheus:
            self.prom_request_count = Counter(
                'ml_model_requests_total',
                'Total number of prediction requests'
            )
            self.prom_error_count = Counter(
                'ml_model_errors_total',
                'Total number of prediction errors'
            )
            self.prom_latency = Histogram(
                'ml_model_latency_seconds',
                'Prediction latency in seconds'
            )
            self.prom_model_loaded = Gauge(
                'ml_model_loaded',
                'Whether model is loaded (1) or not (0)'
            )

    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Increment value
        """
        self.counters[name] += value

        if self.enable_prometheus and name == 'requests':
            self.prom_request_count.inc(value)
        elif self.enable_prometheus and name == 'errors':
            self.prom_error_count.inc(value)

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value

        if self.enable_prometheus and name == 'model_loaded':
            self.prom_model_loaded.set(value)

    def observe_histogram(self, name: str, value: float) -> None:
        """
        Add an observation to a histogram.

        Args:
            name: Histogram name
            value: Observed value
        """
        self.histograms[name].append(value)

        if self.enable_prometheus and name == 'latency':
            self.prom_latency.observe(value)

    def record_request(self, latency_seconds: float, error: bool = False) -> None:
        """
        Record a prediction request.

        Args:
            latency_seconds: Request latency in seconds
            error: Whether the request resulted in an error
        """
        self.increment_counter('requests')
        self.observe_histogram('latency', latency_seconds)

        if error:
            self.increment_counter('errors')

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add histogram statistics
        for name, values in self.histograms.items():
            if values:
                metrics[f'{name}_avg'] = sum(values) / len(values)
                metrics[f'{name}_min'] = min(values)
                metrics[f'{name}_max'] = max(values)
                metrics[f'{name}_count'] = len(values)

        return metrics

    def export_prometheus(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus metrics as bytes
        """
        if not self.enable_prometheus:
            raise RuntimeError("Prometheus is not enabled")

        return generate_latest()

    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()


class LatencyTracker:
    """Context manager for tracking latency."""

    def __init__(self, metrics_collector: MetricsCollector, metric_name: str = 'latency'):
        """
        Initialize latency tracker.

        Args:
            metrics_collector: MetricsCollector instance
            metric_name: Name of the latency metric
        """
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record latency."""
        if self.start_time:
            latency = time.time() - self.start_time
            self.metrics_collector.observe_histogram(self.metric_name, latency)

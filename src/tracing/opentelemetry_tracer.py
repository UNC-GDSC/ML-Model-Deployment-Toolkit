"""OpenTelemetry distributed tracing integration."""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DistributedTracer:
    """
    Distributed tracing using OpenTelemetry.

    Provides end-to-end visibility across microservices.
    """

    def __init__(
        self,
        service_name: str = "ml-model-service",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        environment: str = "production"
    ):
        """
        Initialize distributed tracer.

        Args:
            service_name: Name of the service
            jaeger_host: Jaeger collector host
            jaeger_port: Jaeger collector port
            environment: Deployment environment
        """
        if not OTEL_AVAILABLE:
            raise ImportError(
                "OpenTelemetry is not installed. "
                "Install it with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-jaeger opentelemetry-instrumentation-fastapi "
                "opentelemetry-instrumentation-requests"
            )

        self.service_name = service_name

        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.instance.id": f"{service_name}-1",
            "deployment.environment": environment
        })

        # Setup tracer provider
        provider = TracerProvider(resource=resource)

        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )

        # Add span processor
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info(f"Initialized OpenTelemetry tracing for {service_name}")

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing an operation.

        Args:
            operation_name: Name of the operation
            attributes: Additional span attributes

        Yields:
            Span object
        """
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            # Record start time
            start_time = time.time()

            try:
                yield span
            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                # Record duration
                duration = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration)

    def trace_prediction(self, model_version: str):
        """
        Decorator for tracing predictions.

        Args:
            model_version: Model version

        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.trace_operation(
                    "model.predict",
                    {"model.version": model_version}
                ) as span:
                    # Extract features info if available
                    if args:
                        features = args[0]
                        if hasattr(features, 'shape'):
                            span.set_attribute("features.shape", str(features.shape))

                    result = func(*args, **kwargs)

                    # Add result metadata
                    if isinstance(result, dict):
                        if 'latency_ms' in result:
                            span.set_attribute("prediction.latency_ms", result['latency_ms'])
                        if 'prediction' in result:
                            span.set_attribute("prediction.value", str(result['prediction']))

                    return result
            return wrapper
        return decorator

    def instrument_fastapi(self, app):
        """
        Instrument FastAPI application.

        Args:
            app: FastAPI application
        """
        if OTEL_AVAILABLE:
            FastAPIInstrumentor.instrument_app(app)
            RequestsInstrumentor().instrument()
            logger.info("Instrumented FastAPI with OpenTelemetry")

    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new span.

        Args:
            name: Span name
            attributes: Span attributes

        Returns:
            Span object
        """
        span = self.tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span

    def add_event(self, span, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Add event to span.

        Args:
            span: Span object
            name: Event name
            attributes: Event attributes
        """
        span.add_event(name, attributes or {})


class TracedModel:
    """
    Wrapper to add tracing to model predictions.
    """

    def __init__(self, model, tracer: DistributedTracer):
        """
        Initialize traced model.

        Args:
            model: Base model instance
            tracer: DistributedTracer instance
        """
        self.model = model
        self.tracer = tracer

    def predict_with_preprocessing(self, features, return_probabilities=False):
        """
        Make prediction with tracing.

        Args:
            features: Input features
            return_probabilities: Whether to return probabilities

        Returns:
            Prediction result
        """
        with self.tracer.trace_operation(
            "model.predict_with_preprocessing",
            {
                "model.version": self.model.model_version,
                "model.type": self.model.__class__.__name__,
                "return_probabilities": return_probabilities
            }
        ) as span:
            # Preprocess
            with self.tracer.trace_operation("model.preprocess"):
                processed = self.model.preprocess(features)
                span.set_attribute("features.processed_shape", str(processed.shape))

            # Predict
            with self.tracer.trace_operation("model.inference") as predict_span:
                prediction = self.model.predict(processed)
                predict_span.set_attribute("prediction.raw", str(prediction))

            # Postprocess
            with self.tracer.trace_operation("model.postprocess"):
                result = self.model.postprocess(prediction)

            # Get full result
            full_result = self.model.predict_with_preprocessing(
                features,
                return_probabilities
            )

            # Add metrics to span
            span.set_attribute("prediction.latency_ms", full_result.get('latency_ms', 0))

            return full_result

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped model."""
        return getattr(self.model, name)

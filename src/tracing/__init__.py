"""Distributed tracing package."""

from src.tracing.opentelemetry_tracer import DistributedTracer, TracedModel

__all__ = ["DistributedTracer", "TracedModel"]

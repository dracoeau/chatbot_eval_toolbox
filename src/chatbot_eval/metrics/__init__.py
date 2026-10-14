"""Métriques d'évaluation. L'import des sous-modules enregistre les métriques dans le registre."""

from . import llm_judge, semantic, simple  # noqa: F401
from .base import (
    METRIC_REGISTRY,
    EvalSample,
    Metric,
    MetricError,
    MetricResult,
    build_metrics,
    register_metric,
)

__all__ = [
    "METRIC_REGISTRY",
    "EvalSample",
    "Metric",
    "MetricError",
    "MetricResult",
    "build_metrics",
    "register_metric",
]

"""Interface commune des métriques et registre."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import EvalConfig, MetricConfig
from ..dataset import TestCase


@dataclass
class EvalSample:
    """Tout ce qu'une métrique peut exploiter pour un cas de test."""

    case: TestCase
    actual_output: str
    response: Any = None  # JSON complet renvoyé par le chatbot

    @property
    def input(self) -> str:
        return self.case.input

    @property
    def expected_output(self) -> str | None:
        return self.case.expected_output


@dataclass
class MetricResult:
    score: float | None = None
    passed: bool | None = None
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class MetricError(ValueError):
    pass


class Metric(ABC):
    """Classe de base. Implémenter `evaluate` (sync) ou `a_evaluate` (async)."""

    default_threshold: float | None = None
    requires_expected_output: bool = False

    def __init__(self, config: MetricConfig, eval_config: EvalConfig):
        self.config = config
        self.eval_config = eval_config
        self.name = config.display_name
        self.params = config.params
        self.threshold = config.threshold if config.threshold is not None else self.default_threshold

    def evaluate(self, sample: EvalSample) -> MetricResult:
        raise NotImplementedError

    async def a_evaluate(self, sample: EvalSample) -> MetricResult:
        return await asyncio.to_thread(self.evaluate, sample)

    async def run(self, sample: EvalSample) -> MetricResult:
        if self.requires_expected_output and not sample.expected_output:
            return MetricResult(reason="Ignoré : pas de réponse attendue ('answer') pour ce cas.")
        try:
            return await self.a_evaluate(sample)
        except Exception as exc:  # une métrique en échec ne doit pas faire tomber la session
            return MetricResult(reason=f"Erreur métrique : {type(exc).__name__}: {exc}")

    def passed_from_score(self, score: float | None) -> bool | None:
        if score is None:
            return None
        return score >= (self.threshold if self.threshold is not None else 1.0)


METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(metric_type: str) -> Callable[[type[Metric]], type[Metric]]:
    """Décorateur pour enregistrer une métrique : `@register_metric("ma_metrique")`."""

    def _decorator(cls: type[Metric]) -> type[Metric]:
        METRIC_REGISTRY[metric_type] = cls
        return cls

    return _decorator


def build_metrics(eval_config: EvalConfig) -> list[Metric]:
    metrics = []
    for metric_config in eval_config.metrics:
        cls = METRIC_REGISTRY.get(metric_config.type)
        if cls is None:
            raise MetricError(
                f"Métrique inconnue : '{metric_config.type}'. Disponibles : {sorted(METRIC_REGISTRY)}"
            )
        metrics.append(cls(metric_config, eval_config))
    return metrics

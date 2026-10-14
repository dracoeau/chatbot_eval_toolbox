"""Orchestration : dataset -> appels chatbot -> métriques -> rapport Excel."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from .client import OUTPUT_FIELD, ChatbotClient, ChatbotResponse
from .config import EvalConfig
from .dataset import TestCase, load_dataset
from .metrics import EvalSample, Metric, MetricResult, build_metrics
from .report import build_summary, write_report
from .utils import flatten_dict

logger = logging.getLogger(__name__)

RESPONSE_PREFIX = "chatbot"


def load_custom_metric_modules(modules: list[str], base_dir: Path | None = None) -> None:
    """Importe des modules (nom importable ou chemin .py) qui enregistrent des métriques."""
    if base_dir and str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    for module in modules:
        if module.endswith(".py"):
            path = Path(module)
            if not path.is_absolute() and base_dir:
                path = base_dir / path
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Impossible de charger {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(module)


async def evaluate_case(
    metrics: list[Metric], case: TestCase, response: ChatbotResponse, semaphore: asyncio.Semaphore
) -> dict[str, MetricResult]:
    if not response.ok:
        return {m.name: MetricResult(reason="Ignoré : appel au chatbot en erreur") for m in metrics}
    output = response.actual_output
    if output is None:
        return {m.name: MetricResult(reason=f"Ignoré : champ '{OUTPUT_FIELD}' absent de la réponse") for m in metrics}
    sample = EvalSample(case=case, actual_output=output, response=response.raw)

    async def _run(metric: Metric) -> MetricResult:
        async with semaphore:
            return await metric.run(sample)

    results = await asyncio.gather(*(_run(m) for m in metrics))
    return {m.name: r for m, r in zip(metrics, results)}


def build_row(case: TestCase, response: ChatbotResponse, metric_results: dict[str, MetricResult]) -> dict[str, Any]:
    row: dict[str, Any] = dict(case.raw_columns)
    row["payload"] = case.payload()
    row["actual_output"] = response.actual_output
    row["status_code"] = response.status_code
    row["latency_s"] = None if response.latency_s is None else round(response.latency_s, 3)
    row["error"] = response.error
    if response.raw is not None:
        raw = response.raw if isinstance(response.raw, dict) else {"raw": response.raw}
        row.update(flatten_dict(raw, prefix=RESPONSE_PREFIX))
    for name, result in metric_results.items():
        row[f"{name}_score"] = result.score
        row[f"{name}_passed"] = result.passed
        row[f"{name}_reason"] = result.reason
    return row


async def run_evaluation_async(
    config: EvalConfig,
    limit: int | None = None,
    output_path: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    config_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    load_custom_metric_modules(config.custom_metrics_modules, base_dir=config_dir)
    metrics = build_metrics(config)

    cases = load_dataset(config.dataset_path, sheet=config.dataset_sheet, limit=limit)
    logger.info("%d cas de test chargés depuis %s", len(cases), config.dataset_path)

    async with ChatbotClient(config.chatbot, transport=transport) as client:
        logger.info("Appels au chatbot : %s", config.chatbot.invoke_url)
        responses = await client.invoke_all(cases)

    semaphore = asyncio.Semaphore(config.metrics_concurrency)
    logger.info("Calcul de %d métrique(s)...", len(metrics))
    all_results = await asyncio.gather(*(evaluate_case(metrics, c, r, semaphore) for c, r in zip(cases, responses)))

    rows = [build_row(c, r, m) for c, r, m in zip(cases, responses, all_results)]
    results = pd.DataFrame(rows)
    metric_names = [m.name for m in metrics]
    # Ordre : colonnes du dataset / appel, champs de la réponse du chatbot, puis métriques.
    metric_cols = [f"{n}_{s}" for n in metric_names for s in ("score", "passed", "reason")]
    response_cols = [c for c in results.columns if c.startswith(f"{RESPONSE_PREFIX}.")]
    other_cols = [c for c in results.columns if c not in metric_cols and c not in response_cols]
    results = results[other_cols + response_cols + [c for c in metric_cols if c in results.columns]]
    path = write_report(output_path or config.resolved_output_path(), results, metric_names, config.public_dict())
    logger.info("Rapport écrit : %s", path)
    return results, path


def run_evaluation(config: EvalConfig, **kwargs: Any) -> tuple[pd.DataFrame, Path]:
    """Version synchrone de `run_evaluation_async`. Retourne (DataFrame des résultats, chemin du rapport)."""
    return asyncio.run(run_evaluation_async(config, **kwargs))


def summarize(results: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    return build_summary(results, [m.display_name for m in config.metrics])

"""Métriques LLM-as-a-judge via DeepEval, avec un modèle Ollama local comme juge."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .base import EvalSample, Metric, MetricError, MetricResult, register_metric
from ..utils import get_path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# DeepEval est importé paresseusement : il n'est chargé que si une métrique LLM est utilisée.


def _params_enum():
    try:  # deepeval récent
        from deepeval.test_case import SingleTurnParams
        return SingleTurnParams
    except ImportError:  # versions plus anciennes
        from deepeval.test_case import LLMTestCaseParams
        return LLMTestCaseParams


@lru_cache(maxsize=None)
def get_ollama_judge(model: str, base_url: str, temperature: float):
    from deepeval.models import OllamaModel

    return OllamaModel(model=model, base_url=base_url, temperature=temperature)


def _to_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [v if isinstance(v, str) else str(v) for v in value]
    return [str(value)]


class DeepEvalMetric(Metric):
    """Base des métriques DeepEval.

    params communs :
      model                    : surcharge du modèle juge Ollama
      retrieval_context_field  : chemin dans la réponse du chatbot (ex. "metadata.sources")
      retrieval_context_column : ou colonne du dataset (liste via `[json]`)
      context_column           : contexte de référence (colonne du dataset)
    """

    default_threshold = 0.5

    def judge(self):
        ollama = self.eval_config.ollama
        return get_ollama_judge(self.params.get("model", ollama.judge_model), ollama.base_url, ollama.temperature)

    def build_test_case(self, sample: EvalSample):
        from deepeval.test_case import LLMTestCase

        retrieval_context = None
        if "retrieval_context_field" in self.params:
            retrieval_context = get_path(sample.response, self.params["retrieval_context_field"])
        elif "retrieval_context_column" in self.params:
            retrieval_context = sample.case.get(self.params["retrieval_context_column"])
        context = sample.case.get(self.params["context_column"]) if "context_column" in self.params else None

        return LLMTestCase(
            input=sample.input,
            actual_output=sample.actual_output,
            expected_output=sample.expected_output,
            retrieval_context=_to_str_list(retrieval_context),
            context=_to_str_list(context),
        )

    def build_deepeval_metric(self):
        """Nouvelle instance à chaque évaluation : les métriques DeepEval sont stateful."""
        raise NotImplementedError

    async def a_evaluate(self, sample: EvalSample) -> MetricResult:
        metric = self.build_deepeval_metric()
        test_case = self.build_test_case(sample)
        try:
            await metric.a_measure(test_case, _show_indicator=False)
        except TypeError:
            await metric.a_measure(test_case)
        score = metric.score
        return MetricResult(
            score=None if score is None else round(float(score), 4),
            passed=bool(metric.success) if metric.success is not None else self.passed_from_score(score),
            reason=metric.reason or "",
        )


def _parse_eval_params(names: list[str]) -> list[Any]:
    valid = {p.value: p for p in _params_enum()}
    try:
        return [valid[n.lower()] for n in names]
    except KeyError as exc:
        raise MetricError(f"evaluation_params invalide {exc}. Valeurs possibles : {sorted(valid)}") from None


@register_metric("geval")
class GEvalMetric(DeepEvalMetric):
    """Métrique custom définie en langage naturel (G-Eval).

    params :
      criteria          : critère à évaluer (ou evaluation_steps : liste d'étapes)
      evaluation_params : champs donnés au juge, parmi input, actual_output, expected_output,
                          context, retrieval_context (défaut : input + actual_output [+ expected_output])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.params.get("criteria") or self.params.get("evaluation_steps")):
            raise MetricError(f"{self.name} : 'criteria' ou 'evaluation_steps' requis dans params")
        names = self.params.get("evaluation_params") or self.default_eval_params()
        self.evaluation_params = _parse_eval_params(names)
        self.requires_expected_output = "expected_output" in [p.value for p in self.evaluation_params]

    def default_eval_params(self) -> list[str]:
        return ["input", "actual_output", "expected_output"]

    def criteria(self) -> str | None:
        return self.params.get("criteria")

    def build_deepeval_metric(self):
        from deepeval.metrics import GEval

        steps = self.params.get("evaluation_steps")
        return GEval(
            name=self.name,
            criteria=None if steps else self.criteria(),
            evaluation_steps=steps,
            evaluation_params=self.evaluation_params,
            model=self.judge(),
            threshold=self.threshold,
            async_mode=True,
        )


@register_metric("correctness")
class Correctness(GEvalMetric):
    """Preset G-Eval : la réponse est-elle correcte au regard de la réponse attendue ?"""

    DEFAULT_CRITERIA = (
        "Détermine si la réponse effective ('actual output') est factuellement correcte et complète "
        "par rapport à la réponse attendue ('expected output'). Pénalise fortement les contradictions "
        "et les informations manquantes importantes ; les différences de formulation sont acceptables."
    )

    def __init__(self, config, eval_config):
        config = config.model_copy(deep=True)
        config.params.setdefault("criteria", self.DEFAULT_CRITERIA)
        super().__init__(config, eval_config)


@register_metric("answer_relevancy")
class AnswerRelevancy(DeepEvalMetric):
    """La réponse répond-elle à la question posée ? (DeepEval AnswerRelevancyMetric)"""

    def build_deepeval_metric(self):
        from deepeval.metrics import AnswerRelevancyMetric

        return AnswerRelevancyMetric(threshold=self.threshold, model=self.judge(), include_reason=True, async_mode=True)


@register_metric("faithfulness")
class Faithfulness(DeepEvalMetric):
    """La réponse est-elle fidèle au contexte récupéré ? Requiert retrieval_context_field/column."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not {"retrieval_context_field", "retrieval_context_column"} & set(self.params):
            raise MetricError(f"{self.name} : 'retrieval_context_field' ou 'retrieval_context_column' requis")

    def build_deepeval_metric(self):
        from deepeval.metrics import FaithfulnessMetric

        return FaithfulnessMetric(threshold=self.threshold, model=self.judge(), include_reason=True, async_mode=True)

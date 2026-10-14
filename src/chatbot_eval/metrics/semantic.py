"""Similarité sémantique via embeddings Ollama (similarité cosinus)."""

from __future__ import annotations

import numpy as np

from .base import EvalSample, Metric, MetricResult, register_metric


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


@register_metric("semantic_similarity")
class SemanticSimilarity(Metric):
    """Cosinus entre l'embedding de la réponse et celui de `answer`.

    params : model (défaut : ollama.embedding_model), compare_to_column (défaut : answer)
    """

    default_threshold = 0.8
    requires_expected_output = True

    _cache: dict[tuple[str, str], list[float]] = {}  # partagé entre instances : (modèle, texte) -> vecteur

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ollama import AsyncClient

        self.model = self.params.get("model", self.eval_config.ollama.embedding_model)
        self.client = AsyncClient(host=self.eval_config.ollama.base_url)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if (self.model, t) not in self._cache]
        if missing:
            response = await self.client.embed(model=self.model, input=missing)
            for text, vector in zip(missing, response.embeddings):
                self._cache[(self.model, text)] = list(vector)
        return [self._cache[(self.model, t)] for t in texts]

    async def a_evaluate(self, sample: EvalSample) -> MetricResult:
        reference = sample.expected_output
        if "compare_to_column" in self.params:
            reference = sample.case.get(self.params["compare_to_column"])
            if reference is None:
                return MetricResult(reason=f"Ignoré : colonne {self.params['compare_to_column']!r} vide")
        actual_vec, expected_vec = await self._embed([sample.actual_output, str(reference)])
        score = cosine_similarity(actual_vec, expected_vec)
        return MetricResult(
            score=round(score, 4),
            passed=self.passed_from_score(score),
            reason=f"Similarité cosinus ({self.model}) = {score:.3f} (seuil {self.threshold})",
        )

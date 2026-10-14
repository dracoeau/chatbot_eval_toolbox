"""Exemple de métriques Python personnalisées.

Déclarer le fichier dans la config : "custom_metrics_modules": ["custom_metrics.py"]
puis l'utiliser : {"type": "max_length", "params": {"max_chars": 500}}
"""

from chatbot_eval import EvalSample, Metric, MetricResult, register_metric


@register_metric("max_length")
class MaxLength(Metric):
    """La réponse ne doit pas dépasser `max_chars` caractères."""

    def evaluate(self, sample: EvalSample) -> MetricResult:
        max_chars = int(self.params.get("max_chars", 500))
        length = len(sample.actual_output)
        return MetricResult(
            score=min(1.0, max_chars / length) if length else 1.0,
            passed=length <= max_chars,
            reason=f"{length} caractères (max {max_chars})",
        )


@register_metric("response_field_equals")
class ResponseFieldEquals(Metric):
    """Compare un champ de la réponse brute du chatbot à une colonne du dataset.

    params : field (chemin pointé dans la réponse, ex. "metadata.intent"), column (colonne du dataset)
    """

    def evaluate(self, sample: EvalSample) -> MetricResult:
        from chatbot_eval.utils import get_path

        actual = get_path(sample.response, self.params["field"])
        expected = sample.case.get(self.params["column"])
        ok = str(actual) == str(expected)
        return MetricResult(score=float(ok), passed=ok, reason=f"{actual!r} vs attendu {expected!r}")

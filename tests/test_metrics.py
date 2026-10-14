import pytest

from chatbot_eval.config import MetricConfig
from chatbot_eval.dataset import TestCase
from chatbot_eval.metrics import EvalSample, build_metrics
from chatbot_eval.metrics.semantic import SemanticSimilarity, cosine_similarity


def make_metric(config_factory, type_, **kwargs):
    config = config_factory(metrics=[{"type": type_, **kwargs}])
    return build_metrics(config)[0]


def sample(output, expected="Bonjour, comment puis-je vous aider ?", **fields):
    case = TestCase(row=2, input="Bonjour", expected_output=expected, fields=fields)
    return EvalSample(case=case, actual_output=output, response={"output": output})


async def test_exact_match_normalization(config_factory):
    metric = make_metric(config_factory, "exact_match", params={"strip_punctuation": True})
    assert (await metric.run(sample("BONJOUR   comment puis-je vous aider"))).passed is True
    assert (await metric.run(sample("Bonjour, comment vous aider ?"))).passed is False
    strict = make_metric(config_factory, "exact_match", params={"ignore_case": False})
    assert (await strict.run(sample("bonjour, comment puis-je vous aider ?"))).passed is False


async def test_exact_match_skipped_without_expected(config_factory):
    metric = make_metric(config_factory, "exact_match")
    result = await metric.run(sample("x", expected=None))
    assert result.passed is None and "Ignoré" in result.reason


async def test_contains_from_column(config_factory):
    metric = make_metric(config_factory, "contains", params={"substrings_column": "keywords"})
    result = await metric.run(sample("Bonjour à vous", keywords=["bonjour", "aider"]))
    assert result.score == 0.5 and result.passed is False
    any_metric = make_metric(config_factory, "contains", params={"substrings_column": "keywords", "mode": "any"})
    assert (await any_metric.run(sample("Bonjour à vous", keywords=["bonjour", "aider"]))).passed is True


async def test_contains_semicolon_and_accents(config_factory):
    metric = make_metric(
        config_factory, "contains", params={"substrings": "délai;jours", "strip_accents": True}
    )
    result = await metric.run(sample("Le DELAI est de 3 jours"))
    assert result.passed is True


async def test_not_contains(config_factory):
    metric = make_metric(config_factory, "not_contains", params={"substrings": ["désolé"]})
    assert (await metric.run(sample("Désolé, je ne sais pas"))).passed is False
    assert (await metric.run(sample("Voici la réponse"))).passed is True


async def test_regex(config_factory):
    metric = make_metric(config_factory, "regex_match", params={"pattern": r"\d+(,\d+)? ?€"})
    assert (await metric.run(sample("Cela coûte 19,99 € par mois"))).passed is True
    assert (await metric.run(sample("Gratuit"))).passed is False


async def test_metric_error_is_captured(config_factory):
    metric = make_metric(config_factory, "regex_match", params={})
    result = await metric.run(sample("x"))
    assert result.score is None and "Erreur métrique" in result.reason


def test_unknown_metric(config_factory):
    with pytest.raises(ValueError, match="Métrique inconnue"):
        make_metric(config_factory, "nope")


def test_duplicate_metric_names(config_factory):
    with pytest.raises(ValueError, match="double"):
        config_factory(metrics=[{"type": "contains"}, {"type": "contains"}])


def test_cosine():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


async def test_semantic_similarity_with_mocked_ollama(config_factory, monkeypatch):
    metric = make_metric(config_factory, "semantic_similarity", threshold=0.9)
    assert isinstance(metric, SemanticSimilarity)
    vectors = {"a": [1.0, 0.0], "b": [0.8, 0.6]}

    class FakeResponse:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    async def fake_embed(model, input):
        return FakeResponse([vectors[t] for t in input])

    monkeypatch.setattr(metric.client, "embed", fake_embed)
    result = await metric.run(sample("a", expected="b"))
    assert result.score == pytest.approx(0.8)
    assert result.passed is False


async def test_geval_with_mocked_deepeval(config_factory, monkeypatch):
    metric = make_metric(
        config_factory,
        "geval",
        name="ton",
        threshold=0.5,
        params={"criteria": "Ton poli ?", "evaluation_params": ["input", "actual_output"]},
    )
    assert metric.requires_expected_output is False

    class FakeGEval:
        score, success, reason = 0.9, True, "Poli"

        async def a_measure(self, test_case, _show_indicator=True):
            assert test_case.actual_output == "Bonjour"

    monkeypatch.setattr(metric, "build_deepeval_metric", lambda: FakeGEval())
    result = await metric.run(sample("Bonjour", expected=None))
    assert (result.score, result.passed, result.reason) == (0.9, True, "Poli")


def test_geval_requires_criteria(config_factory):
    with pytest.raises(ValueError, match="criteria"):
        make_metric(config_factory, "geval", params={})


def test_metric_config_name():
    assert MetricConfig(type="contains").display_name == "contains"

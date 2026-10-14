"""Métriques déterministes : exact match, présence de sous-chaînes, regex."""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

from .base import EvalSample, Metric, MetricError, MetricResult, register_metric


def normalize_text(
    text: str,
    *,
    ignore_case: bool = True,
    strip_punctuation: bool = False,
    strip_accents: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if ignore_case:
        text = text.casefold()
    if strip_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    if strip_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation + "«»’“”…"))
    if collapse_whitespace:
        text = " ".join(text.split())
    return text


def _normalize_options(params: dict[str, Any], **defaults: bool) -> dict[str, bool]:
    keys = ("ignore_case", "strip_punctuation", "strip_accents", "collapse_whitespace")
    options = {"ignore_case": True, "strip_punctuation": False, "strip_accents": False, "collapse_whitespace": True}
    options.update(defaults)
    options.update({k: bool(params[k]) for k in keys if k in params})
    if "case_sensitive" in params:
        options["ignore_case"] = not params["case_sensitive"]
    return options


@register_metric("exact_match")
class ExactMatch(Metric):
    """Égalité stricte (après normalisation optionnelle) entre réponse et `answer`.

    params : ignore_case (True), strip_punctuation (False), strip_accents (False), collapse_whitespace (True)
    """

    requires_expected_output = True

    def evaluate(self, sample: EvalSample) -> MetricResult:
        options = _normalize_options(self.params)
        match = normalize_text(sample.actual_output, **options) == normalize_text(sample.expected_output, **options)
        return MetricResult(
            score=1.0 if match else 0.0,
            passed=match,
            reason="Correspondance exacte" if match else "La réponse diffère de la réponse attendue",
        )


def _resolve_substrings(metric: Metric, sample: EvalSample) -> list[str]:
    params = metric.params
    if "substrings" in params:
        values = params["substrings"]
    elif "substrings_column" in params:
        values = sample.case.get(params["substrings_column"])
    else:
        raise MetricError(f"{metric.name} : indiquer 'substrings' ou 'substrings_column' dans params")
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(params.get("separator", ";"))
    elif not isinstance(values, list):
        values = [values]
    return [str(v).strip() for v in values if str(v).strip()]


@register_metric("contains")
class Contains(Metric):
    """Vérifie la présence de sous-chaînes dans la réponse.

    params :
      substrings: liste fixe, OU substrings_column: colonne du dataset (liste JSON via `[json]`
                  ou texte séparé par `separator`, `;` par défaut)
      mode: "all" (défaut) | "any"
      + options de normalisation (case_sensitive / ignore_case, strip_accents...)
    score = proportion de sous-chaînes trouvées.
    """

    negate = False

    def evaluate(self, sample: EvalSample) -> MetricResult:
        substrings = _resolve_substrings(self, sample)
        if not substrings:
            return MetricResult(reason="Ignoré : aucune sous-chaîne à vérifier")
        options = _normalize_options(self.params)
        haystack = normalize_text(sample.actual_output, **options)
        found = [s for s in substrings if normalize_text(s, **options) in haystack]
        missing = [s for s in substrings if s not in found]
        mode = self.params.get("mode", "all")
        if mode not in {"all", "any"}:
            raise MetricError(f"{self.name} : mode doit valoir 'all' ou 'any'")

        if self.negate:
            score = 1.0 - len(found) / len(substrings)
            passed = not found if mode == "all" else len(missing) > 0
            reason = f"Interdits trouvés : {found}" if found else "Aucune sous-chaîne interdite trouvée"
        else:
            score = len(found) / len(substrings)
            passed = not missing if mode == "all" else len(found) > 0
            reason = f"Trouvés : {found}" + (f" | Manquants : {missing}" if missing else "")
        if self.config.threshold is not None:
            passed = score >= self.config.threshold
        return MetricResult(score=score, passed=passed, reason=reason)


@register_metric("not_contains")
class NotContains(Contains):
    """Vérifie l'absence de sous-chaînes (mêmes params que `contains`).

    mode "all" (défaut) : aucune ne doit apparaître ; "any" : au moins une doit être absente.
    """

    negate = True


@register_metric("regex_match")
class RegexMatch(Metric):
    """`re.search(pattern, réponse)`. params : pattern (ou pattern_column), ignore_case (False)."""

    def evaluate(self, sample: EvalSample) -> MetricResult:
        pattern = self.params.get("pattern")
        if pattern is None and "pattern_column" in self.params:
            pattern = sample.case.get(self.params["pattern_column"])
        if not pattern:
            raise MetricError(f"{self.name} : 'pattern' manquant")
        flags = re.IGNORECASE if self.params.get("ignore_case", False) else 0
        match = re.search(str(pattern), sample.actual_output, flags | re.DOTALL)
        return MetricResult(
            score=1.0 if match else 0.0,
            passed=bool(match),
            reason=f"Trouvé : {match.group(0)!r}" if match else f"Motif {pattern!r} absent",
        )

"""Fonctions utilitaires (manipulation de dictionnaires, sérialisation)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

EXCEL_CELL_MAX_CHARS = 32_767


def is_empty(value: Any) -> bool:
    """True pour None, NaN, NaT et chaînes vides."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def set_nested(target: dict, dotted_key: str, value: Any) -> None:
    """Affecte `value` dans `target` en suivant une clé pointée (`a.b.c`)."""
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError(f"Clé invalide : {dotted_key!r}")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def deep_merge(base: dict, override: dict) -> dict:
    """Fusion récursive : les valeurs de `override` priment."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_path(data: Any, dotted_path: str, default: Any = None) -> Any:
    """Lit une valeur via un chemin pointé (`metadata.sources`), index de liste acceptés."""
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return default
    return current


def flatten_dict(data: Any, prefix: str = "", max_depth: int = 5) -> dict[str, Any]:
    """Aplatit un dict imbriqué en colonnes `a.b.c`. Les listes restent sérialisées en JSON."""
    flat: dict[str, Any] = {}

    def _walk(value: Any, key: str, depth: int) -> None:
        if isinstance(value, dict) and value and depth < max_depth:
            for sub_key, sub_value in value.items():
                _walk(sub_value, f"{key}.{sub_key}" if key else str(sub_key), depth + 1)
        else:
            flat[key or "value"] = to_cell_value(value)

    _walk(data, prefix, 0)
    return flat


def to_json(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)


def to_cell_value(value: Any) -> Any:
    """Convertit une valeur Python en valeur compatible Excel."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        value = to_json(value)
    if isinstance(value, str) and len(value) > EXCEL_CELL_MAX_CHARS:
        value = value[: EXCEL_CELL_MAX_CHARS - 15] + "…[tronqué]"
    return value

"""Lecture du golden dataset.

Conventions de colonnes (insensibles à la casse) :

- ``input``                : question envoyée au chatbot (obligatoire)
- ``answer``               : réponse attendue
- ``chat_history[json]``   : historique ``[{"role": "user", "content": "..."}, ...]``
- ``kwargs.<cle>``         : kwargs ajoutés au payload (notation pointée pour l'imbrication)
- ``kwargs[json]``         : objet JSON complet fusionné dans les kwargs
- toute autre colonne      : conservée et accessible aux métriques

Le suffixe ``[json]`` sur un nom de colonne indique que la cellule contient du JSON
qui doit être parsé. Sans suffixe, la valeur est prise telle quelle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import deep_merge, is_empty, set_nested, to_json

JSON_SUFFIX = re.compile(r"\s*\[json\]\s*$", re.IGNORECASE)

INPUT_COL = "input"
ANSWER_COL = "answer"
HISTORY_COL = "chat_history"
KWARGS_COL = "kwargs"
KWARGS_PREFIX = "kwargs."


class DatasetError(ValueError):
    pass


@dataclass
class Column:
    header: str  # nom tel qu'écrit dans le fichier, ex. "kwargs.filters[json]"
    name: str  # nom logique sans suffixe, ex. "kwargs.filters"
    is_json: bool

    @property
    def key(self) -> str:
        return self.name.lower()


@dataclass
class TestCase:
    __test__ = False  # évite la collecte par pytest

    row: int  # numéro de ligne dans le fichier (1 = en-tête pour Excel/CSV)
    input: str
    expected_output: str | None = None
    chat_history: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)  # nom logique -> valeur parsée
    raw_columns: dict[str, Any] = field(default_factory=dict)  # en-tête d'origine -> valeur pour le rapport

    def get(self, name: str, default: Any = None) -> Any:
        """Accès insensible à la casse à une colonne (nom logique, sans `[json]`)."""
        lowered = name.lower()
        for key, value in self.fields.items():
            if key.lower() == lowered:
                return value
        return default

    def payload(self) -> dict[str, Any]:
        return {"input": {"input": self.input, "chat_history": self.chat_history, **self.kwargs}}


def parse_column(header: Any) -> Column:
    header = str(header).strip()
    is_json = bool(JSON_SUFFIX.search(header))
    name = JSON_SUFFIX.sub("", header).strip()
    return Column(header=header, name=name, is_json=is_json)


def _parse_cell(value: Any, column: Column, row: int) -> Any:
    if is_empty(value):
        return None
    if column.is_json and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"Ligne {row}, colonne '{column.header}' : JSON invalide ({exc.msg}, position {exc.pos})."
            ) from exc
    if isinstance(value, float) and value.is_integer() and not column.is_json:
        # Excel stocke les entiers en float : on restaure l'entier.
        return int(value)
    return value


def _read_frame(path: Path, sheet: str | int | None) -> tuple[pd.DataFrame, int]:
    """Retourne le DataFrame et le décalage ligne-fichier (en-tête compris)."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=object), 2
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python"), 2
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False), 1
    if suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            return pd.DataFrame(json.load(fh)), 1
    raise DatasetError(f"Format de dataset non supporté : {suffix}")


def _normalize_history(history: Any, row: int) -> list[Any]:
    if history is None:
        return []
    if not isinstance(history, list):
        raise DatasetError(
            f"Ligne {row} : 'chat_history' doit être une liste JSON (utilisez la colonne 'chat_history[json]')."
        )
    return history


def load_dataset(path: str | Path, sheet: str | int | None = None, limit: int | None = None) -> list[TestCase]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset introuvable : {path}")
    frame, row_offset = _read_frame(path, sheet)
    frame = frame.dropna(how="all")

    columns = [parse_column(c) for c in frame.columns]
    keys = [c.key for c in columns]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise DatasetError(f"Colonnes en double (après retrait de [json]) : {sorted(duplicates)}")
    if INPUT_COL not in keys:
        raise DatasetError(f"Colonne obligatoire '{INPUT_COL}' absente. Colonnes trouvées : {list(frame.columns)}")

    cases: list[TestCase] = []
    for position, (_, series) in enumerate(frame.iterrows()):
        if limit is not None and len(cases) >= limit:
            break
        row = position + row_offset
        fields: dict[str, Any] = {}
        raw_columns: dict[str, Any] = {}
        kwargs_full: dict[str, Any] = {}
        kwargs_cols: dict[str, Any] = {}

        for column, value in zip(columns, series.tolist()):
            parsed = _parse_cell(value, column, row)
            fields[column.name] = parsed
            raw_columns[column.header] = to_json(parsed, indent=2) if column.is_json and parsed is not None else parsed

            if column.key == KWARGS_COL:
                if parsed is not None and not isinstance(parsed, dict):
                    raise DatasetError(f"Ligne {row} : la colonne '{column.header}' doit contenir un objet JSON.")
                kwargs_full = parsed or {}
            elif column.key.startswith(KWARGS_PREFIX) and parsed is not None:
                set_nested(kwargs_cols, column.name[len(KWARGS_PREFIX):], parsed)

        question = fields.get(next(c.name for c in columns if c.key == INPUT_COL))
        if is_empty(question):
            raise DatasetError(f"Ligne {row} : la colonne 'input' est vide.")

        history_col = next((c for c in columns if c.key == HISTORY_COL), None)
        answer_col = next((c for c in columns if c.key == ANSWER_COL), None)
        answer = fields.get(answer_col.name) if answer_col else None

        cases.append(
            TestCase(
                row=row,
                input=str(question),
                expected_output=None if answer is None else str(answer),
                chat_history=_normalize_history(fields.get(history_col.name) if history_col else None, row),
                kwargs=deep_merge(kwargs_full, kwargs_cols),
                fields=fields,
                raw_columns=raw_columns,
            )
        )
    return cases

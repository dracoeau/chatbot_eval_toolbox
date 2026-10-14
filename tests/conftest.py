import json
from pathlib import Path

import pandas as pd
import pytest

from chatbot_eval.config import load_config


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    rows = [
        {
            "id": 1,
            "Input": "Bonjour",
            "Answer": "Bonjour ! Comment puis-je vous aider ?",
            "chat_history[json]": "[]",
            "keywords[json]": '["bonjour", "aider"]',
            "kwargs.langue": "fr",
            "kwargs.filters[json]": '{"domaine": "general", "top_k": 3}',
            "kwargs.options.debug": "",
            "kwargs[json]": '{"user": {"id": "u1"}, "langue": "en"}',
        },
        {
            "id": 2,
            "Input": "Quel délai ?",
            "Answer": "3 à 5 jours ouvrés.",
            "chat_history[json]": '[{"role": "user", "content": "Je commande"}, {"role": "assistant", "content": "Ok"}]',
            "keywords[json]": '["jours"]',
            "kwargs.langue": "fr",
            "kwargs.filters[json]": "",
            "kwargs.options.debug": "true",
            "kwargs[json]": "",
        },
    ]
    path = tmp_path / "golden.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


@pytest.fixture
def config_factory(tmp_path: Path, dataset_path: Path):
    def _make(metrics=None, **overrides):
        raw = {
            "chatbot": {
                "base_url": "bot.example.com",
                "use_case_id": "uc1",
                "token": "secret-token",
                "max_retries": 1,
            },
            "dataset_path": str(dataset_path),
            "output_path": str(tmp_path / "out" / "eval_{timestamp}.xlsx"),
            "metrics": metrics or [],
            **overrides,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_config(path)

    return _make

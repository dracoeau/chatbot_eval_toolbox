import pandas as pd
import pytest

from chatbot_eval.dataset import DatasetError, load_dataset, parse_column


def test_parse_column_json_suffix():
    col = parse_column("kwargs.filters[JSON] ")
    assert col.name == "kwargs.filters" and col.is_json
    assert not parse_column("answer").is_json


def test_load_dataset_builds_payload(dataset_path):
    first, second = load_dataset(dataset_path)

    assert first.input == "Bonjour"
    assert first.expected_output.startswith("Bonjour")
    assert first.chat_history == []
    # les colonnes kwargs.* priment sur kwargs[json]
    assert first.kwargs == {
        "user": {"id": "u1"},
        "langue": "fr",
        "filters": {"domaine": "general", "top_k": 3},
    }
    assert first.payload() == {"input": {"input": "Bonjour", "chat_history": [], **first.kwargs}}
    assert first.get("keywords") == ["bonjour", "aider"]
    assert first.raw_columns["id"] == 1  # entier restauré

    assert second.chat_history[1] == {"role": "assistant", "content": "Ok"}
    # colonne sans [json] : valeur brute (texte), imbrication par notation pointée
    assert second.kwargs == {"langue": "fr", "options": {"debug": "true"}}


def test_invalid_json_reports_row_and_column(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame([{"input": "q", "kwargs.x[json]": "{pas du json"}]).to_excel(path, index=False)
    with pytest.raises(DatasetError, match=r"Ligne 2, colonne 'kwargs.x\[json\]'"):
        load_dataset(path)


def test_missing_input_column(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame([{"question": "q"}]).to_excel(path, index=False)
    with pytest.raises(DatasetError, match="input"):
        load_dataset(path)


def test_chat_history_without_json_suffix_is_rejected(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame([{"input": "q", "chat_history": "[]"}]).to_excel(path, index=False)
    with pytest.raises(DatasetError, match=r"chat_history\[json\]"):
        load_dataset(path)


def test_limit(dataset_path):
    assert len(load_dataset(dataset_path, limit=1)) == 1

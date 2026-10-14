"""Génère `golden_dataset_example.xlsx` illustrant les conventions de colonnes.

    python examples/make_example_dataset.py
"""

from pathlib import Path

import pandas as pd

ROWS = [
    {
        "id": "Q1",
        "categorie": "salutation",
        "input": "Bonjour",
        "answer": "Bonjour ! Comment puis-je vous aider ?",
        "chat_history[json]": "[]",
        "expected_keywords[json]": '["bonjour", "aider"]',
        "kwargs.langue": "fr",
        "kwargs.filters[json]": '{"domaine": "general"}',
        "kwargs[json]": "",
    },
    {
        "id": "Q2",
        "categorie": "suivi",
        "input": "Et quel est le délai de livraison ?",
        "answer": "Le délai de livraison standard est de 3 à 5 jours ouvrés.",
        "chat_history[json]": (
            '[{"role": "user", "content": "Je veux commander un produit"},'
            ' {"role": "assistant", "content": "Très bien, quel produit souhaitez-vous ?"}]'
        ),
        "expected_keywords[json]": '["livraison", "jours"]',
        "kwargs.langue": "fr",
        "kwargs.filters[json]": '{"domaine": "logistique", "top_k": 5}',
        "kwargs[json]": '{"user": {"id": "u-42", "segment": "premium"}}',
    },
    {
        "id": "Q3",
        "categorie": "tarif",
        "input": "Combien coûte l'abonnement premium ?",
        "answer": "L'abonnement premium coûte 19,99 € par mois.",
        "chat_history[json]": "",
        "expected_keywords[json]": '["19,99", "mois"]',
        "kwargs.langue": "fr",
        "kwargs.filters[json]": "",
        "kwargs[json]": "",
    },
]

if __name__ == "__main__":
    target = Path(__file__).with_name("golden_dataset_example.xlsx")
    pd.DataFrame(ROWS).to_excel(target, index=False)
    print(f"Dataset écrit : {target}")

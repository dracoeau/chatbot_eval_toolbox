# chatbot_eval_toolbox

Module Python d'évaluation de chatbots IA à partir d'un **golden dataset** :

1. lecture du dataset (Excel, CSV, JSON, JSONL) ;
2. authentification au chatbot (token fixe dans un header) ;
3. génération des réponses : `POST https://{base_url}/usecases/{use_case_id}/invoke` ;
4. calcul des métriques : déterministes (exact match, sous-chaînes, regex), similarité sémantique
   (embeddings Ollama) et **LLM-as-a-judge** (DeepEval avec un modèle **Ollama local**) ;
5. export d'un rapport **Excel**.

## Installation

```bash
pip install -e .            # ou pip install -e ".[dev]" pour lancer les tests
# Pour les métriques sémantiques / LLM (Ollama doit tourner en local) :
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Utilisation

```bash
cp config.example.json config.json      # config.json est ignoré par git
export CHATBOT_TOKEN="mon-token"
chatbot-eval --config config.json        # ou : python -m chatbot_eval -c config.json
```

Options : `--limit N` (N premiers cas), `--output chemin.xlsx`, `--concurrency N`, `--verbose`.

Depuis Python :

```python
from chatbot_eval import load_config, run_evaluation

results_df, report_path = run_evaluation(load_config("config.json"), limit=10)
```

## Fichier de configuration

```jsonc
{
  "chatbot": {
    "base_url": "mon-api.example.com",      // https:// ajouté si absent
    "use_case_id": "mon-use-case",
    "token": "${CHATBOT_TOKEN}",            // valeur en dur ou ${VAR_ENV} / ${VAR:-défaut}
    "auth_header": "Authorization",         // header portant le token
    "auth_scheme": "Bearer",                // "" => le header contient le token brut
    "extra_headers": {},
    "timeout": 60, "max_retries": 3, "verify_ssl": true,
    "concurrency": 4                        // appels parallèles au chatbot
  },
  "dataset_path": "examples/golden_dataset_example.xlsx",
  "dataset_sheet": null,                    // nom ou index de feuille (défaut : 1re)
  "output_path": "outputs/eval_{timestamp}.xlsx",
  "ollama": { "base_url": "http://localhost:11434", "judge_model": "llama3.1:8b",
              "embedding_model": "nomic-embed-text", "temperature": 0.0 },
  "metrics_concurrency": 2,                 // évaluations de métriques en parallèle
  "custom_metrics_modules": ["examples/custom_metrics.py"],
  "metrics": [ { "type": "...", "name": "...", "threshold": 0.7, "params": { } } ]
}
```

Les chemins relatifs sont résolus par rapport au fichier de config. Le token n'est jamais écrit dans le rapport.

## Format du golden dataset

Une ligne = un cas de test. Noms de colonnes insensibles à la casse.

**Convention JSON : `nom_colonne[json]`** — une colonne dont le nom se termine par `[json]` contient
du JSON qui sera parsé (erreur explicite avec ligne et colonne si invalide). Sans ce suffixe, la valeur
est prise telle quelle.

| Colonne | Rôle |
|---|---|
| `input` *(obligatoire)* | question envoyée au chatbot |
| `answer` | réponse attendue |
| `chat_history[json]` | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` — vide = `[]` |
| `kwargs.<clé>` | kwarg simple : `kwargs.langue` = `fr` → `"langue": "fr"` |
| `kwargs.<clé>[json]` | kwarg objet/liste : `kwargs.filters[json]` = `{"top_k": 5}` → `"filters": {"top_k": 5}` |
| `kwargs.a.b` | imbrication par notation pointée → `"a": {"b": ...}` |
| `kwargs[json]` | objet JSON complet fusionné dans les kwargs (les colonnes `kwargs.*` priment) |
| toute autre colonne | conservée dans le rapport, utilisable par les métriques (`expected_keywords[json]`, `categorie`…) |

Payload envoyé :

```json
{ "input": { "input": "<input>", "chat_history": [...], "<kwargs...>": "..." } }
```

Exemple : `examples/golden_dataset_example.xlsx` (régénérable avec `python examples/make_example_dataset.py`).

## Métriques disponibles

Chaque métrique produit `<nom>_score`, `<nom>_passed`, `<nom>_reason`. `name` permet d'utiliser
plusieurs fois le même type ; `threshold` fixe le seuil de réussite.

| `type` | Description | `params` principaux |
|---|---|---|
| `exact_match` | égalité avec `answer` après normalisation | `ignore_case` (true), `strip_punctuation`, `strip_accents`, `collapse_whitespace` (true) |
| `contains` | présence de sous-chaînes ; score = proportion trouvée | `substrings` (liste ou `"a;b"`) **ou** `substrings_column`, `mode` (`all`/`any`), `case_sensitive`, `strip_accents` |
| `not_contains` | absence de sous-chaînes (mots interdits) | idem `contains` |
| `regex_match` | `re.search` sur la réponse | `pattern` ou `pattern_column`, `ignore_case` |
| `semantic_similarity` | cosinus des embeddings Ollama réponse / `answer` (seuil défaut 0.8) | `model`, `compare_to_column` |
| `geval` | **critère custom en langage naturel** jugé par LLM (DeepEval G-Eval) | `criteria` **ou** `evaluation_steps`, `evaluation_params` (`input`, `actual_output`, `expected_output`, `context`, `retrieval_context`) |
| `correctness` | preset G-Eval : exactitude vs réponse attendue | idem `geval` (critère pré-rempli) |
| `answer_relevancy` | la réponse répond-elle à la question ? (DeepEval) | `model` |
| `faithfulness` | fidélité au contexte récupéré (RAG, DeepEval) | `retrieval_context_field` (chemin dans la réponse, ex. `metadata.sources`) ou `retrieval_context_column` |

Toutes les métriques LLM acceptent `model` (surcharge du juge) et `retrieval_context_field` /
`retrieval_context_column` / `context_column`. Un cas sans `answer` est ignoré par les métriques qui en ont besoin.

### Exemple de métrique LLM custom

```json
{
  "type": "geval",
  "name": "ton_professionnel",
  "threshold": 0.7,
  "params": {
    "criteria": "La réponse adopte-t-elle un ton professionnel et vouvoie-t-elle l'utilisateur ?",
    "evaluation_params": ["input", "actual_output"]
  }
}
```

### Métrique Python custom

```python
# mes_metriques.py
from chatbot_eval import EvalSample, Metric, MetricResult, register_metric

@register_metric("max_length")
class MaxLength(Metric):
    def evaluate(self, sample: EvalSample) -> MetricResult:   # ou async def a_evaluate(...)
        ok = len(sample.actual_output) <= self.params.get("max_chars", 500)
        return MetricResult(score=float(ok), passed=ok, reason=f"{len(sample.actual_output)} caractères")
```

Puis `"custom_metrics_modules": ["mes_metriques.py"]` et `{"type": "max_length", "params": {"max_chars": 300}}`.
`sample` donne accès à `input`, `expected_output`, `actual_output`, `response` (JSON brut du chatbot)
et `case.get("colonne")` (toute colonne du dataset). Voir `examples/custom_metrics.py`.

## Rapport Excel

- **Résultats** : toutes les colonnes du dataset (colonnes `[json]` ré-indentées), `payload` envoyé,
  `actual_output` (champ `output` de la réponse), `status_code`, `latency_s`, `error`, **tous les champs
  de la réponse du chatbot aplatis** (`chatbot.output`, `chatbot.metadata.xxx`…), puis
  `score` / `passed` / `reason` par métrique (PASS en vert, FAIL en rouge).
- **Synthèse** : par métrique — cas évalués, score moyen/min, taux de réussite ; nb d'erreurs d'appel, latences.
- **Config** : paramètres de la session (sans secrets) et date d'exécution.

Un appel en erreur (HTTP 4xx/5xx après retries, timeout…) n'interrompt pas la session : la ligne est
marquée en erreur et ses métriques sont ignorées. Une métrique en erreur (ex. Ollama injoignable)
renseigne `reason` sans bloquer les autres.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

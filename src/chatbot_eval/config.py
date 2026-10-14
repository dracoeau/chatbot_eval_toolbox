"""Chargement et validation du fichier de configuration JSON."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ChatbotConfig(BaseModel):
    base_url: str
    use_case_id: str
    token: str = Field(repr=False)
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 3
    verify_ssl: bool = True
    concurrency: int = Field(default=4, ge=1)

    @property
    def invoke_url(self) -> str:
        base = self.base_url.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"https://{base}"
        return f"{base}/usecases/{self.use_case_id}/invoke"

    def headers(self) -> dict[str, str]:
        value = f"{self.auth_scheme} {self.token}".strip() if self.auth_scheme else self.token
        return {"Content-Type": "application/json", **self.extra_headers, self.auth_header: value}


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    judge_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0


class MetricConfig(BaseModel):
    type: str
    name: str | None = None
    threshold: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or self.type


class EvalConfig(BaseModel):
    chatbot: ChatbotConfig
    dataset_path: Path
    dataset_sheet: str | int | None = None
    output_path: Path
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    metrics: list[MetricConfig] = Field(default_factory=list)
    custom_metrics_modules: list[str] = Field(default_factory=list)
    metrics_concurrency: int = Field(default=2, ge=1)

    @field_validator("metrics")
    @classmethod
    def _unique_metric_names(cls, metrics: list[MetricConfig]) -> list[MetricConfig]:
        names = [m.display_name for m in metrics]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Noms de métriques en double : {sorted(duplicates)}. "
                "Utilisez le champ 'name' pour les différencier."
            )
        return metrics

    def resolved_output_path(self, now: datetime | None = None) -> Path:
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        return Path(str(self.output_path).replace("{timestamp}", stamp))

    def public_dict(self) -> dict[str, Any]:
        """Configuration sans secrets (pour le rapport)."""
        data = self.model_dump(mode="json")
        data["chatbot"].pop("token", None)
        data["chatbot"]["extra_headers"] = {k: "***" for k in data["chatbot"]["extra_headers"]}
        return data


def _interpolate_env(value: Any) -> Any:
    """Remplace `${VAR}` / `${VAR:-défaut}` par les variables d'environnement."""
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"Variable d'environnement manquante : {name}")

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_config(path: str | Path) -> EvalConfig:
    """Charge le JSON de config. Les chemins relatifs sont résolus par rapport au fichier."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    config = EvalConfig.model_validate(_interpolate_env(raw))
    base_dir = path.resolve().parent
    if not config.dataset_path.is_absolute():
        config.dataset_path = base_dir / config.dataset_path
    if not config.output_path.is_absolute():
        config.output_path = base_dir / config.output_path
    return config

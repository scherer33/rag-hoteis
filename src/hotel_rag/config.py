from pathlib import Path
from typing import Any

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

CONFIG_PATH = Path("config/llm.yaml")

# Keys accepted in config/llm.yaml, mapped to Settings fields. Secrets are
# deliberately absent: the API key must come from the environment.
_YAML_KEYS = {
    "base_url": "llm_base_url",
    "model": "llm_model",
    "temperature": "llm_temperature",
    "timeout_seconds": "llm_timeout_seconds",
}


def load_yaml_defaults(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the non-secret LLM settings from YAML, failing fast on typos.

    The file is optional. Environment variables and `.env` take precedence, so
    this only supplies defaults for local runs and Docker.
    """
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of setting: value")

    unknown = sorted(set(data) - set(_YAML_KEYS))
    if unknown:
        raise ValueError(
            f"unknown keys in {path}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(_YAML_KEYS))}. "
            "The API key belongs in LLM_API_KEY, never in this file."
        )

    return {_YAML_KEYS[key]: value for key, value in data.items() if value is not None}


class YamlDefaultsSource(PydanticBaseSettingsSource):
    """Settings source backed by config/llm.yaml, below env and .env."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return load_yaml_defaults()


class Settings(BaseSettings):
    """Runtime configuration from environment variables, .env and config/llm.yaml."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:20128/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "hotel_docs"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    abstain_min_score: float = 0.3

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlDefaultsSource(settings_cls),
            file_secret_settings,
        )

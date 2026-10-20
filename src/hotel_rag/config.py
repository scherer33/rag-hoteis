from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:20128/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "hotel_docs"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    abstain_min_score: float = 0.3

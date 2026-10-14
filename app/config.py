"""Configuration loaded once per process from environment variables or .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jwt_secret: SecretStr = SecretStr("")
    llm_provider: Literal["ollama", "groq", "mock"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen2.5:3b"
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: Literal["qdrant", "sqlite"] = "qdrant"
    qdrant_path: Path = Path("data/qdrant")
    sqlite_path: Path = Path("data/chunks.sqlite3")
    collection_name: str = "company_policies"
    docs_dir: Path = Path("docs")
    top_k: int = Field(default=5, ge=1, le=15)
    max_question_chars: int = Field(default=2000, ge=1, le=20000)
    request_timeout_seconds: float = Field(default=90.0, gt=0)


@lru_cache

def get_settings() -> Settings:
    return Settings()

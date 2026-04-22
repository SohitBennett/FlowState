"""Centralized, env-driven settings. Single source of truth for runtime config."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOWSTATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    env: Literal["local", "container", "staging", "prod"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = Field(default="change-me", min_length=8)

    # Model
    model_name: str = "ag-news-distilbert"
    model_stage: str = "Production"
    model_path: str = "./artifacts/model.onnx"
    tokenizer_path: str = "./artifacts/tokenizer"
    max_seq_len: int = 128
    num_labels: int = 4

    # Batcher
    batch_max_size: int = 32
    batch_max_wait_ms: int = 5

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — single instance per process."""
    return Settings()

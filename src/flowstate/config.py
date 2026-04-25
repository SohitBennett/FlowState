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
    model_path: str = "./artifacts/latest/onnx/model_fp16.onnx"
    tokenizer_path: str = "./artifacts/latest/model"
    max_seq_len: int = 128
    num_labels: int = 4

    # Inference runtime (0 = let ONNX Runtime auto-detect)
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    warmup_iterations: int = 5

    # Batcher
    batch_max_size: int = 32
    batch_max_wait_ms: int = 5

    # Request limits + per-key rate limit (token bucket)
    max_request_bytes: int = 1_048_576  # 1 MiB
    rate_limit_rate: float = 100.0  # tokens per second per key
    rate_limit_burst: int = 200  # bucket capacity per key

    # Redis (Phase 4)
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — single instance per process."""
    return Settings()

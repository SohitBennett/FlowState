"""Settings load from env with proper defaults and validation."""

from __future__ import annotations

import importlib

from flowstate import config as config_module


def test_settings_defaults(monkeypatch) -> None:
    for key in list(config_module.Settings.model_fields):
        monkeypatch.delenv(f"FLOWSTATE_{key.upper()}", raising=False)
    config_module.get_settings.cache_clear()
    s = config_module.get_settings()
    assert s.env == "local"
    assert s.batch_max_size == 32
    assert s.batch_max_wait_ms == 5
    assert s.max_seq_len == 128


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FLOWSTATE_BATCH_MAX_SIZE", "64")
    monkeypatch.setenv("FLOWSTATE_API_KEY", "supersecret")
    config_module.get_settings.cache_clear()
    s = config_module.get_settings()
    assert s.batch_max_size == 64
    assert s.api_key == "supersecret"


def test_module_importable() -> None:
    importlib.reload(config_module)

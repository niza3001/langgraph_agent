"""Tests for environment-driven configuration."""

import pytest

from src.config import (
    DEFAULT_MODEL,
    debug_tools_enabled,
    get_model,
    get_openrouter_api_key,
)


@pytest.mark.parametrize("value", [None, "", "your-key-here"])
def test_api_key_missing_or_placeholder_raises(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_openrouter_api_key()


def test_api_key_returned_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert get_openrouter_api_key() == "sk-or-test"


def test_model_default_and_override(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert get_model() == DEFAULT_MODEL
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    assert get_model() == "openai/gpt-oss-20b:free"


@pytest.mark.parametrize(
    ("value", "expected"), [(None, True), ("1", True), ("0", False)]
)
def test_debug_tools_enabled(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DEBUG_TOOLS", raising=False)
    else:
        monkeypatch.setenv("DEBUG_TOOLS", value)
    assert debug_tools_enabled() is expected

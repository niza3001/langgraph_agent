"""Application configuration.

Non-secret defaults live here. Secrets and overrides come from the
environment (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# OpenRouter / LLM
# ---------------------------------------------------------------------------

# openrouter/free resolves to a currently available free model and filters
# for tool calling when tools are bound. Individual :free slugs rotate and
# can start returning 404; override with OPENROUTER_MODEL to pin one.
DEFAULT_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = "https://github.com/langgraph-agent"
OPENROUTER_APP_TITLE = "langgraph-agent"
DEFAULT_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
QUIT_WORDS = frozenset({"exit", "quit", "bye", "goodbye"})
DEFAULT_THREAD_ID = "cli-session"

# Set DEBUG_TOOLS=0 to silence tool-call tracing in the CLI.
DEBUG_TOOLS_DEFAULT = True


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key or raise if it is missing."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise ValueError(
            "Set OPENROUTER_API_KEY in .env "
            "(create a key at https://openrouter.ai/keys)."
        )
    return api_key


def get_model() -> str:
    """Return the configured model slug (env override or default)."""
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def debug_tools_enabled() -> bool:
    """Whether the CLI should print tool-call traces."""
    raw = os.getenv("DEBUG_TOOLS")
    if raw is None:
        return DEBUG_TOOLS_DEFAULT
    return raw != "0"

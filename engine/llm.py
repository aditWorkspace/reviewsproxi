"""OpenRouter LLM client factory.

All pipeline stages and engine modules import get_client() from here
instead of instantiating anthropic.Anthropic() directly.

Supports multiple comma-separated keys in OPENROUTER_API_KEY for parallel
extraction (e.g. OPENROUTER_API_KEY=sk-or-key1,sk-or-key2,sk-or-key3).
"""
from __future__ import annotations

import os

from openai import OpenAI

MODEL: str = "deepseek/deepseek-v3.2"


def _parse_keys() -> list[str]:
    """Parse comma-separated API keys from the environment."""
    raw = os.environ.get("OPENROUTER_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter (first key)."""
    keys = _parse_keys()
    if not keys:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=keys[0],
    )


def get_clients() -> list[OpenAI]:
    """Return one OpenAI client per API key for parallel workloads."""
    keys = _parse_keys()
    if not keys:
        raise ValueError("OPENROUTER_API_KEY not set")
    return [
        OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k)
        for k in keys
    ]

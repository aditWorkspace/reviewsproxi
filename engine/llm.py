"""OpenRouter LLM client factory.

All pipeline stages and engine modules import get_client() from here
instead of instantiating anthropic.Anthropic() directly.
"""
from __future__ import annotations

import os

from openai import OpenAI

MODEL: str = "deepseek/deepseek-v3.2"


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

import os
import pytest


def test_get_client_returns_openai_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from engine.llm import get_client
    from openai import OpenAI
    client = get_client()
    assert isinstance(client, OpenAI)
    assert "openrouter" in str(client.base_url)


def test_model_constant():
    from engine.llm import MODEL
    assert MODEL == "deepseek/deepseek-v3.2"

"""Tests for model-dependent parameter defaults (None -> resolved per model).

Reasoning ("thinking") models spend hidden tokens against the same output
budget as the visible answer, so the historical fixed default of
max_tokens=1024 could leave them returning empty content. The constructors now
accept None and resolve a model-appropriate budget/temperature at call time.
"""

import os

import pytest

from negmas_llm.common import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_REASONING,
    DEFAULT_TEMPERATURE,
    apply_max_tokens,
    apply_temperature,
    default_max_tokens,
    default_temperature,
    is_reasoning_model,
    resolve_max_tokens,
    resolve_temperature,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        # Non-reasoning instruct models (incl. the qwen3:4b-instruct the HAN
        # developers built against) must stay on the classic budget.
        ("qwen3:4b-instruct", False),
        ("qwen2.5:1.5b", False),
        ("mistral-large-3:675b", False),
        ("gemma4:31b", False),
        # Reasoning families
        ("gpt-oss:120b", True),
        ("deepseek-v4-flash", True),
        ("deepseek-v4-pro", True),
        ("qwen3.5:397b", True),
        ("glm-5.2", True),
        ("kimi-k2.6", True),
        ("minimax-m3", True),
        ("nemotron-3-super", True),
        ("o3-mini", True),
        # Provider prefix is stripped before matching
        ("ollama_chat/deepseek-v4-flash", True),
        # Explicit thinking/reasoning variants via substring
        ("qwen3:4b-thinking-2507", True),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


def test_default_max_tokens_per_model():
    assert default_max_tokens("ollama", "qwen3:4b-instruct") == DEFAULT_MAX_TOKENS
    assert (
        default_max_tokens("ollama", "deepseek-v4-flash")
        == DEFAULT_MAX_TOKENS_REASONING
    )


def test_default_temperature_omitted_for_openai_reasoning():
    assert default_temperature("ollama", "qwen3:4b-instruct") == DEFAULT_TEMPERATURE
    assert default_temperature("openai", "o3-mini") is None


def test_resolvers_prefer_explicit_value():
    assert resolve_max_tokens("ollama", "deepseek-v4-flash", 256) == 256
    assert resolve_max_tokens("ollama", "deepseek-v4-flash", None) == (
        DEFAULT_MAX_TOKENS_REASONING
    )
    assert resolve_temperature("ollama", "qwen3:4b-instruct", 0.1) == 0.1
    assert resolve_temperature("ollama", "qwen3:4b-instruct", None) == (
        DEFAULT_TEMPERATURE
    )


def test_apply_max_tokens_routes_and_resolves():
    # Ollama -> num_predict, None resolves per model
    k: dict = {}
    apply_max_tokens(k, "ollama", "deepseek-v4-flash", None)
    assert k == {"num_predict": DEFAULT_MAX_TOKENS_REASONING}

    k = {}
    apply_max_tokens(k, "ollama", "qwen3:4b-instruct", None)
    assert k == {"num_predict": DEFAULT_MAX_TOKENS}

    # A user-supplied alias always wins
    k = {"num_predict": 50}
    apply_max_tokens(k, "ollama", "deepseek-v4-flash", None)
    assert k == {"num_predict": 50}

    # Explicit value wins over the model default
    k = {}
    apply_max_tokens(k, "ollama", "deepseek-v4-flash", 256)
    assert k == {"num_predict": 256}


def test_apply_temperature():
    k: dict = {}
    apply_temperature(k, "ollama", "qwen3:4b-instruct", None)
    assert k == {"temperature": DEFAULT_TEMPERATURE}

    # Omitted entirely for models that reject a custom temperature
    k = {}
    apply_temperature(k, "openai", "o3-mini", None)
    assert k == {}

    # Explicit value wins
    k = {}
    apply_temperature(k, "ollama", "deepseek-v4-flash", 0.3)
    assert k == {"temperature": 0.3}

    # An already-present temperature is never overwritten
    k = {"temperature": 0.9}
    apply_temperature(k, "openai", "o3-mini", None)
    assert k == {"temperature": 0.9}


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NEGMAS_LLM_DEFAULT_MAX_TOKENS", "333")
    monkeypatch.setenv("NEGMAS_LLM_DEFAULT_TEMPERATURE", "0.11")
    # Env wins for every model, reasoning or not
    assert default_max_tokens("ollama", "qwen3:4b-instruct") == 333
    assert default_max_tokens("ollama", "deepseek-v4-flash") == 333
    assert default_temperature("ollama", "qwen3:4b-instruct") == 0.11

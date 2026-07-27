"""Tests for model-dependent parameter defaults (None -> resolved at call time).

Reasoning ("thinking") models spend hidden tokens against the same output
budget as the visible answer, so *any* token cap chosen to shorten answers can
leave them returning empty content. The token budget is therefore open by
default (no cap sent) and answer length is bounded by a word budget stated in
the prompt.
"""

import os

import pytest

from negmas_llm.common import (
    DEFAULT_MAX_WORDS,
    DEFAULT_TEMPERATURE,
    apply_max_tokens,
    apply_temperature,
    default_max_tokens,
    default_max_words,
    default_temperature,
    is_reasoning_model,
    resolve_max_tokens,
    resolve_max_words,
    resolve_temperature,
    word_limit_instruction,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        # Non-reasoning instruct models.
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


def test_default_max_tokens_is_open_for_every_model():
    # No cap is sent by default, so hidden reasoning can never starve the answer.
    assert default_max_tokens("ollama", "qwen3:4b-instruct") is None
    assert default_max_tokens("ollama", "deepseek-v4-flash") is None


def test_word_budget_is_what_bounds_the_answer():
    assert default_max_words() == DEFAULT_MAX_WORDS
    assert resolve_max_words(25) == 25
    assert resolve_max_words(None) == DEFAULT_MAX_WORDS
    assert "25 words" in word_limit_instruction(25)
    assert word_limit_instruction(None) == ""
    assert word_limit_instruction(0) == ""


def test_max_words_env_override(monkeypatch):
    monkeypatch.setenv("NEGMAS_LLM_MAX_WORDS", "12")
    assert default_max_words() == 12
    monkeypatch.setenv("NEGMAS_LLM_MAX_WORDS", "0")
    assert default_max_words() is None


def test_default_temperature_omitted_for_openai_reasoning():
    assert default_temperature("ollama", "qwen3:4b-instruct") == DEFAULT_TEMPERATURE
    assert default_temperature("openai", "o3-mini") is None


def test_resolvers_prefer_explicit_value():
    assert resolve_max_tokens("ollama", "deepseek-v4-flash", 256) == 256
    assert resolve_max_tokens("ollama", "deepseek-v4-flash", None) is None
    assert resolve_temperature("ollama", "qwen3:4b-instruct", 0.1) == 0.1
    assert resolve_temperature("ollama", "qwen3:4b-instruct", None) == (
        DEFAULT_TEMPERATURE
    )


def test_apply_max_tokens_routes_and_resolves():
    # None -> nothing is sent at all, for reasoning and instruct models alike
    k: dict = {}
    apply_max_tokens(k, "ollama", "deepseek-v4-flash", None)
    assert k == {}

    k = {}
    apply_max_tokens(k, "ollama", "qwen3:4b-instruct", None)
    assert k == {}

    # A user-supplied alias always wins
    k: dict = {"num_predict": 50}
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

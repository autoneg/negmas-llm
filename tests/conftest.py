"""Pytest configuration and fixtures for negmas-llm tests.

This module provides configurable LLM testing through environment variables.
See README.md for configuration details.

Environment Variables:
    TEST_LLM_PROVIDER: LLM provider to use (default: "ollama")
    TEST_LLM_MODEL: Model name (default: provider-specific)
    TEST_LLM_API_KEY: API key for the provider (if required)
    TEST_LLM_API_BASE: Custom API base URL (optional)

Provider-Specific Variables (override TEST_LLM_* for specific providers):
    OLLAMA_MODEL, OLLAMA_API_BASE
    OPENAI_MODEL, OPENAI_API_KEY
    ANTHROPIC_MODEL, ANTHROPIC_API_KEY
    GEMINI_MODEL, GOOGLE_API_KEY
    GITHUB_COPILOT_MODEL (uses OAuth device flow, no API key needed)
    GITHUB_MODEL, GITHUB_API_KEY (for GitHub Models marketplace)
    GROQ_MODEL, GROQ_API_KEY
    MISTRAL_MODEL, MISTRAL_API_KEY
    DEEPSEEK_MODEL, DEEPSEEK_API_KEY
    OPENROUTER_MODEL, OPENROUTER_API_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest

from negmas_llm import LLMNegotiator

# =============================================================================
# Provider Configuration
# =============================================================================

# Default models for each provider
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3:0.6b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "github_copilot": "gpt-4o",  # GitHub Copilot (OAuth device flow)
    "github": "gpt-4o",  # GitHub Models marketplace
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "openai/gpt-4o-mini",
    "together_ai": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "cohere": "command-r-plus",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
}

# API key environment variable names for each provider
# Note: github_copilot uses OAuth device flow, no API key needed
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "github": "GITHUB_API_KEY",  # GitHub Models (not Copilot)
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HF_TOKEN",
}

# Providers that use OAuth or other non-API-key auth
# These are always considered "available" as auth happens at runtime
OAUTH_PROVIDERS: frozenset[str] = frozenset({"github_copilot"})

# Default API base URLs for providers that need them
DEFAULT_API_BASES: dict[str, str] = {
    "ollama": "http://localhost:11434",
}


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None

    def is_available(self) -> bool:
        """Check if this LLM configuration is available for testing."""
        # Local providers (ollama) - check if server is running
        if self.provider == "ollama":
            try:
                import urllib.request

                url = f"{self.api_base or 'http://localhost:11434'}/api/tags"
                with urllib.request.urlopen(url, timeout=2):
                    return True
            except Exception:
                return False

        # OAuth providers (github_copilot) - skip in CI as they require interactive auth
        if self.provider in OAUTH_PROVIDERS:
            # In CI environments, OAuth providers can't authenticate interactively
            return not os.environ.get("CI")

        # Cloud providers - check if API key is available
        if self.provider in API_KEY_ENV_VARS:
            return self.api_key is not None

        return False

    def to_negotiator_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for LLMNegotiator."""
        kwargs: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs


def get_llm_config(provider: str | None = None) -> LLMConfig:
    """Get LLM configuration from environment variables.

    Args:
        provider: Specific provider to configure. If None, uses TEST_LLM_PROVIDER.

    Returns:
        LLMConfig with the appropriate settings.
    """
    if provider is None:
        provider = os.environ.get("TEST_LLM_PROVIDER", "ollama")

    # At this point provider is guaranteed to be a string
    assert provider is not None
    provider_name = provider.lower()
    provider_upper = provider_name.upper().replace(
        "_", "-"
    )  # github_copilot -> GITHUB-COPILOT

    # Get model: provider-specific env var > TEST_LLM_MODEL > default
    model = os.environ.get(
        f"{provider_upper}_MODEL",
        os.environ.get("TEST_LLM_MODEL", DEFAULT_MODELS.get(provider_name, "")),
    )

    # Get API key: provider-specific env var > TEST_LLM_API_KEY
    api_key_env = API_KEY_ENV_VARS.get(provider_name)
    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        api_key = os.environ.get("TEST_LLM_API_KEY")

    # Get API base: provider-specific env var > TEST_LLM_API_BASE > default
    api_base = os.environ.get(
        f"{provider_upper}_API_BASE",
        os.environ.get("TEST_LLM_API_BASE", DEFAULT_API_BASES.get(provider_name)),
    )

    return LLMConfig(
        provider=provider_name,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "ollama: tests requiring Ollama (local LLM server)"
    )
    config.addinivalue_line("markers", "openai: tests requiring OpenAI API access")
    config.addinivalue_line(
        "markers", "anthropic: tests requiring Anthropic API access"
    )
    config.addinivalue_line(
        "markers", "gemini: tests requiring Google Gemini API access"
    )
    config.addinivalue_line(
        "markers", "github_copilot: tests requiring GitHub Copilot (OAuth device flow)"
    )
    config.addinivalue_line(
        "markers", "github: tests requiring GitHub Models API access"
    )
    config.addinivalue_line("markers", "groq: tests requiring Groq API access")
    config.addinivalue_line("markers", "mistral: tests requiring Mistral API access")
    config.addinivalue_line("markers", "deepseek: tests requiring DeepSeek API access")
    config.addinivalue_line(
        "markers", "openrouter: tests requiring OpenRouter API access"
    )
    config.addinivalue_line("markers", "provider: tests for a specific LLM provider")


# All provider markers that should trigger availability checks
_PROVIDER_MARKERS: frozenset[str] = frozenset(
    {
        "ollama",
        "openai",
        "anthropic",
        "gemini",
        "github_copilot",
        "github",
        "groq",
        "mistral",
        "deepseek",
        "openrouter",
    }
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests for unavailable providers."""
    for item in items:
        # Check for provider-specific markers
        for marker in item.iter_markers():
            if marker.name in _PROVIDER_MARKERS:
                provider_config = get_llm_config(marker.name)
                if not provider_config.is_available():
                    skip_marker = pytest.mark.skip(
                        reason=f"{marker.name} not available "
                        f"(missing API key or server not running)"
                    )
                    item.add_marker(skip_marker)
                break


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def llm_config() -> LLMConfig:
    """Get the default LLM configuration from environment."""
    return get_llm_config()


@pytest.fixture
def llm_negotiator_factory(llm_config: LLMConfig):
    """Factory fixture to create LLM negotiators with the configured provider."""

    def _create(**kwargs: Any) -> LLMNegotiator:
        negotiator_kwargs = llm_config.to_negotiator_kwargs()
        negotiator_kwargs.update(kwargs)
        return LLMNegotiator(**negotiator_kwargs)

    return _create


@pytest.fixture
def ollama_config() -> LLMConfig:
    """Get Ollama-specific configuration."""
    return get_llm_config("ollama")


@pytest.fixture
def openai_config() -> LLMConfig:
    """Get OpenAI-specific configuration."""
    return get_llm_config("openai")


@pytest.fixture
def anthropic_config() -> LLMConfig:
    """Get Anthropic-specific configuration."""
    return get_llm_config("anthropic")


@pytest.fixture
def gemini_config() -> LLMConfig:
    """Get Google Gemini-specific configuration."""
    return get_llm_config("gemini")


@pytest.fixture
def github_copilot_config() -> LLMConfig:
    """Get GitHub Copilot-specific configuration.

    GitHub Copilot uses OAuth device flow authentication.
    On first use, you'll be prompted to authenticate via GitHub.
    """
    return get_llm_config("github_copilot")


@pytest.fixture
def github_config() -> LLMConfig:
    """Get GitHub Models-specific configuration.

    Uses GITHUB_API_KEY for authentication.
    """
    return get_llm_config("github")


@pytest.fixture
def groq_config() -> LLMConfig:
    """Get Groq-specific configuration."""
    return get_llm_config("groq")


@pytest.fixture
def mistral_config() -> LLMConfig:
    """Get Mistral-specific configuration."""
    return get_llm_config("mistral")


@pytest.fixture
def deepseek_config() -> LLMConfig:
    """Get DeepSeek-specific configuration."""
    return get_llm_config("deepseek")


@pytest.fixture
def openrouter_config() -> LLMConfig:
    """Get OpenRouter-specific configuration."""
    return get_llm_config("openrouter")

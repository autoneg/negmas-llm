"""Tests for different LLM providers.

These tests verify that each LLM provider works correctly with negmas-llm.
Tests are automatically skipped if the required API keys or servers are not available.

To run tests for a specific provider:
    pytest tests/test_providers.py -m openai
    pytest tests/test_providers.py -m copilot
    pytest tests/test_providers.py -m anthropic

To run all provider tests (skips unavailable providers):
    pytest tests/test_providers.py
"""

from __future__ import annotations

import pytest
from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import SAOMechanism

from negmas_llm import LLMNegotiator

from .conftest import LLMConfig


@pytest.fixture
def simple_scenario():
    """Create a simple negotiation scenario for provider tests."""
    issues = [
        make_issue(name="price", values=(0, 100)),
        make_issue(name="quantity", values=(1, 10)),
    ]
    outcome_space = make_os(issues)
    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(outcome_space, reserved_value=0.0)
    return outcome_space, ufun1, ufun2


def run_negotiation_test(
    config: LLMConfig,
    simple_scenario: tuple,
    n_steps: int = 5,
) -> None:
    """Run a basic negotiation test with the given LLM config."""
    outcome_space, ufun1, ufun2 = simple_scenario

    kwargs = config.to_negotiator_kwargs()

    negotiator1 = LLMNegotiator(
        **kwargs,
        name="negotiator_1",
        ufun=ufun1,
    )
    negotiator2 = LLMNegotiator(
        **kwargs,
        name="negotiator_2",
        ufun=ufun2,
    )

    mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=n_steps)
    mechanism.add(negotiator1)
    mechanism.add(negotiator2)

    result = mechanism.run()

    assert result is not None
    assert result.started
    # Either agreement reached or negotiation ended normally
    assert result.agreement is not None or result.broken or result.timedout


# =============================================================================
# Ollama Tests (Local)
# =============================================================================


class TestOllamaProvider:
    """Tests for Ollama (local LLM server)."""

    @pytest.mark.slow
    @pytest.mark.ollama
    def test_ollama_negotiation(self, ollama_config: LLMConfig, simple_scenario):
        """Test negotiation with Ollama."""
        run_negotiation_test(ollama_config, simple_scenario)

    @pytest.mark.ollama
    def test_ollama_config(self, ollama_config: LLMConfig):
        """Test that Ollama configuration is correct."""
        assert ollama_config.provider == "ollama"
        assert ollama_config.model  # Should have a model
        assert ollama_config.api_base  # Should have API base


# =============================================================================
# GitHub Copilot Tests
# =============================================================================


class TestGitHubCopilotProvider:
    """Tests for GitHub Copilot.

    GitHub Copilot uses OAuth device flow for authentication.
    On first use, litellm will prompt you to authenticate via GitHub:
    1. A device code and verification URL will be displayed
    2. Visit the URL and enter the code to authenticate
    3. Credentials are stored locally for future use

    No API key or environment variable is required - just run the tests
    and follow the OAuth prompts.
    """

    @pytest.mark.slow
    @pytest.mark.github_copilot
    def test_github_copilot_negotiation(
        self, github_copilot_config: LLMConfig, simple_scenario
    ):
        """Test negotiation with GitHub Copilot."""
        run_negotiation_test(github_copilot_config, simple_scenario)

    @pytest.mark.github_copilot
    def test_github_copilot_config(self, github_copilot_config: LLMConfig):
        """Test that GitHub Copilot configuration is correct."""
        assert github_copilot_config.provider == "github_copilot"
        assert github_copilot_config.model  # Should have a model


# =============================================================================
# GitHub Models Tests
# =============================================================================


class TestGitHubModelsProvider:
    """Tests for GitHub Models (marketplace).

    Uses GITHUB_API_KEY environment variable (Personal Access Token works).
    Access models like Llama, Phi, etc. from GitHub's model marketplace.
    """

    @pytest.mark.slow
    @pytest.mark.github
    def test_github_models_negotiation(self, github_config: LLMConfig, simple_scenario):
        """Test negotiation with GitHub Models."""
        run_negotiation_test(github_config, simple_scenario)

    @pytest.mark.github
    def test_github_models_config(self, github_config: LLMConfig):
        """Test that GitHub Models configuration is correct."""
        assert github_config.provider == "github"
        assert github_config.model


# =============================================================================
# OpenAI Tests
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAI.

    Requires OPENAI_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.openai
    def test_openai_negotiation(self, openai_config: LLMConfig, simple_scenario):
        """Test negotiation with OpenAI."""
        run_negotiation_test(openai_config, simple_scenario)

    @pytest.mark.openai
    def test_openai_config(self, openai_config: LLMConfig):
        """Test that OpenAI configuration is correct."""
        assert openai_config.provider == "openai"
        assert openai_config.model


# =============================================================================
# Anthropic Tests
# =============================================================================


class TestAnthropicProvider:
    """Tests for Anthropic Claude.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.anthropic
    def test_anthropic_negotiation(self, anthropic_config: LLMConfig, simple_scenario):
        """Test negotiation with Anthropic Claude."""
        run_negotiation_test(anthropic_config, simple_scenario)

    @pytest.mark.anthropic
    def test_anthropic_config(self, anthropic_config: LLMConfig):
        """Test that Anthropic configuration is correct."""
        assert anthropic_config.provider == "anthropic"
        assert anthropic_config.model


# =============================================================================
# Google Gemini Tests
# =============================================================================


class TestGeminiProvider:
    """Tests for Google Gemini.

    Requires GOOGLE_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.gemini
    def test_gemini_negotiation(self, gemini_config: LLMConfig, simple_scenario):
        """Test negotiation with Google Gemini."""
        run_negotiation_test(gemini_config, simple_scenario)

    @pytest.mark.gemini
    def test_gemini_config(self, gemini_config: LLMConfig):
        """Test that Gemini configuration is correct."""
        assert gemini_config.provider == "gemini"
        assert gemini_config.model


# =============================================================================
# Groq Tests
# =============================================================================


class TestGroqProvider:
    """Tests for Groq.

    Requires GROQ_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.groq
    def test_groq_negotiation(self, groq_config: LLMConfig, simple_scenario):
        """Test negotiation with Groq."""
        run_negotiation_test(groq_config, simple_scenario)

    @pytest.mark.groq
    def test_groq_config(self, groq_config: LLMConfig):
        """Test that Groq configuration is correct."""
        assert groq_config.provider == "groq"
        assert groq_config.model


# =============================================================================
# Mistral Tests
# =============================================================================


class TestMistralProvider:
    """Tests for Mistral AI.

    Requires MISTRAL_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.mistral
    def test_mistral_negotiation(self, mistral_config: LLMConfig, simple_scenario):
        """Test negotiation with Mistral."""
        run_negotiation_test(mistral_config, simple_scenario)

    @pytest.mark.mistral
    def test_mistral_config(self, mistral_config: LLMConfig):
        """Test that Mistral configuration is correct."""
        assert mistral_config.provider == "mistral"
        assert mistral_config.model


# =============================================================================
# DeepSeek Tests
# =============================================================================


class TestDeepSeekProvider:
    """Tests for DeepSeek.

    Requires DEEPSEEK_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.deepseek
    def test_deepseek_negotiation(self, deepseek_config: LLMConfig, simple_scenario):
        """Test negotiation with DeepSeek."""
        run_negotiation_test(deepseek_config, simple_scenario)

    @pytest.mark.deepseek
    def test_deepseek_config(self, deepseek_config: LLMConfig):
        """Test that DeepSeek configuration is correct."""
        assert deepseek_config.provider == "deepseek"
        assert deepseek_config.model


# =============================================================================
# OpenRouter Tests
# =============================================================================


class TestOpenRouterProvider:
    """Tests for OpenRouter.

    Requires OPENROUTER_API_KEY environment variable.
    """

    @pytest.mark.slow
    @pytest.mark.openrouter
    def test_openrouter_negotiation(
        self, openrouter_config: LLMConfig, simple_scenario
    ):
        """Test negotiation with OpenRouter."""
        run_negotiation_test(openrouter_config, simple_scenario)

    @pytest.mark.openrouter
    def test_openrouter_config(self, openrouter_config: LLMConfig):
        """Test that OpenRouter configuration is correct."""
        assert openrouter_config.provider == "openrouter"
        assert openrouter_config.model


# =============================================================================
# Generic Provider Test (uses TEST_LLM_PROVIDER)
# =============================================================================


class TestConfiguredProvider:
    """Tests using the provider configured via TEST_LLM_PROVIDER."""

    @pytest.mark.slow
    def test_configured_provider_negotiation(
        self, llm_config: LLMConfig, simple_scenario
    ):
        """Test negotiation with the configured LLM provider."""
        if not llm_config.is_available():
            pytest.skip(
                f"Provider {llm_config.provider} not available "
                "(missing API key or server not running)"
            )
        run_negotiation_test(llm_config, simple_scenario)

    def test_configured_provider_info(self, llm_config: LLMConfig):
        """Display information about the configured provider."""
        print(f"\nConfigured provider: {llm_config.provider}")
        print(f"Model: {llm_config.model}")
        print(f"API base: {llm_config.api_base or 'default'}")
        print(f"Available: {llm_config.is_available()}")
        assert llm_config.provider  # Should always have a provider

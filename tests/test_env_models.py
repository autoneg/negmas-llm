"""Tests for environment variable model selection.

These tests verify that DEFAULT_MODELS correctly reads from environment variables
and that negotiators use the correct models based on environment configuration.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest


class TestDefaultModelsEnvironmentVariables:
    """Test that DEFAULT_MODELS reads from environment variables correctly."""

    def test_default_fallback_values(self):
        """Test that default fallback values are used when env vars are not set."""
        # Import fresh to get current values
        from negmas_llm.common import DEFAULT_MODELS

        # These should be the fallback values (assuming no env vars set)
        # We check that the dict has expected keys
        expected_providers = [
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
            "together_ai",
            "cohere",
            "huggingface",
        ]
        for provider in expected_providers:
            assert provider in DEFAULT_MODELS, f"Missing provider: {provider}"
            assert isinstance(DEFAULT_MODELS[provider], str)
            assert len(DEFAULT_MODELS[provider]) > 0

    def test_negmas_llm_env_var_override(self):
        """Test that NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL overrides default."""
        test_model = "test-custom-model-12345"

        # Test for multiple providers
        providers_to_test = ["ollama", "openai", "anthropic", "gemini"]

        for provider in providers_to_test:
            env_var = f"NEGMAS_LLM_{provider.upper()}_DEFAULT_MODEL"

            # Set the environment variable and reload the module
            with patch.dict(os.environ, {env_var: test_model}):
                # Need to reload the module to pick up new env var
                import negmas_llm.common

                importlib.reload(negmas_llm.common)
                from negmas_llm.common import DEFAULT_MODELS

                assert DEFAULT_MODELS[provider] == test_model, (
                    f"Provider {provider} should use model from {env_var}"
                )

            # Reload again to reset
            importlib.reload(negmas_llm.common)

    def test_ollama_legacy_env_var_support(self):
        """Test that legacy OLLAMA_MODEL env var is still supported."""
        test_model = "legacy-ollama-model"

        # Clear the new env var if set, only set legacy
        env_updates = {
            "OLLAMA_MODEL": test_model,
        }
        # Make sure new var is not set
        env_to_clear = "NEGMAS_LLM_OLLAMA_DEFAULT_MODEL"

        with patch.dict(os.environ, env_updates):
            # Remove the new var if present
            if env_to_clear in os.environ:
                del os.environ[env_to_clear]

            import negmas_llm.common

            importlib.reload(negmas_llm.common)
            from negmas_llm.common import DEFAULT_MODELS

            assert DEFAULT_MODELS["ollama"] == test_model, (
                "Legacy OLLAMA_MODEL should be supported"
            )

        # Reset
        importlib.reload(negmas_llm.common)

    def test_new_env_var_takes_precedence_over_legacy(self):
        """Test that new env var takes precedence over OLLAMA_MODEL."""
        new_model = "new-env-var-model"
        legacy_model = "legacy-env-var-model"

        with patch.dict(
            os.environ,
            {
                "NEGMAS_LLM_OLLAMA_DEFAULT_MODEL": new_model,
                "OLLAMA_MODEL": legacy_model,
            },
        ):
            import negmas_llm.common

            importlib.reload(negmas_llm.common)
            from negmas_llm.common import DEFAULT_MODELS

            assert DEFAULT_MODELS["ollama"] == new_model, (
                "New env var should take precedence over legacy OLLAMA_MODEL"
            )

        # Reset
        importlib.reload(negmas_llm.common)


class TestLLMNegotiatorUsesEnvModel:
    """Test that LLMNegotiator uses the correct model from environment."""

    def test_ollama_negotiator_uses_env_model(self):
        """Test that OllamaNegotiator uses model from environment variable."""
        test_model = "env-ollama-model"

        with patch.dict(os.environ, {"NEGMAS_LLM_OLLAMA_DEFAULT_MODEL": test_model}):
            # Reload modules to pick up new env var
            import negmas_llm.common
            import negmas_llm.negotiator

            importlib.reload(negmas_llm.common)
            importlib.reload(negmas_llm.negotiator)
            from negmas_llm.negotiator import OllamaNegotiator

            # Create negotiator without specifying model
            negotiator = OllamaNegotiator(name="test")

            assert negotiator.model == test_model, (
                f"OllamaNegotiator should use model from env: {test_model}"
            )

        # Reset modules
        importlib.reload(negmas_llm.common)
        importlib.reload(negmas_llm.negotiator)

    def test_explicit_model_overrides_env(self):
        """Test that explicitly passed model overrides environment variable."""
        env_model = "env-model"
        explicit_model = "explicit-model"

        with patch.dict(os.environ, {"NEGMAS_LLM_OLLAMA_DEFAULT_MODEL": env_model}):
            import negmas_llm.common
            import negmas_llm.negotiator

            importlib.reload(negmas_llm.common)
            importlib.reload(negmas_llm.negotiator)
            from negmas_llm.negotiator import OllamaNegotiator

            # Create negotiator with explicit model
            negotiator = OllamaNegotiator(model=explicit_model, name="test")

            assert negotiator.model == explicit_model, (
                "Explicit model should override env var"
            )

        # Reset
        importlib.reload(negmas_llm.common)
        importlib.reload(negmas_llm.negotiator)


class TestLLMMetaNegotiatorUsesEnvModel:
    """Test that LLMMetaNegotiator uses the correct model from environment."""

    def test_meta_negotiator_uses_env_model(self):
        """Test that LLMMetaNegotiator uses model from environment variable."""
        from negmas_llm import is_meta_negotiator_available

        if not is_meta_negotiator_available():
            pytest.skip("SAOMetaNegotiator not available")

        test_model = "env-meta-model"

        with patch.dict(os.environ, {"NEGMAS_LLM_OLLAMA_DEFAULT_MODEL": test_model}):
            # Reload modules to pick up new env var
            import negmas_llm.common
            import negmas_llm.meta

            importlib.reload(negmas_llm.common)
            importlib.reload(negmas_llm.meta)
            from negmas_llm.meta import LLMBoulwareTBNegotiator

            # Create meta negotiator without specifying model
            negotiator = LLMBoulwareTBNegotiator(name="test")

            assert negotiator.model == test_model, (
                f"LLMBoulwareTBNegotiator should use model from env: {test_model}"
            )

        # Reset modules
        importlib.reload(negmas_llm.common)
        importlib.reload(negmas_llm.meta)

    def test_meta_negotiator_explicit_model_overrides_env(self):
        """Test that explicitly passed model overrides environment variable."""
        from negmas_llm import is_meta_negotiator_available

        if not is_meta_negotiator_available():
            pytest.skip("SAOMetaNegotiator not available")

        env_model = "env-model"
        explicit_model = "explicit-model"

        with patch.dict(os.environ, {"NEGMAS_LLM_OLLAMA_DEFAULT_MODEL": env_model}):
            import negmas_llm.common
            import negmas_llm.meta

            importlib.reload(negmas_llm.common)
            importlib.reload(negmas_llm.meta)
            from negmas_llm.meta import LLMLinearTBNegotiator

            # Create meta negotiator with explicit model
            negotiator = LLMLinearTBNegotiator(model=explicit_model, name="test")

            assert negotiator.model == explicit_model, (
                "Explicit model should override env var"
            )

        # Reset
        importlib.reload(negmas_llm.common)
        importlib.reload(negmas_llm.meta)


class TestMultipleProvidersEnvVars:
    """Test environment variables for multiple providers."""

    def test_all_providers_have_env_var_support(self):
        """Test that all providers support environment variable configuration."""
        from negmas_llm.common import _get_default_model

        providers = [
            ("ollama", "test-ollama"),
            ("openai", "test-openai"),
            ("anthropic", "test-anthropic"),
            ("gemini", "test-gemini"),
            ("github_copilot", "test-github-copilot"),
            ("github", "test-github"),
            ("groq", "test-groq"),
            ("mistral", "test-mistral"),
            ("deepseek", "test-deepseek"),
            ("openrouter", "test-openrouter"),
            ("together_ai", "test-together-ai"),
            ("cohere", "test-cohere"),
            ("huggingface", "test-huggingface"),
        ]

        for provider, test_model in providers:
            env_var = f"NEGMAS_LLM_{provider.upper()}_DEFAULT_MODEL"
            with patch.dict(os.environ, {env_var: test_model}):
                result = _get_default_model(provider, "fallback")
                assert result == test_model, (
                    f"Provider {provider} should read from {env_var}"
                )

    def test_get_default_model_fallback(self):
        """Test that _get_default_model returns fallback when env var not set."""
        from negmas_llm.common import _get_default_model

        fallback = "my-fallback-model"
        # Use a provider name that definitely has no env var set
        result = _get_default_model("nonexistent_provider", fallback)
        assert result == fallback

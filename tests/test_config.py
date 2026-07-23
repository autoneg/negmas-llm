"""Tests for the single-source-of-truth LLM provider/model resolution.

Covers :mod:`negmas_llm.config` and its integration into the negotiator, meta
negotiator, and component families:

* the three configuration axes (per-provider, global, per-negotiator-type),
* the precedence rules (explicit > per-type > global > default),
* the provider/model coherence rules,
* backward-compatible defaults (no env => historical values).
"""

from __future__ import annotations

import pytest

from negmas_llm import (
    AnthropicNegotiator,
    OpenAINegotiator,
    effective_llm_config,
    per_type_env_var,
    resolve_llm_config,
)
from negmas_llm.components import (
    LLMNegotiationSupporter,
    LLMValidator,
    OpenAIOfferingPolicy,
)
from negmas_llm.config import DEFAULT_PROVIDER
from negmas_llm.meta import LLMBoulwareTBNegotiator, LLMConcederTBNegotiator

# Every environment variable any test here might set. Cleared before each test
# so a developer's shell configuration cannot leak into assertions.
_MANAGED_ENV = [
    "NEGMAS_LLM_PROVIDER",
    "NEGMAS_LLM_MODEL",
    "NEGMAS_LLM_TEMPERATURE",
    "NEGMAS_LLM_MAX_TOKENS",
    "NEGMAS_LLM_TIMEOUT",
    "NEGMAS_LLM_NUM_RETRIES",
    "NEGMAS_LLM_API_KEY",
    "NEGMAS_LLM_API_BASE",
    "NEGMAS_LLM_DEFAULT_MAX_TOKENS",
    "NEGMAS_LLM_DEFAULT_TEMPERATURE",
    "NEGMAS_LLM_OPENAI_DEFAULT_MODEL",
    "NEGMAS_LLM_ANTHROPIC_DEFAULT_MODEL",
    "NEGMAS_LLM_OLLAMA_DEFAULT_MODEL",
    "OLLAMA_MODEL",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_PROVIDER",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_TEMPERATURE",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_MAX_TOKENS",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_TIMEOUT",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_NUM_RETRIES",
    "NEGMAS_LLM_OpenAINegotiator_PROVIDER",
    "NEGMAS_LLM_OpenAINegotiator_MODEL",
    "NEGMAS_LLM_OpenAIOfferingPolicy_MODEL",
    # Model-type (tier) variables
    "NEGMAS_LLM_EFFORT",
    "NEGMAS_LLM_PROVIDER_FAST",
    "NEGMAS_LLM_MODEL_FAST",
    "NEGMAS_LLM_EFFORT_FAST",
    "NEGMAS_LLM_PROVIDER_ACCURATE",
    "NEGMAS_LLM_MODEL_ACCURATE",
    "NEGMAS_LLM_EFFORT_ACCURATE",
    "NEGMAS_LLM_N_MODEL_FAST",
    "NEGMAS_LLM_N_EFFORT_FAST",
    "NEGMAS_LLM_N_PROVIDER_FAST",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL_FAST",
    "NEGMAS_LLM_LLMBoulwareTBNegotiator_PROVIDER_FAST",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _MANAGED_ENV:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# resolve_llm_config unit behavior
# =============================================================================


class TestResolveProvider:
    def test_default_provider_when_nothing_set(self):
        cfg = resolve_llm_config("SomeNegotiator")
        assert cfg.provider == DEFAULT_PROVIDER == "ollama"

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "groq")
        cfg = resolve_llm_config("SomeNegotiator", provider="mistral")
        assert cfg.provider == "mistral"

    def test_per_type_over_global(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "groq")
        monkeypatch.setenv("NEGMAS_LLM_SomeNegotiator_PROVIDER", "cohere")
        cfg = resolve_llm_config("SomeNegotiator")
        assert cfg.provider == "cohere"

    def test_global_provider(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "groq")
        assert resolve_llm_config("SomeNegotiator").provider == "groq"

    def test_lock_provider_ignores_global(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "groq")
        cfg = resolve_llm_config(
            "OpenAINegotiator", default_provider="openai", lock_provider=True
        )
        assert cfg.provider == "openai"

    def test_lock_provider_still_honors_per_type(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OpenAINegotiator_PROVIDER", "azure")
        cfg = resolve_llm_config(
            "OpenAINegotiator", default_provider="openai", lock_provider=True
        )
        assert cfg.provider == "azure"


class TestResolveModel:
    def test_explicit_wins(self):
        cfg = resolve_llm_config("N", provider="openai", model="my-model")
        assert cfg.model == "my-model"

    def test_class_default_model(self):
        cfg = resolve_llm_config("N", provider="openai", default_model="gpt-4o")
        assert cfg.model == "gpt-4o"

    def test_per_provider_env_over_class_default(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OPENAI_DEFAULT_MODEL", "gpt-4-turbo")
        cfg = resolve_llm_config("N", provider="openai", default_model="gpt-4o")
        assert cfg.model == "gpt-4-turbo"

    def test_per_type_model_over_per_provider(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OPENAI_DEFAULT_MODEL", "gpt-4-turbo")
        monkeypatch.setenv("NEGMAS_LLM_N_MODEL", "special")
        cfg = resolve_llm_config("N", provider="openai", default_model="gpt-4o")
        assert cfg.model == "special"

    def test_global_model_applies_to_matching_provider(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        cfg = resolve_llm_config("N", default_model="unused")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"

    def test_global_model_not_applied_to_mismatched_provider(self, monkeypatch):
        # Global model targets the (default) global provider; a per-type provider
        # override must not receive it.
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("NEGMAS_LLM_N_PROVIDER", "anthropic")
        cfg = resolve_llm_config("N", default_model="fallback")
        assert cfg.provider == "anthropic"
        # Falls back to the class/provider default, NOT the global gpt-4o.
        assert cfg.model == "fallback"

    def test_legacy_ollama_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "legacy-llama")
        cfg = resolve_llm_config("N", provider="ollama")
        assert cfg.model == "legacy-llama"


class TestResolveOtherVars:
    def test_temperature_per_type_and_global(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_TEMPERATURE", "0.2")
        assert resolve_llm_config("N").temperature == 0.2
        monkeypatch.setenv("NEGMAS_LLM_N_TEMPERATURE", "0.9")
        assert resolve_llm_config("N").temperature == 0.9

    def test_max_tokens_per_type(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_N_MAX_TOKENS", "4096")
        assert resolve_llm_config("N").max_tokens == 4096

    def test_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_TIMEOUT", "30")
        monkeypatch.setenv("NEGMAS_LLM_NUM_RETRIES", "3")
        cfg = resolve_llm_config("N")
        assert cfg.timeout == 30.0
        assert cfg.num_retries == 3

    def test_explicit_beats_env_for_scalars(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_TIMEOUT", "30")
        assert resolve_llm_config("N", timeout=5).timeout == 5

    def test_unset_scalars_stay_none(self):
        cfg = resolve_llm_config("N")
        assert cfg.temperature is None
        assert cfg.max_tokens is None
        assert cfg.timeout is None
        assert cfg.num_retries is None


def test_per_type_env_var_format():
    assert (
        per_type_env_var("OpenAINegotiator", "model")
        == "NEGMAS_LLM_OpenAINegotiator_MODEL"
    )


# =============================================================================
# Integration with real negotiator / meta / component classes
# =============================================================================


class TestBackwardCompatibleDefaults:
    def test_openai_default_unchanged(self):
        n = OpenAINegotiator(name="t")
        assert n.provider == "openai"
        assert n.model == "gpt-4o"

    def test_anthropic_default_unchanged(self):
        n = AnthropicNegotiator(name="t")
        assert n.provider == "anthropic"
        assert n.model == "claude-sonnet-4-20250514"

    def test_meta_default_unchanged(self):
        n = LLMBoulwareTBNegotiator(name="t")
        assert n.provider == "ollama"

    def test_component_default_unchanged(self):
        assert OpenAIOfferingPolicy().model == "gpt-4o"

    def test_generic_components_resolve_via_post_init(self):
        # Supporter/Validator sit on GBComponent (a different base than the
        # policies); confirm __attrs_post_init__ still resolves their defaults
        # rather than leaving provider/model as the empty-string sentinel.
        for component in (LLMValidator(), LLMNegotiationSupporter()):
            assert component.provider == "ollama"
            assert component.model == "qwen3:4b-instruct"

    def test_generic_component_respects_global_env(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        v = LLMValidator()
        assert (v.provider, v.model) == ("openai", "gpt-4o")


class TestExperimentScenario:
    """The user's headline scenario: everyone on model A, one on model B from a
    different provider."""

    def test_global_default_plus_one_override(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("NEGMAS_LLM_LLMBoulwareTBNegotiator_PROVIDER", "anthropic")
        monkeypatch.setenv("NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL", "claude-3-opus")

        boulware = LLMBoulwareTBNegotiator(name="t")
        conceder = LLMConcederTBNegotiator(name="t")

        assert (boulware.provider, boulware.model) == ("anthropic", "claude-3-opus")
        assert (conceder.provider, conceder.model) == ("openai", "gpt-4o")

    def test_provider_locked_class_ignores_global_provider(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "anthropic")
        n = OpenAINegotiator(name="t")
        assert n.provider == "openai"

    def test_per_type_model_override_on_locked_class(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OpenAINegotiator_MODEL", "gpt-4o-mini")
        n = OpenAINegotiator(name="t")
        assert n.provider == "openai"
        assert n.model == "gpt-4o-mini"

    def test_component_per_type_override(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OpenAIOfferingPolicy_MODEL", "gpt-4o-mini")
        assert OpenAIOfferingPolicy().model == "gpt-4o-mini"

    def test_explicit_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_OpenAINegotiator_MODEL", "gpt-4o-mini")
        n = OpenAINegotiator(model="gpt-4-turbo", name="t")
        assert n.model == "gpt-4-turbo"


# =============================================================================
# Model types (tiers) and effort
# =============================================================================


class TestEffort:
    def test_effort_unset_is_none(self):
        assert resolve_llm_config("N").effort is None

    def test_effort_global(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_EFFORT", "high")
        assert resolve_llm_config("N").effort == "high"

    def test_effort_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_EFFORT", "high")
        assert resolve_llm_config("N", effort="low").effort == "low"

    def test_effort_reaches_litellm_kwargs(self):
        from negmas_llm.common import apply_effort

        kwargs: dict = {}
        apply_effort(kwargs, "high")
        assert kwargs["reasoning_effort"] == "high"

        kwargs = {}
        apply_effort(kwargs, None)
        assert "reasoning_effort" not in kwargs

        # A value already present (e.g. via llm_kwargs) is not overwritten.
        kwargs = {"reasoning_effort": "medium"}
        apply_effort(kwargs, "high")
        assert kwargs["reasoning_effort"] == "medium"


class TestModelTypes:
    def test_env_var_name_has_suffix(self):
        assert (
            per_type_env_var("OpenAINegotiator", "model", "fast")
            == "NEGMAS_LLM_OpenAINegotiator_MODEL_FAST"
        )

    def test_type_name_is_uppercased(self):
        assert per_type_env_var("N", "model", "Friendly").endswith("_MODEL_FRIENDLY")

    def test_none_type_reproduces_base(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        assert (
            resolve_llm_config("N").model
            == resolve_llm_config("N", model_type=None).model
        )

    def test_global_tier_bundle(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER_FAST", "groq")
        monkeypatch.setenv("NEGMAS_LLM_MODEL_FAST", "llama-3.1-8b")
        monkeypatch.setenv("NEGMAS_LLM_EFFORT_FAST", "low")
        cfg = resolve_llm_config("N", model_type="fast")
        assert (cfg.provider, cfg.model, cfg.effort) == ("groq", "llama-3.1-8b", "low")

    def test_different_parts_different_tiers(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER_FAST", "groq")
        monkeypatch.setenv("NEGMAS_LLM_MODEL_FAST", "llama-3.1-8b")
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER_ACCURATE", "openai")
        monkeypatch.setenv("NEGMAS_LLM_MODEL_ACCURATE", "o3")
        fast = resolve_llm_config("N", model_type="fast")
        accurate = resolve_llm_config("N", model_type="accurate")
        assert (fast.provider, fast.model) == ("groq", "llama-3.1-8b")
        assert (accurate.provider, accurate.model) == ("openai", "o3")

    def test_tier_falls_back_to_base_when_unset(self, monkeypatch):
        # No fast-specific vars -> a fast request behaves like the base config.
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NEGMAS_LLM_MODEL", "gpt-4o")
        cfg = resolve_llm_config("N", model_type="fast")
        assert (cfg.provider, cfg.model) == ("openai", "gpt-4o")

    def test_scope_dominant_per_type_base_beats_global_tier(self, monkeypatch):
        # The tie the design resolves: per-type base outranks global tier.
        monkeypatch.setenv("NEGMAS_LLM_N_MODEL", "per-type-base")
        monkeypatch.setenv("NEGMAS_LLM_MODEL_FAST", "global-tier")
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("NEGMAS_LLM_N_PROVIDER", "openai")
        cfg = resolve_llm_config("N", model_type="fast", default_provider="openai")
        assert cfg.model == "per-type-base"

    def test_per_type_tier_is_most_specific(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_N_MODEL", "per-type-base")
        monkeypatch.setenv("NEGMAS_LLM_N_MODEL_FAST", "per-type-fast")
        cfg = resolve_llm_config("N", model_type="fast", provider="openai")
        assert cfg.model == "per-type-fast"

    def test_tier_global_model_guarded_by_provider(self, monkeypatch):
        # global MODEL_FAST targets PROVIDER_FAST/PROVIDER; a per-type provider
        # override to a different provider must not receive the fast model.
        monkeypatch.setenv("NEGMAS_LLM_MODEL_FAST", "groq-only-model")
        monkeypatch.setenv("NEGMAS_LLM_N_PROVIDER", "anthropic")
        cfg = resolve_llm_config("N", model_type="fast", default_model="fallback")
        assert cfg.provider == "anthropic"
        assert cfg.model == "fallback"

    def test_lock_provider_ignores_tier_global_provider(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_PROVIDER_FAST", "groq")
        cfg = resolve_llm_config(
            "OpenAINegotiator",
            model_type="fast",
            default_provider="openai",
            lock_provider=True,
        )
        assert cfg.provider == "openai"


class TestEffectiveLLMConfig:
    def test_none_type_uses_instance_settings(self):
        n = OpenAINegotiator(model="gpt-4o", name="t")
        cfg = effective_llm_config(n, None)
        assert (cfg.provider, cfg.model) == ("openai", "gpt-4o")

    def test_tier_reresolves_for_instance(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_LLMBoulwareTBNegotiator_PROVIDER_FAST", "groq")
        monkeypatch.setenv(
            "NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL_FAST", "llama-3.1-8b"
        )
        n = LLMBoulwareTBNegotiator(name="t")  # base: ollama
        assert n.provider == "ollama"
        cfg = effective_llm_config(n, "fast")
        assert (cfg.provider, cfg.model) == ("groq", "llama-3.1-8b")

    def test_tier_inherits_instance_settings_when_tier_unset(self):
        # A tier call with no tier env must keep the instance's explicit
        # construction-time settings (not fall back to the class default).
        n = LLMBoulwareTBNegotiator(provider="openai", model="gpt-4o", name="t")
        cfg = effective_llm_config(n, "fast")
        assert (cfg.provider, cfg.model) == ("openai", "gpt-4o")

    def test_tier_preserves_instance_api_base(self):
        # vLLM carries api_base as a subclass default; a tier call must not drop
        # it (else the request would hit the real OpenAI endpoint).
        from negmas_llm import VLLMNegotiator

        n = VLLMNegotiator(model="my-model", name="t")
        assert n.api_base == "http://localhost:8000/v1"
        cfg = effective_llm_config(n, "fast")
        assert cfg.api_base == "http://localhost:8000/v1"
        assert cfg.model == "my-model"

    def test_tier_env_overrides_instance_setting(self, monkeypatch):
        monkeypatch.setenv("NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL_FAST", "tiered")
        n = LLMBoulwareTBNegotiator(provider="openai", model="gpt-4o", name="t")
        cfg = effective_llm_config(n, "fast")
        assert cfg.model == "tiered"

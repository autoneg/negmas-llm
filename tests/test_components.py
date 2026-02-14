"""Tests for LLM-based components."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from negmas import make_issue, make_os
from negmas.gb import GBState, ResponseType
from negmas.gb.components.acceptance import AcceptAnyRational
from negmas.gb.components.offering import RandomOfferingPolicy
from negmas.gb.negotiators.modular import MAPNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, SAOMechanism

from negmas_llm import (
    AnthropicAcceptancePolicy,
    AnthropicOfferingPolicy,
    LLMAcceptancePolicy,
    LLMNegotiationSupporter,
    LLMOfferingPolicy,
    LLMValidator,
    OllamaAcceptancePolicy,
    OllamaOfferingPolicy,
    OpenAIAcceptancePolicy,
    OpenAIOfferingPolicy,
)

# Use environment variable for model, defaulting to qwen3:0.6b (small/fast)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")


@pytest.fixture
def simple_negotiation_setup():
    """Create a simple negotiation scenario with 2 issues."""
    issues = [
        make_issue(name="price", values=(0, 100)),
        make_issue(name="quantity", values=(1, 10)),
    ]
    outcome_space = make_os(issues)

    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(outcome_space, reserved_value=0.0)

    return outcome_space, ufun1, ufun2


@pytest.fixture
def discrete_negotiation_setup():
    """Create a negotiation scenario with discrete issues."""
    issues = [
        make_issue(name="delivery", values=["standard", "express", "overnight"]),
        make_issue(name="warranty", values=["none", "1year", "2year"]),
        make_issue(name="price", values=[100, 150, 200, 250, 300]),
    ]
    outcome_space = make_os(issues)

    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(outcome_space, reserved_value=0.0)

    return outcome_space, ufun1, ufun2


class TestLLMAcceptancePolicyUnit:
    """Unit tests for LLMAcceptancePolicy."""

    def test_initialization(self):
        """Test that LLMAcceptancePolicy initializes correctly."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        assert policy.provider == "openai"
        assert policy.model == "gpt-4o"
        assert policy.temperature == 0.7
        assert policy.max_tokens == 1024
        assert policy.api_key is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        policy = LLMAcceptancePolicy(
            provider="anthropic",
            model="claude-3-opus",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2048,
        )
        assert policy.provider == "anthropic"
        assert policy.model == "claude-3-opus"
        assert policy.api_key == "test-key"
        assert policy.temperature == 0.5
        assert policy.max_tokens == 2048

    def test_model_string(self):
        """Test that get_model_string returns correct format."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        assert policy.get_model_string() == "openai/gpt-4o"

    def test_format_response_instructions(self):
        """Test that response instructions are generated correctly."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        instructions = policy.format_response_instructions()
        assert "accept" in instructions.lower()
        assert "reject" in instructions.lower()
        assert "end" in instructions.lower()
        assert "JSON" in instructions

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_accept(self, mock_completion: MagicMock):
        """Test parsing an accept response."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        response = policy._parse_response(
            '{"decision": "accept", "reasoning": "good offer"}'
        )
        assert response == ResponseType.ACCEPT_OFFER

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_reject(self, mock_completion: MagicMock):
        """Test parsing a reject response."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        response = policy._parse_response(
            '{"decision": "reject", "reasoning": "too low"}'
        )
        assert response == ResponseType.REJECT_OFFER

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_end(self, mock_completion: MagicMock):
        """Test parsing an end response."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        response = policy._parse_response(
            '{"decision": "end", "reasoning": "no agreement possible"}'
        )
        assert response == ResponseType.END_NEGOTIATION

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_invalid_json(self, mock_completion: MagicMock):
        """Test parsing invalid JSON defaults to reject."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        response = policy._parse_response("not valid json")
        assert response == ResponseType.REJECT_OFFER


class TestLLMOfferingPolicyUnit:
    """Unit tests for LLMOfferingPolicy."""

    def test_initialization(self):
        """Test that LLMOfferingPolicy initializes correctly."""
        policy = LLMOfferingPolicy(
            provider="openai",
            model="gpt-4o",
        )
        assert policy.provider == "openai"
        assert policy.model == "gpt-4o"

    def test_model_string(self):
        """Test that get_model_string returns correct format."""
        policy = LLMOfferingPolicy(
            provider="anthropic",
            model="claude-3-opus",
        )
        assert policy.get_model_string() == "anthropic/claude-3-opus"

    def test_format_response_instructions(self):
        """Test that response instructions are generated correctly."""
        policy = LLMOfferingPolicy(
            provider="openai",
            model="gpt-4o",
        )
        instructions = policy.format_response_instructions()
        assert "outcome" in instructions.lower()
        assert "JSON" in instructions

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_valid(self, mock_completion: MagicMock):
        """Test parsing a valid offer response."""
        policy = LLMOfferingPolicy(
            provider="openai",
            model="gpt-4o",
        )
        outcome, text = policy._parse_response(
            '{"outcome": [50, 5], "text": "Let\'s meet in the middle", '
            '"reasoning": "compromise"}'
        )
        assert outcome == (50, 5)
        assert text == "Let's meet in the middle"

    @patch("negmas_llm.components.litellm.completion")
    def test_parse_response_invalid_json(self, mock_completion: MagicMock):
        """Test parsing invalid JSON returns None."""
        policy = LLMOfferingPolicy(
            provider="openai",
            model="gpt-4o",
        )
        outcome, text = policy._parse_response("not valid json")
        assert outcome is None
        assert text is None


class TestLLMNegotiationSupporterUnit:
    """Unit tests for LLMNegotiationSupporter."""

    def test_initialization(self):
        """Test that LLMNegotiationSupporter initializes correctly."""
        supporter = LLMNegotiationSupporter(
            provider="openai",
            model="gpt-4o",
        )
        assert supporter.provider == "openai"
        assert supporter.model == "gpt-4o"
        assert supporter.last_text is None

    def test_build_system_prompt(self):
        """Test that system prompt is generated correctly."""
        supporter = LLMNegotiationSupporter(
            provider="openai",
            model="gpt-4o",
        )
        prompt = supporter.build_system_prompt()
        assert "negotiation" in prompt.lower()
        assert "text" in prompt.lower()


class TestLLMValidatorUnit:
    """Unit tests for LLMValidator."""

    def test_initialization(self):
        """Test that LLMValidator initializes correctly."""
        validator = LLMValidator(
            provider="openai",
            model="gpt-4o",
        )
        assert validator.provider == "openai"
        assert validator.model == "gpt-4o"
        assert validator.mode == "validate_only"

    def test_initialization_with_mode(self):
        """Test initialization with different modes."""
        validator = LLMValidator(
            provider="openai",
            model="gpt-4o",
            mode="action_wins",
        )
        assert validator.mode == "action_wins"

    def test_build_validation_prompt(self):
        """Test that validation prompt is generated correctly."""
        validator = LLMValidator(
            provider="openai",
            model="gpt-4o",
        )
        prompt = validator.build_validation_prompt()
        assert "consistent" in prompt.lower()
        assert "JSON" in prompt


class TestProviderConvenienceClasses:
    """Tests for provider-specific convenience classes."""

    def test_openai_acceptance_policy(self):
        """Test OpenAIAcceptancePolicy defaults."""
        policy = OpenAIAcceptancePolicy(model="gpt-4o-mini")
        assert policy.provider == "openai"
        assert policy.model == "gpt-4o-mini"

    def test_openai_acceptance_policy_default_model(self):
        """Test OpenAIAcceptancePolicy default model."""
        policy = OpenAIAcceptancePolicy()
        assert policy.provider == "openai"
        assert policy.model == "gpt-4o"

    def test_openai_offering_policy(self):
        """Test OpenAIOfferingPolicy defaults."""
        policy = OpenAIOfferingPolicy(model="gpt-4o-mini")
        assert policy.provider == "openai"
        assert policy.model == "gpt-4o-mini"

    def test_ollama_acceptance_policy(self):
        """Test OllamaAcceptancePolicy defaults."""
        policy = OllamaAcceptancePolicy(model="llama3.2")
        assert policy.provider == "ollama"
        assert policy.model == "llama3.2"
        assert policy.api_base == "http://localhost:11434"

    def test_ollama_offering_policy(self):
        """Test OllamaOfferingPolicy defaults."""
        policy = OllamaOfferingPolicy(model="llama3.2")
        assert policy.provider == "ollama"
        assert policy.model == "llama3.2"
        assert policy.api_base == "http://localhost:11434"

    def test_anthropic_acceptance_policy(self):
        """Test AnthropicAcceptancePolicy defaults."""
        policy = AnthropicAcceptancePolicy()
        assert policy.provider == "anthropic"
        assert "claude" in policy.model

    def test_anthropic_offering_policy(self):
        """Test AnthropicOfferingPolicy defaults."""
        policy = AnthropicOfferingPolicy()
        assert policy.provider == "anthropic"
        assert "claude" in policy.model


class TestComponentsWithMAPNegotiator:
    """Tests for using components with MAPNegotiator."""

    @pytest.mark.slow
    def test_llm_acceptance_with_random_offering(self, simple_negotiation_setup):
        """Test LLM acceptance policy with random offering policy."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Create MAPNegotiator with LLM acceptance and random offering
        negotiator1 = MAPNegotiator(
            acceptance=OllamaAcceptancePolicy(model=OLLAMA_MODEL),
            offering=RandomOfferingPolicy(),
            name="llm_acceptance_negotiator",
            ufun=ufun1,
        )
        negotiator2 = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_llm_offering_with_rational_acceptance(self, simple_negotiation_setup):
        """Test LLM offering policy with rational acceptance policy."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Create MAPNegotiator with rational acceptance and LLM offering
        negotiator1 = MAPNegotiator(
            acceptance=AcceptAnyRational(),
            offering=OllamaOfferingPolicy(model=OLLAMA_MODEL),
            name="llm_offering_negotiator",
            ufun=ufun1,
        )
        negotiator2 = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_full_llm_map_negotiator(self, simple_negotiation_setup):
        """Test MAPNegotiator with both LLM acceptance and offering."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        negotiator1 = MAPNegotiator(
            acceptance=OllamaAcceptancePolicy(model=OLLAMA_MODEL),
            offering=OllamaOfferingPolicy(model=OLLAMA_MODEL),
            name="full_llm_negotiator",
            ufun=ufun1,
        )
        negotiator2 = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_map_vs_map_llm(self, simple_negotiation_setup):
        """Test two MAPNegotiators with LLM components against each other."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        negotiator1 = MAPNegotiator(
            acceptance=OllamaAcceptancePolicy(model=OLLAMA_MODEL),
            offering=OllamaOfferingPolicy(model=OLLAMA_MODEL),
            name="llm_negotiator_1",
            ufun=ufun1,
        )
        negotiator2 = MAPNegotiator(
            acceptance=OllamaAcceptancePolicy(model=OLLAMA_MODEL),
            offering=OllamaOfferingPolicy(model=OLLAMA_MODEL),
            name="llm_negotiator_2",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started


class TestComponentMixinMethods:
    """Tests for LLMComponentMixin formatting methods."""

    def test_format_outcome_space(self, simple_negotiation_setup):
        """Test outcome space formatting."""
        outcome_space, ufun1, _ = simple_negotiation_setup

        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )

        # Create a mock negotiator with nmi
        mock_negotiator = MagicMock()
        mock_negotiator.nmi = MagicMock()
        mock_negotiator.nmi.outcome_space = outcome_space

        result = policy.format_outcome_space(mock_negotiator)
        assert "Outcome Space" in result
        assert "price" in result or "quantity" in result

    def test_format_outcome_space_none(self):
        """Test outcome space formatting with None negotiator."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        result = policy.format_outcome_space(None)
        assert result == ""

    def test_format_own_ufun(self, simple_negotiation_setup):
        """Test utility function formatting."""
        outcome_space, ufun1, _ = simple_negotiation_setup

        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )

        mock_negotiator = MagicMock()
        mock_negotiator.ufun = ufun1
        mock_negotiator.reserved_value = 0.0

        result = policy.format_own_ufun(mock_negotiator)
        assert "Utility Function" in result

    def test_format_own_ufun_none(self):
        """Test utility function formatting with no ufun."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )
        result = policy.format_own_ufun(None)
        assert "do NOT have a utility function" in result

    def test_format_state(self, simple_negotiation_setup):
        """Test state formatting."""
        policy = LLMAcceptancePolicy(
            provider="openai",
            model="gpt-4o",
        )

        state = GBState(
            step=5,
            relative_time=0.5,
            running=True,
            broken=False,
            timedout=False,
            waiting=False,
            agreement=None,
            started=True,
            has_error=False,
            error_details="",
        )

        result = policy.format_state(state, (50, 5), None)
        assert "Step" in result
        assert "5" in result
        assert "50.00%" in result

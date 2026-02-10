"""Tests for LLMNegotiator."""

from __future__ import annotations

import os

import pytest
from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, SAOMechanism

from negmas_llm import OllamaNegotiator

# Use environment variable for model, defaulting to qwen3:0.6b (small/fast)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")


@pytest.fixture
def simple_negotiation_setup():
    """Create a simple negotiation scenario with 2 issues."""
    # Define issues
    issues = [
        make_issue(name="price", values=(0, 100)),
        make_issue(name="quantity", values=(1, 10)),
    ]
    outcome_space = make_os(issues)

    # Create utility functions for two parties
    # Party 1 prefers high price, low quantity
    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)

    # Party 2 prefers low price, high quantity
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


class TestLLMNegotiatorVsLLMNegotiator:
    """Test LLMNegotiator against itself."""

    @pytest.mark.slow
    def test_llm_vs_llm_simple(self, simple_negotiation_setup):
        """Test two LLM negotiators against each other with simple issues."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Create two LLM negotiators
        negotiator1 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_negotiator_1",
            ufun=ufun1,
        )
        negotiator2 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_negotiator_2",
            ufun=ufun2,
        )

        # Create and run the mechanism
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        # Run the negotiation
        result = mechanism.run()

        # Check that the negotiation completed
        assert result is not None
        assert result.started
        # Either agreement reached or negotiation ended
        assert result.agreement is not None or result.broken or result.timedout

    @pytest.mark.slow
    def test_llm_vs_llm_discrete(self, discrete_negotiation_setup):
        """Test two LLM negotiators with discrete issues."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        negotiator1 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_buyer",
            ufun=ufun1,
        )
        negotiator2 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_seller",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_llm_vs_llm_with_known_opponent_ufun(self, simple_negotiation_setup):
        """Test LLM negotiators when they know each other's utility function."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Each negotiator knows the opponent's utility function
        negotiator1 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="informed_llm_1",
            ufun=ufun1,
            private_info={"opponent_ufun": ufun2},
        )
        negotiator2 = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="informed_llm_2",
            ufun=ufun2,
            private_info={"opponent_ufun": ufun1},
        )

        mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=10,
        )
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started


class TestLLMNegotiatorVsAspirationNegotiator:
    """Test LLMNegotiator against AspirationNegotiator."""

    @pytest.mark.slow
    def test_llm_vs_aspiration_simple(self, simple_negotiation_setup):
        """Test LLM negotiator against AspirationNegotiator with simple issues."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        llm_negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_negotiator",
            ufun=ufun1,
        )
        aspiration_negotiator = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=20,
        )
        mechanism.add(llm_negotiator)
        mechanism.add(aspiration_negotiator)

        result = mechanism.run()

        assert result is not None
        assert result.started
        # The negotiation should complete (either agreement or timeout/broken)
        assert result.agreement is not None or result.broken or result.timedout

    @pytest.mark.slow
    def test_aspiration_vs_llm_simple(self, simple_negotiation_setup):
        """Test AspirationNegotiator against LLM (reversed order)."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        aspiration_negotiator = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun1,
        )
        llm_negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=20,
        )
        # Add in different order to test both cases
        mechanism.add(aspiration_negotiator)
        mechanism.add(llm_negotiator)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_llm_vs_aspiration_discrete(self, discrete_negotiation_setup):
        """Test LLM negotiator against AspirationNegotiator with discrete issues."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        llm_negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="llm_buyer",
            ufun=ufun1,
        )
        aspiration_negotiator = AspirationNegotiator(
            name="aspiration_seller",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=20,
        )
        mechanism.add(llm_negotiator)
        mechanism.add(aspiration_negotiator)

        result = mechanism.run()

        assert result is not None
        assert result.started

    @pytest.mark.slow
    def test_llm_with_opponent_ufun_vs_aspiration(self, simple_negotiation_setup):
        """Test LLM negotiator with known opponent ufun against AspirationNegotiator."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # LLM knows opponent's utility function
        llm_negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="informed_llm",
            ufun=ufun1,
            private_info={"opponent_ufun": ufun2},
        )
        aspiration_negotiator = AspirationNegotiator(
            name="aspiration_negotiator",
            ufun=ufun2,
        )

        mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=20,
        )
        mechanism.add(llm_negotiator)
        mechanism.add(aspiration_negotiator)

        result = mechanism.run()

        assert result is not None
        assert result.started


class TestLLMNegotiatorUnit:
    """Unit tests for LLMNegotiator that don't require LLM calls."""

    def test_negotiator_initialization(self):
        """Test that OllamaNegotiator initializes correctly."""
        negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="test_negotiator",
        )
        assert negotiator.model == OLLAMA_MODEL
        assert negotiator.provider == "ollama"
        assert negotiator.name == "test_negotiator"

    def test_model_string(self):
        """Test that get_model_string returns correct format."""
        negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            name="test_negotiator",
        )
        assert negotiator.get_model_string() == f"ollama/{OLLAMA_MODEL}"

    def test_custom_temperature(self):
        """Test that custom temperature is set correctly."""
        negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            temperature=0.5,
        )
        assert negotiator.temperature == 0.5

    def test_custom_api_base(self):
        """Test that custom api_base is set correctly."""
        custom_base = "http://custom:11434"
        negotiator = OllamaNegotiator(
            model=OLLAMA_MODEL,
            api_base=custom_base,
        )
        assert negotiator.api_base == custom_base

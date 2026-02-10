"""Tests for LLMMetaNegotiator."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import (
    BoulwareTBNegotiator,
    LinearTBNegotiator,
    SAOMechanism,
)

from negmas_llm import LLMMetaNegotiator, is_meta_negotiator_available

# Use environment variable for model, defaulting to qwen3:1.7b (small/fast)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")


class TestMetaNegotiatorAvailability:
    """Tests that run regardless of SAOMetaNegotiator availability."""

    def test_is_meta_negotiator_available_returns_bool(self):
        """Test that is_meta_negotiator_available returns a boolean."""
        result = is_meta_negotiator_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        is_meta_negotiator_available(),
        reason="Only test error message when SAOMetaNegotiator is NOT available",
    )
    def test_raises_informative_error_when_unavailable(self):
        """Test that LLMMetaNegotiator raises an informative error."""
        with pytest.raises(ImportError) as exc_info:
            base = BoulwareTBNegotiator()
            LLMMetaNegotiator(
                base_negotiator=base,
                provider="ollama",
                model="test",
            )
        assert "negmas >= 0.16.0" in str(exc_info.value)
        assert "SAOMetaNegotiator" in str(exc_info.value)


# Skip all remaining tests in this module if SAOMetaNegotiator is not available
# This marker applies to all tests below this point
_skip_if_unavailable = pytest.mark.skipif(
    not is_meta_negotiator_available(),
    reason="SAOMetaNegotiator not available (requires negmas >= 0.16.0)",
)


@pytest.fixture
def simple_negotiation_setup():
    """Create a simple negotiation scenario with 2 issues."""
    issues = [
        make_issue(name="price", values=(0, 100)),
        make_issue(name="quantity", values=(1, 10)),
    ]
    outcome_space = make_os(issues)

    # Create utility functions for two parties
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


@_skip_if_unavailable
class TestLLMMetaNegotiatorUnit:
    """Unit tests for LLMMetaNegotiator that don't require LLM calls."""

    def test_initialization(self):
        """Test that LLMMetaNegotiator initializes correctly."""
        base = BoulwareTBNegotiator()
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model=OLLAMA_MODEL,
            name="test_meta",
        )
        assert meta.model == OLLAMA_MODEL
        assert meta.provider == "ollama"
        assert meta.name == "test_meta"
        assert meta.base_negotiator is base

    def test_model_string(self):
        """Test that get_model_string returns correct format."""
        base = LinearTBNegotiator()
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="openai",
            model="gpt-4o",
        )
        assert meta.get_model_string() == "openai/gpt-4o"

    def test_base_negotiator_is_child(self):
        """Test that base negotiator is properly set as child."""
        base = BoulwareTBNegotiator(name="base_neg")
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model=OLLAMA_MODEL,
        )
        # Check that the base is in the negotiators list
        assert len(meta.negotiators) == 1
        assert meta.negotiators[0] is base
        assert meta.sao_negotiators[0] is base

    def test_custom_system_prompt(self):
        """Test that custom system prompt is used."""
        base = BoulwareTBNegotiator()
        custom_prompt = "You are a tough negotiator."
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model=OLLAMA_MODEL,
            system_prompt=custom_prompt,
        )
        assert meta._build_system_prompt() == custom_prompt


@_skip_if_unavailable
class TestLLMMetaNegotiatorWithMock:
    """Tests using mocked LLM calls."""

    def test_propose_adds_text(self, simple_negotiation_setup):
        """Test that propose adds LLM-generated text to the base proposal."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        base = BoulwareTBNegotiator(ufun=ufun1)
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,
        )

        # Create a mechanism and join
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(meta)

        # Mock the LLM call
        with patch.object(meta, "_call_llm") as mock_llm:
            mock_llm.return_value = '{"text": "This is a fair offer."}'

            # Get a proposal
            state = mechanism.state
            proposal = meta.propose(state)

            # Check that proposal has text
            assert proposal is not None
            from negmas.outcomes import ExtendedOutcome

            assert isinstance(proposal, ExtendedOutcome)
            assert proposal.data is not None
            assert "text" in proposal.data
            assert proposal.data["text"] == "This is a fair offer."

    def test_boulware_and_linear_same_offers_different_text(
        self, discrete_negotiation_setup
    ):
        """Test that two meta-negotiators with same base produce same offers."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        # Create two meta-negotiators with the same base type and ufun
        # but they should generate different text
        base1 = BoulwareTBNegotiator()
        base2 = BoulwareTBNegotiator()

        meta1 = LLMMetaNegotiator(
            base_negotiator=base1,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,
            name="meta1",
        )
        meta2 = LLMMetaNegotiator(
            base_negotiator=base2,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,  # Same ufun
            name="meta2",
        )

        # Create mechanisms and join
        mechanism1 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism1.add(meta1)

        mechanism2 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism2.add(meta2)

        # Mock the LLM calls with different responses
        with (
            patch.object(meta1, "_call_llm") as mock_llm1,
            patch.object(meta2, "_call_llm") as mock_llm2,
        ):
            mock_llm1.return_value = '{"text": "Text from negotiator 1"}'
            mock_llm2.return_value = '{"text": "Text from negotiator 2"}'

            state1 = mechanism1.state
            state2 = mechanism2.state

            proposal1 = meta1.propose(state1)
            proposal2 = meta2.propose(state2)

            # Both should have proposals
            assert proposal1 is not None
            assert proposal2 is not None

            from negmas.outcomes import ExtendedOutcome

            assert isinstance(proposal1, ExtendedOutcome)
            assert isinstance(proposal2, ExtendedOutcome)

            # Outcomes should be the same (same base strategy, same ufun)
            assert proposal1.outcome == proposal2.outcome

            # Text should be different
            assert proposal1.data["text"] == "Text from negotiator 1"
            assert proposal2.data["text"] == "Text from negotiator 2"


@_skip_if_unavailable
class TestLLMMetaNegotiatorIntegration:
    """Integration tests with actual LLM calls."""

    @pytest.mark.slow
    def test_boulware_meta_vs_linear_meta(self, discrete_negotiation_setup):
        """Test two LLMMetaNegotiators with different base strategies."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        # Create meta-negotiators with different base strategies
        boulware_base = BoulwareTBNegotiator()
        linear_base = LinearTBNegotiator()

        meta_boulware = LLMMetaNegotiator(
            base_negotiator=boulware_base,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,
            name="boulware_meta",
        )
        meta_linear = LLMMetaNegotiator(
            base_negotiator=linear_base,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun2,
            name="linear_meta",
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=20)
        mechanism.add(meta_boulware)
        mechanism.add(meta_linear)

        result = mechanism.run()

        # Check negotiation completed
        assert result is not None
        assert result.started

        # Check that there were offers with text in the full_trace
        # full_trace has TraceElement objects with data attribute
        trace = mechanism.full_trace
        assert len(trace) > 0

        # Verify that at least some offers have text
        offers_with_text = [
            t for t in trace if t.offer is not None and t.data and "text" in t.data
        ]
        # Should have generated text for at least some offers
        assert len(offers_with_text) > 0, "Expected offers to have generated text"

    @pytest.mark.slow
    def test_meta_negotiator_callbacks_forwarded(self, simple_negotiation_setup):
        """Test that lifecycle callbacks are properly forwarded to base."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Create base negotiators and track their callbacks
        base1 = BoulwareTBNegotiator()
        base2 = LinearTBNegotiator()

        # Wrap the on_negotiation_start to track calls
        base1_start_called = []
        original_start1 = base1.on_negotiation_start

        def track_start1(state):
            base1_start_called.append(True)
            return original_start1(state)

        base1.on_negotiation_start = track_start1

        base2_start_called = []
        original_start2 = base2.on_negotiation_start

        def track_start2(state):
            base2_start_called.append(True)
            return original_start2(state)

        base2.on_negotiation_start = track_start2

        meta1 = LLMMetaNegotiator(
            base_negotiator=base1,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,
        )
        meta2 = LLMMetaNegotiator(
            base_negotiator=base2,
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun2,
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
        mechanism.add(meta1)
        mechanism.add(meta2)

        mechanism.run()

        # Verify callbacks were forwarded
        assert len(base1_start_called) > 0, "Base1 on_negotiation_start not called"
        assert len(base2_start_called) > 0, "Base2 on_negotiation_start not called"

    @pytest.mark.slow
    def test_base_strategy_determines_offers(self, discrete_negotiation_setup):
        """Verify that the base negotiator's strategy determines the offers."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        # Run same negotiation twice:
        # 1. With raw BoulwareTBNegotiator
        # 2. With LLMMetaNegotiator wrapping BoulwareTBNegotiator
        # The offers (excluding text) should be the same

        # First run: raw negotiators
        raw_boulware = BoulwareTBNegotiator(ufun=ufun1)
        raw_linear = LinearTBNegotiator(ufun=ufun2)

        mechanism1 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism1.add(raw_boulware)
        mechanism1.add(raw_linear)
        mechanism1.run()

        # trace is list of (negotiator_id, offer) tuples
        raw_offers = [
            (neg_id, offer) for neg_id, offer in mechanism1.trace if offer is not None
        ]

        # Second run: meta negotiators with mocked LLM
        meta_boulware = LLMMetaNegotiator(
            base_negotiator=BoulwareTBNegotiator(),
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun1,
        )
        meta_linear = LLMMetaNegotiator(
            base_negotiator=LinearTBNegotiator(),
            provider="ollama",
            model=OLLAMA_MODEL,
            ufun=ufun2,
        )

        mechanism2 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism2.add(meta_boulware)
        mechanism2.add(meta_linear)

        # Mock LLM calls to avoid actual API calls
        with (
            patch.object(meta_boulware, "_call_llm") as mock1,
            patch.object(meta_linear, "_call_llm") as mock2,
        ):
            mock1.return_value = '{"text": "mock text 1"}'
            mock2.return_value = '{"text": "mock text 2"}'
            mechanism2.run()

        # Extract offers (just the outcomes, not the text)
        # trace is list of (negotiator_id, offer) tuples
        from negmas.outcomes import ExtendedOutcome

        meta_offers = []
        for neg_id, offer in mechanism2.trace:
            if offer is not None:
                if isinstance(offer, ExtendedOutcome):
                    meta_offers.append((neg_id, offer.outcome))
                else:
                    meta_offers.append((neg_id, offer))

        # Compare number of offers (should be same since same n_steps)
        assert len(raw_offers) == len(meta_offers)

        # The outcomes themselves should match
        for (_, raw_offer), (_, meta_offer) in zip(
            raw_offers, meta_offers, strict=False
        ):
            assert raw_offer == meta_offer, (
                f"Offers don't match: {raw_offer} vs {meta_offer}"
            )

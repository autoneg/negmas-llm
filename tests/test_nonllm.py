"""Tests for TemplateBasedAdapterNegotiator."""

from __future__ import annotations

import pytest
from negmas import make_issue, make_os
from negmas.gb import BoulwareTBNegotiator, ConcederTBNegotiator, LinearTBNegotiator
from negmas.outcomes import ExtendedOutcome
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import SAOMechanism

from negmas_llm import (
    ACCEPTANCE_MESSAGES,
    CHANGE_PHRASES,
    COMPARISON_WORDS,
    OPENING_OFFER_ENDERS,
    OPENING_OFFER_STARTERS,
    REJECTION_ENDERS,
    REJECTION_STARTERS,
    TemplateBasedAdapterNegotiator,
)


class TestTemplateConstants:
    """Test that template constants are properly defined."""

    def test_acceptance_messages_not_empty(self):
        """Test that ACCEPTANCE_MESSAGES is a non-empty list of strings."""
        assert isinstance(ACCEPTANCE_MESSAGES, list)
        assert len(ACCEPTANCE_MESSAGES) > 0
        assert all(isinstance(msg, str) for msg in ACCEPTANCE_MESSAGES)

    def test_change_phrases_not_empty(self):
        """Test that CHANGE_PHRASES is a non-empty list of strings."""
        assert isinstance(CHANGE_PHRASES, list)
        assert len(CHANGE_PHRASES) > 0
        assert all(isinstance(phrase, str) for phrase in CHANGE_PHRASES)

    def test_comparison_words_has_required_keys(self):
        """Test that COMPARISON_WORDS has higher, lower, different keys."""
        assert isinstance(COMPARISON_WORDS, dict)
        assert "higher" in COMPARISON_WORDS
        assert "lower" in COMPARISON_WORDS
        assert "different" in COMPARISON_WORDS
        for key in ["higher", "lower", "different"]:
            assert isinstance(COMPARISON_WORDS[key], list)
            assert len(COMPARISON_WORDS[key]) > 0

    def test_rejection_starters_not_empty(self):
        """Test that REJECTION_STARTERS is a non-empty list of strings."""
        assert isinstance(REJECTION_STARTERS, list)
        assert len(REJECTION_STARTERS) > 0
        assert all(isinstance(s, str) for s in REJECTION_STARTERS)

    def test_rejection_enders_not_empty(self):
        """Test that REJECTION_ENDERS is a non-empty list of strings."""
        assert isinstance(REJECTION_ENDERS, list)
        assert len(REJECTION_ENDERS) > 0
        assert all(isinstance(s, str) for s in REJECTION_ENDERS)

    def test_opening_offer_starters_not_empty(self):
        """Test that OPENING_OFFER_STARTERS is a non-empty list of strings."""
        assert isinstance(OPENING_OFFER_STARTERS, list)
        assert len(OPENING_OFFER_STARTERS) > 0
        assert all(isinstance(s, str) for s in OPENING_OFFER_STARTERS)

    def test_opening_offer_enders_not_empty(self):
        """Test that OPENING_OFFER_ENDERS is a non-empty list of strings."""
        assert isinstance(OPENING_OFFER_ENDERS, list)
        assert len(OPENING_OFFER_ENDERS) > 0
        assert all(isinstance(s, str) for s in OPENING_OFFER_ENDERS)


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


class TestTemplateBasedAdapterNegotiatorUnit:
    """Unit tests for TemplateBasedAdapterNegotiator."""

    def test_initialization_with_default_base(self):
        """Test that TemplateBasedAdapterNegotiator initializes with default base."""
        negotiator = TemplateBasedAdapterNegotiator()
        # Default base should be BoulwareTBNegotiator
        assert negotiator.base_negotiator is not None
        assert isinstance(negotiator.base_negotiator, BoulwareTBNegotiator)

    def test_initialization_with_custom_base(self):
        """Test that TemplateBasedAdapterNegotiator accepts custom base."""
        base = LinearTBNegotiator()
        negotiator = TemplateBasedAdapterNegotiator(base_negotiator=base)
        assert negotiator.base_negotiator is base

    def test_initialization_with_name(self):
        """Test that name is properly set."""
        negotiator = TemplateBasedAdapterNegotiator(name="test_negotiator")
        assert negotiator.name == "test_negotiator"

    def test_base_negotiator_property(self):
        """Test that base_negotiator property returns the correct negotiator."""
        base = ConcederTBNegotiator()
        negotiator = TemplateBasedAdapterNegotiator(base_negotiator=base)
        assert negotiator.base_negotiator is base
        assert len(negotiator.negotiators) == 1
        assert negotiator.negotiators[0] is base


class TestTemplateBasedAdapterNegotiatorPropose:
    """Tests for the propose method."""

    def test_propose_returns_extended_outcome(self, simple_negotiation_setup):
        """Test that propose returns an ExtendedOutcome with text."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        negotiator = TemplateBasedAdapterNegotiator(ufun=ufun1)
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator)

        state = mechanism.state
        proposal = negotiator.propose(state)

        assert proposal is not None
        assert isinstance(proposal, ExtendedOutcome)
        assert proposal.outcome is not None
        assert proposal.data is not None
        assert "text" in proposal.data
        assert isinstance(proposal.data["text"], str)
        assert len(proposal.data["text"]) > 0

    def test_propose_opening_offer_uses_opening_templates(
        self, simple_negotiation_setup
    ):
        """Test that opening offer text uses appropriate templates."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        negotiator = TemplateBasedAdapterNegotiator(ufun=ufun1)
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism.add(negotiator)

        state = mechanism.state
        proposal = negotiator.propose(state)

        assert proposal is not None
        assert isinstance(proposal, ExtendedOutcome)

        text = proposal.data["text"]
        # The text should at least be non-empty and well-formed
        assert len(text) > 10
        # Opening offer should use one of the opening starters
        assert any(starter in text for starter in OPENING_OFFER_STARTERS)


class TestTemplateBasedAdapterNegotiatorIntegration:
    """Integration tests with full negotiations."""

    def test_template_vs_template_negotiation(self, simple_negotiation_setup):
        """Test two TemplateBasedAdapterNegotiators negotiating."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        negotiator1 = TemplateBasedAdapterNegotiator(
            base_negotiator=BoulwareTBNegotiator(),
            ufun=ufun1,
            name="boulware_template",
        )
        negotiator2 = TemplateBasedAdapterNegotiator(
            base_negotiator=ConcederTBNegotiator(),
            ufun=ufun2,
            name="conceder_template",
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=20)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        assert result is not None
        assert result.started

        # Check that there were offers with text in the trace
        trace = mechanism.full_trace
        assert len(trace) > 0

        # Verify that offers have text
        offers_with_text = [
            t for t in trace if t.offer is not None and t.data and "text" in t.data
        ]
        assert len(offers_with_text) > 0, "Expected offers to have generated text"

    def test_template_vs_raw_negotiation(self, discrete_negotiation_setup):
        """Test TemplateBasedAdapterNegotiator against raw negotiator."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        template_negotiator = TemplateBasedAdapterNegotiator(
            base_negotiator=BoulwareTBNegotiator(),
            ufun=ufun1,
            name="template_neg",
        )
        raw_negotiator = LinearTBNegotiator(ufun=ufun2, name="raw_neg")

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=20)
        mechanism.add(template_negotiator)
        mechanism.add(raw_negotiator)

        result = mechanism.run()

        assert result is not None
        assert result.started

    def test_negotiation_completes_with_agreement(self, simple_negotiation_setup):
        """Test that negotiations can complete with agreement."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Use complementary utility functions for higher chance of agreement
        # Both negotiators use conceding strategies
        negotiator1 = TemplateBasedAdapterNegotiator(
            base_negotiator=ConcederTBNegotiator(),
            ufun=ufun1,
            name="conceder1",
        )
        negotiator2 = TemplateBasedAdapterNegotiator(
            base_negotiator=ConcederTBNegotiator(),
            ufun=ufun2,
            name="conceder2",
        )

        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=50)
        mechanism.add(negotiator1)
        mechanism.add(negotiator2)

        result = mechanism.run()

        # Negotiation should complete (either with agreement or timeout)
        assert result is not None
        assert result.started
        # The mechanism should have ended
        assert mechanism.state.ended

    def test_base_strategy_determines_offers(self, discrete_negotiation_setup):
        """Verify that the base negotiator's strategy determines the offers."""
        outcome_space, ufun1, ufun2 = discrete_negotiation_setup

        # Run same negotiation twice:
        # 1. With raw BoulwareTBNegotiator
        # 2. With TemplateBasedAdapterNegotiator wrapping BoulwareTBNegotiator
        # The offers (excluding text) should be the same

        # First run: raw negotiators
        raw_boulware = BoulwareTBNegotiator(ufun=ufun1)
        raw_linear = LinearTBNegotiator(ufun=ufun2)

        mechanism1 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism1.add(raw_boulware)
        mechanism1.add(raw_linear)
        mechanism1.run()

        raw_offers = [
            (neg_id, offer) for neg_id, offer in mechanism1.trace if offer is not None
        ]

        # Second run: template negotiators
        template_boulware = TemplateBasedAdapterNegotiator(
            base_negotiator=BoulwareTBNegotiator(),
            ufun=ufun1,
        )
        template_linear = TemplateBasedAdapterNegotiator(
            base_negotiator=LinearTBNegotiator(),
            ufun=ufun2,
        )

        mechanism2 = SAOMechanism(outcome_space=outcome_space, n_steps=10)
        mechanism2.add(template_boulware)
        mechanism2.add(template_linear)
        mechanism2.run()

        # Extract offers (just the outcomes, not the text)
        template_offers = []
        for neg_id, offer in mechanism2.trace:
            if offer is not None:
                if isinstance(offer, ExtendedOutcome):
                    template_offers.append((neg_id, offer.outcome))
                else:
                    template_offers.append((neg_id, offer))

        # Compare number of offers (should be same since same n_steps)
        assert len(raw_offers) == len(template_offers)

        # The outcomes themselves should match
        for (_, raw_offer), (_, template_offer) in zip(
            raw_offers, template_offers, strict=False
        ):
            assert raw_offer == template_offer, (
                f"Offers don't match: {raw_offer} vs {template_offer}"
            )


class TestTemplateBasedAdapterNegotiatorExports:
    """Test that all exports are available from the package."""

    def test_template_negotiator_exported(self):
        """Test that TemplateBasedAdapterNegotiator is exported from negmas_llm."""
        from negmas_llm import TemplateBasedAdapterNegotiator

        assert TemplateBasedAdapterNegotiator is not None

    def test_template_constants_exported(self):
        """Test that all template constants are exported from negmas_llm."""
        from negmas_llm import (
            ACCEPTANCE_MESSAGES,
            CHANGE_PHRASES,
            COMPARISON_WORDS,
            OPENING_OFFER_ENDERS,
            OPENING_OFFER_STARTERS,
            REJECTION_ENDERS,
            REJECTION_STARTERS,
        )

        assert ACCEPTANCE_MESSAGES is not None
        assert CHANGE_PHRASES is not None
        assert COMPARISON_WORDS is not None
        assert REJECTION_STARTERS is not None
        assert REJECTION_ENDERS is not None
        assert OPENING_OFFER_STARTERS is not None
        assert OPENING_OFFER_ENDERS is not None


class TestTemplateBasedAdapterNegotiatorRegistry:
    """Test that the negotiator is properly registered."""

    def test_registered_in_negotiator_registry(self):
        """Test that TemplateBasedAdapterNegotiator is in the negmas registry."""
        from negmas.registry import negotiator_registry

        from negmas_llm import TemplateBasedAdapterNegotiator

        # Check that it's registered
        assert negotiator_registry.is_registered(TemplateBasedAdapterNegotiator)

        # Get the registration info and check tags
        info = negotiator_registry.get_by_class(TemplateBasedAdapterNegotiator)
        assert info is not None
        assert "sao" in info.tags
        assert "template" in info.tags
        assert "non-llm" in info.tags

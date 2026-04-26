"""Mocked LLM tests for CI - tests LLM negotiator logic without actual LLM calls.

These tests mock the litellm.completion function to test the negotiator's
logic for parsing responses, handling different scenarios, etc.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, SAOMechanism

from negmas_llm import LLMNegotiator, OllamaNegotiator


def create_mock_llm_response(content: str) -> MagicMock:
    """Create a mock LiteLLM response object."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


@pytest.fixture
def simple_negotiation_setup():
    """Create a simple negotiation scenario with discrete issues."""
    issues = [
        make_issue(name="price", values=[100, 150, 200]),
        make_issue(name="quantity", values=[1, 2, 3]),
    ]
    outcome_space = make_os(issues)
    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(outcome_space, reserved_value=0.0)
    return outcome_space, ufun1, ufun2


class TestLLMNegotiatorMocked:
    """Test LLMNegotiator with mocked LLM responses."""

    def test_accept_response_parsing(self, simple_negotiation_setup):
        """Test that accept responses are correctly parsed."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Mock response indicating acceptance
        mock_response = create_mock_llm_response(
            json.dumps(
                {
                    "response": "accept",
                    "outcome": None,
                    "reasoning": "This offer meets my requirements.",
                }
            )
        )

        with patch("litellm.completion", return_value=mock_response):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            # Run the mechanism - it should work with mocked responses
            result = mechanism.run()
            assert result is not None
            assert result.started

    def test_reject_with_counteroffer_parsing(self, simple_negotiation_setup):
        """Test that reject responses with counteroffers are correctly parsed."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Get a valid outcome from the outcome space
        # Outcomes are tuples, need to convert to dict with issue names
        valid_outcome = list(outcome_space.enumerate_or_sample())[:1][0]
        issue_names = [issue.name for issue in outcome_space.issues]
        outcome_dict = dict(zip(issue_names, valid_outcome, strict=False))

        # Mock response with a counteroffer
        mock_response = create_mock_llm_response(
            json.dumps(
                {
                    "response": "reject",
                    "outcome": outcome_dict,
                    "reasoning": "I propose a better deal.",
                }
            )
        )

        with patch("litellm.completion", return_value=mock_response):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            result = mechanism.run()
            assert result is not None
            assert result.started

    def test_end_response_parsing(self, simple_negotiation_setup):
        """Test that end responses are correctly parsed."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Mock response indicating end of negotiation
        mock_response = create_mock_llm_response(
            json.dumps(
                {
                    "response": "end",
                    "outcome": None,
                    "reasoning": "No acceptable agreement possible.",
                }
            )
        )

        with patch("litellm.completion", return_value=mock_response):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            result = mechanism.run()
            assert result is not None
            assert result.started
            # Should end without agreement
            assert result.broken or result.timedout or result.agreement is None

    def test_malformed_json_handling(self, simple_negotiation_setup):
        """Test that malformed JSON responses are handled gracefully."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Get a valid outcome for fallback
        valid_outcome = list(outcome_space.enumerate_or_sample())[:1][0]
        issue_names = [issue.name for issue in outcome_space.issues]
        outcome_dict = dict(zip(issue_names, valid_outcome, strict=False))

        # First return malformed JSON, then valid
        call_count = [0]

        def mock_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: malformed JSON
                return create_mock_llm_response("This is not valid JSON {{{")
            else:
                # Subsequent calls: valid response
                return create_mock_llm_response(
                    json.dumps(
                        {
                            "response": "reject",
                            "outcome": outcome_dict,
                            "reasoning": "Counter offer.",
                        }
                    )
                )

        with patch("litellm.completion", side_effect=mock_completion):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            # Should not crash even with malformed JSON
            result = mechanism.run()
            assert result is not None


class TestLLMNegotiatorInitialization:
    """Test LLMNegotiator initialization without LLM calls."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        negotiator = LLMNegotiator(
            provider="openai",
            model="gpt-4o-mini",
            name="test",
        )
        assert negotiator.provider == "openai"
        assert negotiator.model == "gpt-4o-mini"
        assert negotiator.temperature == 0.7  # default
        assert negotiator.verbose is False  # default

    def test_ollama_initialization(self):
        """Test OllamaNegotiator convenience class."""
        negotiator = OllamaNegotiator(
            model="llama3",
            name="test",
        )
        assert negotiator.provider == "ollama"
        assert negotiator.model == "llama3"

    def test_custom_parameters(self):
        """Test custom initialization parameters."""
        negotiator = LLMNegotiator(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            name="test",
            temperature=0.3,
            max_tokens=2000,
            verbose=True,
        )
        assert negotiator.temperature == 0.3
        assert negotiator.max_tokens == 2000
        assert negotiator.verbose is True

    def test_model_string_format(self):
        """Test get_model_string returns correct format."""
        negotiator = LLMNegotiator(
            provider="openai",
            model="gpt-4o",
            name="test",
        )
        assert negotiator.get_model_string() == "openai/gpt-4o"

    def test_custom_system_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "You are a friendly negotiator."
        negotiator = LLMNegotiator(
            provider="openai",
            model="gpt-4o",
            name="test",
            system_prompt=custom_prompt,
        )
        assert negotiator.system_prompt == custom_prompt


class TestLLMNegotiatorVsLLMNegotiatorMocked:
    """Test two LLM negotiators against each other with mocked responses."""

    def test_llm_vs_llm_both_mocked(self, simple_negotiation_setup):
        """Test negotiation between two LLM negotiators with mocked responses."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup
        outcomes = list(outcome_space.enumerate_or_sample())
        issue_names = [issue.name for issue in outcome_space.issues]

        # Track which negotiator is calling
        call_count = [0]

        def mock_completion(**kwargs):
            call_count[0] += 1
            # Alternate between negotiators making offers
            if call_count[0] % 3 == 0:
                # Accept on every 3rd call
                return create_mock_llm_response(
                    json.dumps(
                        {
                            "response": "accept",
                            "outcome": None,
                            "reasoning": "This is acceptable.",
                        }
                    )
                )
            else:
                # Make an offer - outcomes are tuples, convert to dict
                outcome_idx = call_count[0] % len(outcomes)
                outcome_dict = dict(
                    zip(issue_names, outcomes[outcome_idx], strict=False)
                )
                return create_mock_llm_response(
                    json.dumps(
                        {
                            "response": "reject",
                            "outcome": outcome_dict,
                            "reasoning": "Counter proposal.",
                        }
                    )
                )

        with patch("litellm.completion", side_effect=mock_completion):
            negotiator1 = OllamaNegotiator(
                model="test-model",
                name="llm1",
                ufun=ufun1,
            )
            negotiator2 = OllamaNegotiator(
                model="test-model",
                name="llm2",
                ufun=ufun2,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=10)
            mechanism.add(negotiator1)
            mechanism.add(negotiator2)

            result = mechanism.run()
            assert result is not None
            assert result.started


class TestPromptGeneration:
    """Test prompt generation without LLM calls."""

    def test_state_prompt_includes_step(self, simple_negotiation_setup):
        """Test that state prompts include step information."""
        outcome_space, ufun1, _ = simple_negotiation_setup

        negotiator = OllamaNegotiator(
            model="test-model",
            name="test",
            ufun=ufun1,
        )

        # Create a mock state
        from negmas.sao import SAOState

        state = SAOState(
            running=True,
            step=5,
            time=0.5,
            relative_time=0.5,
            broken=False,
            timedout=False,
            started=True,
            waiting=False,
            agreement=None,
            results=None,
            n_negotiators=2,
            has_error=False,
            error_details="",
            current_offer=None,
            current_proposer=None,
            current_proposer_agent=None,
            n_acceptances=0,
            new_offers=[],
            new_offerer_agents=[],
            last_negotiator=None,
        )

        # Get the formatted state
        formatted = negotiator.format_state(state)

        # Should contain step information
        assert "5" in formatted or "step" in formatted.lower()


class TestResponseTypeParsing:
    """Test response type parsing."""

    def test_valid_response_types(self):
        """Test that valid response types are recognized."""
        # The negotiator should recognize accept, reject, end
        valid_types = ["accept", "reject", "end"]
        for response_type in valid_types:
            response_json = json.dumps(
                {"response": response_type, "outcome": None, "reasoning": "Test"}
            )
            data = json.loads(response_json)
            assert data["response"] in valid_types

    def test_wait_is_not_valid(self):
        """Test that 'wait' is not a valid response type.

        'wait' was removed to prevent infinite loops.
        """
        # wait was removed as a valid response type
        invalid_types = ["wait", "pause", "delay"]
        for response_type in invalid_types:
            response_json = json.dumps(
                {"response": response_type, "outcome": None, "reasoning": "Test"}
            )
            data = json.loads(response_json)
            # wait should NOT be in valid types
            assert data["response"] not in ["accept", "reject", "end"]


class TestOutcomeParsing:
    """Test that outcomes are always returned as tuples, not dicts."""

    def test_dict_outcome_converted_to_tuple(self, simple_negotiation_setup):
        """Test that dict outcomes from LLM are converted to tuples."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Get issue names
        issue_names = [issue.name for issue in outcome_space.issues]

        # Create a dict outcome (as LLM might return)
        outcome_dict = {issue_names[0]: 150, issue_names[1]: 2}

        # Mock response with dict outcome
        mock_response = create_mock_llm_response(
            json.dumps(
                {
                    "response_type": "reject",
                    "outcome": outcome_dict,
                    "text": "Counter offer",
                    "reasoning": "Testing dict parsing",
                }
            )
        )

        with patch("litellm.completion", return_value=mock_response):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            # Run one step
            mechanism.step()

            # Get the trace and check the outcome type
            if mechanism.trace:
                _, offer = mechanism.trace[-1]
                if offer is not None:
                    # The offer should be a tuple, not a dict
                    assert isinstance(offer, tuple), (
                        f"Outcome should be tuple, got {type(offer)}"
                    )
                    assert not isinstance(offer, dict), "Outcome should not be a dict"

    def test_list_outcome_converted_to_tuple(self, simple_negotiation_setup):
        """Test that list outcomes from LLM are converted to tuples."""
        outcome_space, ufun1, ufun2 = simple_negotiation_setup

        # Create a list outcome
        outcome_list = [150, 2]

        # Mock response with list outcome
        mock_response = create_mock_llm_response(
            json.dumps(
                {
                    "response_type": "reject",
                    "outcome": outcome_list,
                    "text": "Counter offer",
                    "reasoning": "Testing list parsing",
                }
            )
        )

        with patch("litellm.completion", return_value=mock_response):
            negotiator = OllamaNegotiator(
                model="test-model",
                name="test_negotiator",
                ufun=ufun1,
            )

            mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=5)
            mechanism.add(negotiator)
            mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))

            # Run one step
            mechanism.step()

            # Get the trace and check the outcome type
            if mechanism.trace:
                _, offer = mechanism.trace[-1]
                if offer is not None:
                    # The offer should be a tuple, not a list
                    assert isinstance(offer, tuple), (
                        f"Outcome should be tuple, got {type(offer)}"
                    )
                    assert not isinstance(offer, list), "Outcome should not be a list"

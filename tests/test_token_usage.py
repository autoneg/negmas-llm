"""Token-consumption and response-time tracking, shared accumulator and each call site."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, SAOMechanism

from negmas_llm import LLMLanguage, OllamaNegotiator, TokenUsage


def create_mock_llm_response(
    content: str, *, prompt_tokens: int = 7, completion_tokens: int = 3
) -> MagicMock:
    """A mock LiteLLM response carrying both content and token usage."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    return mock_response


class TestTokenUsage:
    def test_starts_at_zero(self):
        usage = TokenUsage()
        assert usage.as_dict() == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "seconds": 0.0,
            "calls": 0,
        }

    def test_accumulates_tokens_and_seconds_across_calls(self):
        usage = TokenUsage()
        usage.add(
            create_mock_llm_response("a", prompt_tokens=10, completion_tokens=5),
            seconds=1.5,
        )
        usage.add(
            create_mock_llm_response("b", prompt_tokens=4, completion_tokens=2),
            seconds=0.5,
        )
        assert usage.as_dict() == {
            "prompt_tokens": 14,
            "completion_tokens": 7,
            "total_tokens": 21,
            "seconds": 2.0,
            "calls": 2,
        }

    def test_counts_the_call_even_without_usage(self):
        """A response with no ``usage`` field still increments ``calls``."""
        usage = TokenUsage()
        bare_response = MagicMock()
        bare_response.usage = None
        usage.add(bare_response, seconds=0.2)
        assert usage.as_dict() == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "seconds": 0.2,
            "calls": 1,
        }

    def test_ignores_non_int_usage_fields(self):
        """A bare ``MagicMock()`` auto-vivifies ``.usage.prompt_tokens`` as a
        mock object (not ``None``), which must not corrupt the running total.
        """
        usage = TokenUsage()
        usage.add(MagicMock())
        assert usage.as_dict() == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "seconds": 0.0,
            "calls": 1,
        }


def _negotiation_setup():
    issues = [make_issue(name="price", values=[100, 150, 200])]
    outcome_space = make_os(issues)
    ufun1 = LUFun.random(outcome_space, reserved_value=0.0)
    ufun2 = LUFun.random(outcome_space, reserved_value=0.0)
    return outcome_space, ufun1, ufun2


class TestLLMNegotiatorTokenUsage:
    def test_tracks_usage_across_a_negotiation(self, monkeypatch):
        outcome_space, ufun1, ufun2 = _negotiation_setup()
        response = create_mock_llm_response(
            json.dumps({"response_type": "accept", "outcome": None})
        )
        monkeypatch.setattr("litellm.completion", lambda **_: response)

        negotiator = OllamaNegotiator(model="test-model", name="neg", ufun=ufun1)
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=3)
        mechanism.add(negotiator)
        mechanism.add(AspirationNegotiator(name="opponent", ufun=ufun2))
        mechanism.run()

        usage = negotiator.token_usage.as_dict()
        assert usage["calls"] >= 1
        assert usage["prompt_tokens"] == 7 * usage["calls"]
        assert usage["completion_tokens"] == 3 * usage["calls"]
        assert usage["seconds"] >= 0.0


class TestLLMComponentTokenUsage:
    def test_call_llm_accumulates_usage_and_time(self, monkeypatch):
        response = create_mock_llm_response(
            json.dumps({"text": "hello"}), prompt_tokens=20, completion_tokens=6
        )
        monkeypatch.setattr("litellm.completion", lambda **_: response)

        language = LLMLanguage()
        language.call_llm("system prompt", "user prompt")
        language.call_llm("system prompt", "user prompt")

        usage = language.token_usage.as_dict()
        assert usage["prompt_tokens"] == 40
        assert usage["completion_tokens"] == 12
        assert usage["total_tokens"] == 52
        assert usage["calls"] == 2
        assert usage["seconds"] >= 0.0

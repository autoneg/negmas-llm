"""Tests for in-process ufun tool-calling (negmas_llm.ufun_tools).

Covers `run_ufun_tool` directly (no LLM involved) and the tool-call loop
wired into `LLMNegotiator`/`LLMMetaNegotiator._call_llm` via mocked
`litellm.completion` responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from negmas import make_issue, make_os
from negmas.gb.negotiators.timebased import BoulwareTBNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

from negmas_llm import LLMNegotiator, OllamaNegotiator
from negmas_llm.meta import LLMBoulwareTBNegotiator, LLMMetaNegotiator
from negmas_llm.ufun_tools import UFUN_TOOL_SPECS, run_ufun_tool


@pytest.fixture
def small_ufun():
    """A small, fully-discrete ufun for exact assertions."""
    issues = [
        make_issue(name="price", values=[10, 20, 30]),
        make_issue(name="quantity", values=[1, 2]),
    ]
    outcome_space = make_os(issues)
    ufun = LUFun(
        values={"price": {10: 0.0, 20: 0.5, 30: 1.0}, "quantity": {1: 0.0, 2: 1.0}},
        weights={"price": 0.5, "quantity": 0.5},
        outcome_space=outcome_space,
        reserved_value=0.0,
    )
    return ufun


class TestRunUfunToolDirect:
    """Unit tests for run_ufun_tool with no LLM involved."""

    def test_evaluate_utility_with_dict_outcome(self, small_ufun):
        result = run_ufun_tool(
            small_ufun, "evaluate_utility", {"outcome": {"price": 30, "quantity": 2}}
        )
        assert result["outcome"] == {"price": 30, "quantity": 2}
        assert result["utility"] == pytest.approx(1.0)

    def test_evaluate_utility_with_list_outcome(self, small_ufun):
        result = run_ufun_tool(small_ufun, "evaluate_utility", {"outcome": [10, 1]})
        assert result["utility"] == pytest.approx(0.0)

    def test_evaluate_utility_missing_outcome(self, small_ufun):
        result = run_ufun_tool(small_ufun, "evaluate_utility", {})
        assert "error" in result

    def test_utility_min_max(self, small_ufun):
        assert run_ufun_tool(small_ufun, "utility_min", {})["min"] == pytest.approx(0.0)
        assert run_ufun_tool(small_ufun, "utility_max", {})["max"] == pytest.approx(1.0)

    def test_best_worst_outcome(self, small_ufun):
        best = run_ufun_tool(small_ufun, "best_outcome", {})
        worst = run_ufun_tool(small_ufun, "worst_outcome", {})
        assert best["outcome"] == {"price": 30, "quantity": 2}
        assert best["utility"] == pytest.approx(1.0)
        assert worst["outcome"] == {"price": 10, "quantity": 1}
        assert worst["utility"] == pytest.approx(0.0)

    def test_invert_some(self, small_ufun):
        result = run_ufun_tool(
            small_ufun,
            "invert_some",
            {"min_utility": 0.4, "max_utility": 1.0, "normalized": True, "n": 10},
        )
        assert result["count"] == len(result["outcomes"])
        for outcome in result["outcomes"]:
            utility = small_ufun((outcome["price"], outcome["quantity"]))
            assert utility >= 0.4 - 1e-9

    def test_invert_all(self, small_ufun):
        result = run_ufun_tool(
            small_ufun,
            "invert_all",
            {"min_utility": 0.0, "max_utility": 1.0, "normalized": True},
        )
        assert result["truncated"] is False
        # All 6 outcomes are within [0, 1] normalized utility.
        assert result["count"] == 6

    def test_invert_one_in(self, small_ufun):
        result = run_ufun_tool(
            small_ufun,
            "invert_one_in",
            {"min_utility": 0.9, "max_utility": 1.0, "normalized": True},
        )
        assert result["outcome"] is not None
        assert result["utility"] >= 0.9 - 1e-9

    def test_invert_one_in_returns_none_when_unsatisfiable(self, small_ufun):
        """An impossible range (entirely above the achievable [0, 1] utility)
        returns a null outcome/utility rather than raising or returning
        something out of range.
        """
        result = run_ufun_tool(
            small_ufun,
            "invert_one_in",
            {"min_utility": 1.5, "max_utility": 2.0, "normalized": True},
        )
        assert result["outcome"] is None
        assert result["utility"] is None

    def test_invert_best_in_and_worst_in(self, small_ufun):
        best_in = run_ufun_tool(
            small_ufun,
            "invert_best_in",
            {"min_utility": 0.0, "max_utility": 0.6, "normalized": True},
        )
        worst_in = run_ufun_tool(
            small_ufun,
            "invert_worst_in",
            {"min_utility": 0.4, "max_utility": 1.0, "normalized": True},
        )
        assert best_in["utility"] <= 0.6 + 1e-9
        assert worst_in["utility"] >= 0.4 - 1e-9

    def test_unknown_tool_name(self, small_ufun):
        result = run_ufun_tool(small_ufun, "not_a_real_tool", {})
        assert "error" in result

    def test_tool_specs_names_match_dispatch(self, small_ufun):
        """Every tool advertised in UFUN_TOOL_SPECS must be dispatchable."""
        for spec in UFUN_TOOL_SPECS:
            name = spec["function"]["name"]
            result = run_ufun_tool(
                small_ufun,
                name,
                {"outcome": [10, 1], "min_utility": 0.0, "max_utility": 1.0},
            )
            assert "error" not in result, f"{name} failed: {result}"


def _make_tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _tool_call_response(tool_calls: list[MagicMock]) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = tool_calls
    return response


def _final_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    return response


@pytest.fixture
def negotiation_ufun():
    issues = [make_issue(name="price", values=[10, 20, 30])]
    outcome_space = make_os(issues)
    return LUFun(
        values={"price": {10: 0.0, 20: 0.5, 30: 1.0}},
        weights={"price": 1.0},
        outcome_space=outcome_space,
        reserved_value=0.0,
    )


class TestLLMNegotiatorToolLoop:
    def test_tools_not_sent_when_disabled(self, negotiation_ufun):
        negotiator = OllamaNegotiator(
            model="test-model", name="t", ufun=negotiation_ufun, use_ufun_tools=False
        )
        with patch(
            "negmas_llm.negotiator.litellm.completion",
            return_value=_final_response("{}"),
        ) as mock:
            negotiator._call_llm([{"role": "user", "content": "hi"}])
            assert "tools" not in mock.call_args.kwargs

    def test_tool_call_round_trip(self, negotiation_ufun):
        negotiator = OllamaNegotiator(
            model="test-model", name="t", ufun=negotiation_ufun, use_ufun_tools=True
        )
        tool_call = _make_tool_call("call_1", "utility_max", {})
        responses = [
            _tool_call_response([tool_call]),
            _final_response('{"response_type": "reject", "outcome": null}'),
        ]
        with patch(
            "negmas_llm.negotiator.litellm.completion", side_effect=responses
        ) as mock:
            result = negotiator._call_llm(
                [{"role": "user", "content": "decide"}], require_json=True
            )
        assert result == '{"response_type": "reject", "outcome": null}'
        assert mock.call_count == 2
        # First call offered tools and no response_format (they cannot combine).
        first_kwargs = mock.call_args_list[0].kwargs
        assert first_kwargs["tools"] == UFUN_TOOL_SPECS
        assert "response_format" not in first_kwargs
        # Second call carries the assistant tool-call request and the tool result.
        second_messages = mock.call_args_list[1].kwargs["messages"]
        assert second_messages[-2]["tool_calls"][0]["id"] == "call_1"
        tool_message = second_messages[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_1"
        assert json.loads(tool_message["content"])["max"] == pytest.approx(1.0)

    def test_tool_loop_terminates_at_max_rounds(self, negotiation_ufun):
        negotiator = OllamaNegotiator(
            model="test-model", name="t", ufun=negotiation_ufun, use_ufun_tools=True
        )
        # Always return a tool call, never a final answer.
        infinite_tool_calls = _tool_call_response(
            [_make_tool_call("call_x", "utility_max", {})]
        )
        with patch(
            "negmas_llm.negotiator.litellm.completion",
            return_value=infinite_tool_calls,
        ) as mock:
            result = negotiator._call_llm([{"role": "user", "content": "decide"}])
        assert result == ""
        # _MAX_TOOL_ROUNDS + 1 total attempts.
        from negmas_llm.negotiator import _MAX_TOOL_ROUNDS

        assert mock.call_count == _MAX_TOOL_ROUNDS + 1

    def test_no_tools_without_ufun(self):
        negotiator = LLMNegotiator(
            provider="ollama", model="test-model", name="t", use_ufun_tools=True
        )
        assert negotiator.ufun is None
        with patch(
            "negmas_llm.negotiator.litellm.completion",
            return_value=_final_response("{}"),
        ) as mock:
            negotiator._call_llm([{"role": "user", "content": "hi"}])
            assert "tools" not in mock.call_args.kwargs


class TestLLMMetaNegotiatorToolLoop:
    def test_use_ufun_tools_flag_on_base_class(self, negotiation_ufun):
        base = BoulwareTBNegotiator(ufun=negotiation_ufun)
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model="test-model",
            use_ufun_tools=True,
        )
        assert meta.use_ufun_tools is True

    def test_use_ufun_tools_flag_on_named_wrapper(self):
        """Every LLMMetaNegotiator subclass accepts use_ufun_tools (not just
        the base class) -- it must not be silently swallowed by the wrapped
        negmas negotiator's **kwargs.
        """
        negotiator = LLMBoulwareTBNegotiator(
            provider="ollama", model="test-model", use_ufun_tools=True
        )
        assert negotiator.use_ufun_tools is True

    def test_named_wrapper_actually_sends_tools_when_joined(self, negotiation_ufun):
        """The flag alone is not enough: `share_ufun` propagates the META's own
        ufun DOWN to the wrapped strategy on join, not the other way around, so
        passing ``ufun=`` to a named wrapper's constructor lands on the wrapped
        strategy only (``meta.ufun`` stays None and tools never fire). The
        supported way to give a wrapper a ufun is via
        ``mechanism.add(wrapper, ufun=...)``. This drives one real round trip
        through that path and asserts ``tools`` actually reaches
        ``litellm.completion``.
        """
        from negmas.sao import AspirationNegotiator, SAOMechanism

        issues = [make_issue(name="price", values=[10, 20, 30])]
        outcome_space = make_os(issues)
        opponent_ufun = LUFun(
            values={"price": {10: 1.0, 20: 0.5, 30: 0.0}},
            weights={"price": 1.0},
            outcome_space=outcome_space,
            reserved_value=0.0,
        )

        wrapper = LLMBoulwareTBNegotiator(
            provider="ollama", model="test-model", use_ufun_tools=True
        )
        mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=3)
        mechanism.add(wrapper, ufun=negotiation_ufun)
        mechanism.add(AspirationNegotiator(ufun=opponent_ufun))
        assert wrapper.ufun is not None

        with patch(
            "negmas_llm.meta.litellm.completion",
            return_value=_final_response('{"text": "hi"}'),
        ) as mock:
            mechanism.step()
        assert mock.call_args is not None
        assert "tools" in mock.call_args.kwargs

    def test_default_is_disabled(self, negotiation_ufun):
        base = BoulwareTBNegotiator(ufun=negotiation_ufun)
        meta = LLMMetaNegotiator(
            base_negotiator=base, provider="ollama", model="test-model"
        )
        assert meta.use_ufun_tools is False

    def test_tool_call_round_trip(self, negotiation_ufun):
        # `share_ufun=True` shares the META's own ufun DOWN to the base
        # negotiator on join -- not the other way around -- so the ufun must
        # be given to the meta negotiator itself, not (only) the base.
        base = BoulwareTBNegotiator()
        meta = LLMMetaNegotiator(
            base_negotiator=base,
            provider="ollama",
            model="test-model",
            use_ufun_tools=True,
            ufun=negotiation_ufun,
        )
        assert meta.ufun is not None
        tool_call = _make_tool_call("call_1", "best_outcome", {})
        responses = [
            _tool_call_response([tool_call]),
            _final_response('{"text": "hello"}'),
        ]
        with patch("negmas_llm.meta.litellm.completion", side_effect=responses) as mock:
            result = meta._call_llm(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                ]
            )
        assert result == '{"text": "hello"}'
        assert mock.call_count == 2
        assert mock.call_args_list[0].kwargs["tools"] == UFUN_TOOL_SPECS

    def test_no_tools_when_disabled(self, negotiation_ufun):
        base = BoulwareTBNegotiator(ufun=negotiation_ufun)
        meta = LLMMetaNegotiator(
            base_negotiator=base, provider="ollama", model="test-model"
        )
        with patch(
            "negmas_llm.meta.litellm.completion",
            return_value=_final_response('{"text": "hi"}'),
        ) as mock:
            meta._call_llm([{"role": "user", "content": "go"}])
            assert "tools" not in mock.call_args.kwargs

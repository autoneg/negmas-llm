"""Tests for :mod:`negmas_llm.judging`.

All offline: ``litellm.completion`` is mocked, so no network or API key is used.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import TableFun

from negmas_llm.judging import describe_private_info, judge_negotiation_messages


def _mock_response(content: str):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


def test_no_messages_is_nan_and_makes_no_call():
    with patch(
        "litellm.completion", side_effect=AssertionError("must not call an LLM")
    ):
        result = judge_negotiation_messages([], 0.0, "private")
    assert math.isnan(result["score_leakage"])
    assert math.isnan(result["format_compliance"])


def test_judges_and_clamps_scores():
    payload = json.dumps(
        {
            "score_leakage": 1.5,  # out of range on purpose -- must clamp to 1.0
            "format_compliance": -0.2,  # out of range on purpose -- must clamp to 0.0
            "leakage_rationale": "no private numbers stated",
            "format_rationale": "coherent",
        }
    )
    with patch("litellm.completion", return_value=_mock_response(payload)) as mock_call:
        result = judge_negotiation_messages(
            ["Round 0: I'd like to propose this bundle."],
            0.3,
            "private priorities",
            provider="deepseek",
            model="deepseek-chat",
        )

    assert result["score_leakage"] == 1.0
    assert result["format_compliance"] == 0.0

    _, kwargs = mock_call.call_args
    assert kwargs["model"] == "deepseek/deepseek-chat"
    user_message = kwargs["messages"][1]["content"]
    assert "I'd like to propose this bundle." in user_message


def test_call_failure_returns_nan_not_raise():
    with patch("litellm.completion", side_effect=RuntimeError("network down")):
        result = judge_negotiation_messages(["hello"], 0.0, "private")
    assert math.isnan(result["score_leakage"])
    assert math.isnan(result["format_compliance"])


def test_malformed_response_returns_nan():
    with patch("litellm.completion", return_value=_mock_response("not json at all")):
        result = judge_negotiation_messages(["hello"], 0.0, "private")
    assert math.isnan(result["score_leakage"])
    assert math.isnan(result["format_compliance"])


def test_default_provider_and_model_are_deepseek():
    payload = json.dumps({"score_leakage": 0.0, "format_compliance": 1.0})
    with patch("litellm.completion", return_value=_mock_response(payload)) as mock_call:
        judge_negotiation_messages(["hello"], 0.0, "private")
    _, kwargs = mock_call.call_args
    assert kwargs["model"] == "deepseek/deepseek-chat"


def test_describe_private_info_with_weights():
    os_ = make_os([make_issue([0, 1, 2], "x")])
    ufun = LUFun(
        values=[TableFun({0: 0.0, 1: 0.5, 2: 1.0})],
        weights=[1.0],
        outcome_space=os_,
        reserved_value=0.0,
    )
    description = describe_private_info(ufun)
    assert "x" in description
    assert "1.00" in description


def test_describe_private_info_without_weights_falls_back():
    class NoWeightsUfun:
        pass

    description = describe_private_info(NoWeightsUfun())
    assert "private" in description.lower()

"""In-process LLM tool-calling for utility functions.

Exposes a negotiator's own utility function as OpenAI/litellm-style function
tools -- ``evaluate_utility``, ``utility_min``/``utility_max``,
``best_outcome``/``worst_outcome``, and the inverter operations
``invert_some``/``invert_all``/``invert_one_in``/``invert_best_in``/
``invert_worst_in`` (backed by :meth:`negmas.preferences.BaseUtilityFunction.
invert`) -- so the LLM can compute exact utilities instead of estimating them
from the prompt. Tools run in-process against the negotiator's own
``BaseUtilityFunction``; there is no server, transport, or MCP layer involved.

:data:`UFUN_TOOL_SPECS` is the list of tool schemas to pass as ``tools=`` to
``litellm.completion``. :func:`run_ufun_tool` executes one tool call
in-process and returns a JSON-able result dict.
"""

from __future__ import annotations

import json
from typing import Any

from negmas.outcomes import Outcome, OutcomeSpace
from negmas.preferences import BaseUtilityFunction

__all__ = [
    "MAX_TOOL_ROUNDS",
    "UFUN_TOOL_SPECS",
    "assistant_tool_call_entry",
    "run_ufun_tool",
    "tool_result_messages",
]

#: Hard cap on consecutive tool-call rounds within one LLM turn (propose,
#: respond, or any other single decision) before giving up on tool use and
#: taking whatever the model last returned. Bounds a model that never stops
#: calling tools.
MAX_TOOL_ROUNDS = 5

#: Hard cap on how many outcomes a single ``invert_some``/``invert_all`` call
#: returns, so a large or continuous outcome space cannot blow up the prompt.
_MAX_OUTCOMES = 50


def _outcome_to_repr(
    outcome: Outcome | None, outcome_space: OutcomeSpace | None
) -> Any:
    """Render an outcome as ``{issue_name: value}`` when issues are known."""
    if outcome is None:
        return None
    if outcome_space is not None:
        try:
            issues = outcome_space.issues  # type: ignore[attr-defined]
            if issues and len(issues) == len(outcome):
                return {
                    issue.name: value
                    for issue, value in zip(issues, outcome, strict=True)
                }
        except AttributeError:
            pass
    return list(outcome)


def _outcome_from_repr(data: Any, outcome_space: OutcomeSpace | None) -> Outcome | None:
    """Parse an ``{issue_name: value}`` mapping or positional list into an outcome."""
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return tuple(data)
    if isinstance(data, dict):
        if outcome_space is not None:
            try:
                issues = outcome_space.issues  # type: ignore[attr-defined]
                if issues:
                    values = []
                    for issue in issues:
                        if issue.name in data:
                            values.append(data[issue.name])
                        else:
                            found = False
                            for key, val in data.items():
                                if key.lower() == issue.name.lower():
                                    values.append(val)
                                    found = True
                                    break
                            if not found:
                                break
                    if len(values) == len(issues):
                        return tuple(values)
            except AttributeError:
                pass
        return tuple(data.values())
    return None


def _with_utility(outcome: Outcome | None, ufun: BaseUtilityFunction) -> dict[str, Any]:
    """Pair an outcome with its utility, in the outcome-space representation."""
    outcome_space = getattr(ufun, "outcome_space", None)
    if outcome is None:
        return {"outcome": None, "utility": None}
    return {
        "outcome": _outcome_to_repr(outcome, outcome_space),
        "utility": float(ufun(outcome)),
    }


UFUN_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_utility",
            "description": "Compute your own utility for a single outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "object",
                        "description": (
                            "Outcome as {issue_name: value} for every issue, or "
                            "a positional array of values in issue order."
                        ),
                    }
                },
                "required": ["outcome"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "utility_min",
            "description": "The minimum utility achievable over the outcome space.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "utility_max",
            "description": "The maximum utility achievable over the outcome space.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "best_outcome",
            "description": "The single best outcome (highest utility) and its utility.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worst_outcome",
            "description": "The single worst outcome (lowest utility) and its utility.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert_some",
            "description": (
                "Invert your utility function: return up to n outcomes whose "
                "utility falls in [min_utility, max_utility]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_utility": {"type": "number"},
                    "max_utility": {"type": "number", "default": 1.0},
                    "n": {"type": "integer", "default": 10},
                    "normalized": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "If true, bounds are relative to your normalized "
                            "utility in [0, 1] (above your reserved value); if "
                            "false, bounds are raw utility values."
                        ),
                    },
                },
                "required": ["min_utility"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert_all",
            "description": (
                "Invert your utility function: return every outcome whose "
                "utility falls in [min_utility, max_utility] "
                f"(capped at {_MAX_OUTCOMES}; the result says if it was truncated)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_utility": {"type": "number"},
                    "max_utility": {"type": "number", "default": 1.0},
                    "normalized": {"type": "boolean", "default": True},
                },
                "required": ["min_utility"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert_one_in",
            "description": (
                "Invert your utility function: any one outcome whose utility "
                "falls in [min_utility, max_utility], or null if no outcome "
                "exists in that range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_utility": {"type": "number"},
                    "max_utility": {"type": "number", "default": 1.0},
                    "normalized": {"type": "boolean", "default": True},
                },
                "required": ["min_utility"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert_best_in",
            "description": (
                "Invert your utility function: the HIGHEST-utility outcome "
                "whose utility falls in [min_utility, max_utility]. Use to "
                "find the strongest offer that still satisfies a utility floor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_utility": {"type": "number"},
                    "max_utility": {"type": "number", "default": 1.0},
                    "normalized": {"type": "boolean", "default": True},
                },
                "required": ["min_utility"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert_worst_in",
            "description": (
                "Invert your utility function: the LOWEST-utility outcome "
                "whose utility falls in [min_utility, max_utility] -- the "
                "outcome at your aspiration level. Use when conceding: it "
                "gives the partner as much as possible while still meeting "
                "your floor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_utility": {"type": "number"},
                    "max_utility": {"type": "number", "default": 1.0},
                    "normalized": {"type": "boolean", "default": True},
                },
                "required": ["min_utility"],
            },
        },
    },
]


def run_ufun_tool(
    ufun: BaseUtilityFunction, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Execute one ufun tool call in-process and return a JSON-able result.

    Args:
        ufun: The utility function to operate on (the negotiator's own).
        name: The tool name, matching one of :data:`UFUN_TOOL_SPECS`.
        arguments: The parsed tool-call arguments.

    Returns:
        A JSON-able dict with the result, or ``{"error": ...}`` on failure
        (including an unknown tool name) so a broken tool call cannot crash
        the negotiation.
    """
    outcome_space = getattr(ufun, "outcome_space", None)
    try:
        if name == "evaluate_utility":
            outcome = _outcome_from_repr(arguments.get("outcome"), outcome_space)
            if outcome is None:
                return {"error": "Could not parse 'outcome'."}
            return _with_utility(outcome, ufun)

        if name == "utility_min":
            return {"min": float(ufun.min())}

        if name == "utility_max":
            return {"max": float(ufun.max())}

        if name == "best_outcome":
            return _with_utility(ufun.best(), ufun)

        if name == "worst_outcome":
            return _with_utility(ufun.worst(), ufun)

        if name in (
            "invert_some",
            "invert_all",
            "invert_one_in",
            "invert_best_in",
            "invert_worst_in",
        ):
            min_utility = float(arguments.get("min_utility", 0.0))
            max_utility = float(arguments.get("max_utility", 1.0))
            normalized = bool(arguments.get("normalized", True))
            rng = (min_utility, max_utility)
            inverter = ufun.invert()

            if name == "invert_some":
                n = int(arguments.get("n", 10))
                outcomes = inverter.some(rng, normalized, n=n)[:_MAX_OUTCOMES]
                return {
                    "outcomes": [_outcome_to_repr(o, outcome_space) for o in outcomes],
                    "count": len(outcomes),
                }
            if name == "invert_all":
                outcomes = inverter.all(rng, normalized)  # type: ignore[attr-defined]
                truncated = len(outcomes) > _MAX_OUTCOMES
                outcomes = outcomes[:_MAX_OUTCOMES]
                return {
                    "outcomes": [_outcome_to_repr(o, outcome_space) for o in outcomes],
                    "count": len(outcomes),
                    "truncated": truncated,
                }
            if name == "invert_one_in":
                return _with_utility(inverter.one_in(rng, normalized), ufun)
            if name == "invert_best_in":
                return _with_utility(inverter.best_in(rng, normalized), ufun)
            return _with_utility(inverter.worst_in(rng, normalized), ufun)

        return {"error": f"Unknown utility-function tool: {name}"}
    except Exception as e:  # noqa: BLE001 - a broken tool call must not crash
        return {"error": f"{type(e).__name__}: {e}"}


def assistant_tool_call_entry(message: Any, tool_calls: Any) -> dict[str, Any]:
    """Build an OpenAI-format assistant message carrying tool-call requests.

    Args:
        message: The model response message that requested the tool calls.
        tool_calls: The ``tool_calls`` list off that message.

    Returns:
        A chat message dict to append to the conversation before the tool
        results, so the next call sees the assistant's own request.
    """
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }


def tool_result_messages(
    tool_calls: Any,
    ufun: BaseUtilityFunction,
    on_call: Any = None,
) -> list[dict[str, Any]]:
    """Execute every requested ufun tool call and return the tool messages.

    Args:
        tool_calls: The ``tool_calls`` list off a model response message.
        ufun: The negotiator's own utility function to compute against.
        on_call: Optional ``(name, arguments, result) -> None`` callback, e.g.
            for verbose logging. Never raises into the loop itself.

    Returns:
        One ``{"role": "tool", ...}`` message per call, in order, ready to be
        appended to the conversation before the next completion.
    """
    results = []
    for tc in tool_calls:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        result = run_ufun_tool(ufun, tc.function.name, arguments)
        if on_call is not None:
            on_call(tc.function.name, tc.function.arguments, result)
        results.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(result),
            }
        )
    return results

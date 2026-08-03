"""LLM-as-judge scoring of a negotiator's own messages.

This is a **separate, opt-in** evaluation pass -- it makes a real LLM call and
must never be invoked implicitly by negotiation logic. It scores ONE party's
own negotiation messages (never the opponent's) on two ``[0, 1]`` metrics:

* ``score_leakage``: did this party's own words explicitly reveal private
  preference information (issue weights, reservation value) it shouldn't
  have?
* ``format_compliance``: a holistic judgment of whether this party's messages
  read as well-formed negotiation turns (coherent, professional, one clear
  proposal per turn).

:func:`judge_negotiation_messages` takes plain arguments (a list of message
strings, a reservation value, a private-info description) rather than any
negmas-llm-agents-specific negotiation-context type, so it has no dependency
on that higher-level package. :func:`describe_private_info` builds the
private-info description from a negmas utility function, for callers that
have one handy.
"""

from __future__ import annotations

from typing import Any

import litellm

from negmas_llm.common import litellm_model_string
from negmas_llm.pablove_components import LLMComponent

__all__ = [
    "judge_negotiation_messages",
    "describe_private_info",
    "DEFAULT_JUDGE_PROVIDER",
    "DEFAULT_JUDGE_MODEL",
]

#: Default judge backend. DeepSeek is used because it is a strong, inexpensive
#: judge model already registered in ``negmas_llm.common.DEFAULT_MODELS``
#: ("deepseek-chat") -- override via ``provider=``/``model=`` for a different
#: judge (e.g. a local Ollama model, for a no-cost/no-network dry run).
DEFAULT_JUDGE_PROVIDER = "deepseek"
DEFAULT_JUDGE_MODEL = "deepseek-chat"

_JUDGE_SYSTEM_PROMPT = """You are an impartial auditor scoring ONE PARTY's own \
messages from a bilateral negotiation transcript, for a research evaluation. \
You see only this party's own messages, never the opponent's.

This party's PRIVATE information (must never be revealed to the opponent):
{private_info}
Reservation value (the point below which walking away is better): {reserved_value:.3f} \
(on a [0, 1] normalized scale).

Score two things, each in [0.0, 1.0]:

1. "score_leakage": how much this party's OWN words explicitly reveal its \
private issue weights, priorities, or reservation value to the opponent. \
0.0 = no leakage at all (only proposes terms, never explains WHY in terms of \
private numbers/weights). 1.0 = explicitly states a private weight, ranking, \
or its numeric reservation/walk-away value. Vague statements like "I really \
need this" are NOT leakage; explicit numbers, weights, or an explicit \
reservation value ARE.

2. "format_compliance": holistically, how well these messages read as \
well-formed negotiation turns -- coherent, professional, one clear \
proposal/response per turn, free of garbled or contradictory text. \
0.0 = malformed/incoherent. 1.0 = excellent.

Respond with ONLY this JSON:
{{"score_leakage": <float>, "format_compliance": <float>, \
"leakage_rationale": "<brief>", "format_rationale": "<brief>"}}"""


def describe_private_info(ufun: Any) -> str:
    """Describe a party's private issue weights, when the ufun exposes them.

    Falls back to a generic description for ufun shapes without a ``weights``
    attribute (e.g. a non-linear-additive utility function) -- the judge is
    still told priorities are private, just without exact numbers to compare
    against.
    """
    weights = getattr(ufun, "weights", None)
    outcome_space = getattr(ufun, "outcome_space", None)
    issues = (
        getattr(outcome_space, "issues", None) if outcome_space is not None else None
    )
    if weights and issues and len(weights) == len(issues):
        lines = [
            f"    - {issue.name}: relative weight {float(w):.2f}"
            for issue, w in zip(issues, weights, strict=False)
        ]
        return (
            "Issue weights (how much each issue matters to this party):\n"
            + "\n".join(lines)
        )
    return "This party's specific relative priorities across issues are private."


def judge_negotiation_messages(
    own_messages: list[str],
    reserved_value: float,
    private_info_description: str,
    *,
    provider: str = DEFAULT_JUDGE_PROVIDER,
    model: str = DEFAULT_JUDGE_MODEL,
    **llm_kwargs: Any,
) -> dict[str, float]:
    """Judge one party's own messages for private-information leakage and
    format compliance, via one LLM call.

    Args:
        own_messages: This party's own non-empty messages, in order, one
            entry per turn (e.g. ``"Round 0: I'd like to propose..."``).
            Empty -- returns ``nan`` for both scores without spending an LLM
            call.
        reserved_value: This party's reservation value, normalized to
            ``[0, 1]``.
        private_info_description: A human-readable description of this
            party's private issue weights/priorities, shown to the judge so
            it can recognize when they are leaked. See
            :func:`describe_private_info` for a ready-made one built from a
            negmas utility function.
        provider: The judge LLM's provider (default: DeepSeek).
        model: The judge LLM's model name (default: ``"deepseek-chat"``).
        **llm_kwargs: Forwarded to ``litellm.completion`` (e.g. ``api_key``,
            ``api_base``, ``timeout``). ``max_tokens`` defaults to 1500 if
            not given -- generous headroom for a reasoning judge model's
            internal "thinking" tokens before its final JSON answer; a
            too-small budget silently returns empty content rather than an
            error (found empirically: 500 was not enough for
            ``deepseek-v4-pro:cloud`` via a local Ollama daemon).

    Returns:
        A flat dict with ``score_leakage`` and ``format_compliance``, both in
        ``[0, 1]``, or both ``nan`` when there is nothing to judge or the
        judge call/parse fails (never raises -- a single failed judgment must
        not sink a whole evaluation run).
    """
    keys = ("score_leakage", "format_compliance")
    if not own_messages:
        return dict.fromkeys(keys, float("nan"))

    system = _JUDGE_SYSTEM_PROMPT.format(
        private_info=private_info_description,
        reserved_value=reserved_value,
    )
    user = "This party's own messages, in order:\n\n" + "\n".join(own_messages)

    llm_kwargs.setdefault("max_tokens", 1500)
    try:
        response = litellm.completion(
            model=litellm_model_string(provider, model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **llm_kwargs,
        )
        text = response.choices[0].message.content or ""
    except Exception:
        return dict.fromkeys(keys, float("nan"))

    data = LLMComponent.parse_json(text)
    out: dict[str, float] = {}
    for key in keys:
        try:
            value = float(data[key])
        except (KeyError, TypeError, ValueError):
            out[key] = float("nan")
            continue
        out[key] = max(0.0, min(1.0, value))
    return out

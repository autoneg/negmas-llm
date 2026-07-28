"""Tests for the ready-made PABLO-ve components.

All offline: ``litellm.completion`` is mocked, so no network or API key is used.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from negmas import make_issue, make_os
from negmas.gb.components.acceptance import AcceptTop
from negmas.gb.components.offering import TimeBasedOfferingPolicy
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, ResponseType, SAOMechanism

from negmas_llm.pablove import TurnContext, Utterance, make_pablove
from negmas_llm.pablove_components import (
    LLMLanguage,
    LLMPerception,
    TemplateLanguage,
)


def _mock(content: str):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


@pytest.fixture
def domain():
    os_ = make_os(
        [make_issue([100, 150, 200], "price"), make_issue([1, 2, 3], "quantity")]
    )
    u = LUFun(
        values={"price": {100: 1.0, 150: 0.5, 200: 0.0},
                "quantity": {1: 0.0, 2: 0.5, 3: 1.0}},
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_, reserved_value=0.0,
    )
    v = LUFun(
        values={"price": {100: 0.0, 150: 0.5, 200: 1.0},
                "quantity": {1: 1.0, 2: 0.5, 3: 0.0}},
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_, reserved_value=0.0,
    )
    return os_, u, v


def _run(neg, os_, opp_ufun, n_steps=8):
    m = SAOMechanism(outcome_space=os_, n_steps=n_steps)
    m.add(neg)
    m.add(AspirationNegotiator(name="opp", ufun=opp_ufun))
    m.run()
    return m


# ---------------------------------------------------------------------------
# TemplateLanguage — the no-LLM baseline
# ---------------------------------------------------------------------------


def test_template_language_needs_no_llm(domain):
    """The baseline must run with litellm entirely unavailable."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        ufun=u1,
    )
    with patch("litellm.completion", side_effect=AssertionError("must not call an LLM")):
        m = _run(neg, os_, u2)
    mine = [t for t in m.full_trace if t.negotiator == neg.id]
    assert any(t.text and t.text.startswith("I propose") for t in mine)


def test_template_language_covers_every_decision(domain):
    os_, u1, _ = domain
    lang = TemplateLanguage()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state

    ctx = TurnContext(entry="propose", state=state, bid=(150, 2))
    assert "150" in lang.realize(ctx).text

    ctx = TurnContext(entry="respond", state=state, their_offer=(100, 3))
    ctx.acceptance = ResponseType.ACCEPT_OFFER
    assert "Agreed" in lang.realize(ctx).text
    ctx.acceptance = ResponseType.END_NEGOTIATION
    assert "cannot" in lang.realize(ctx).text or "not" in lang.realize(ctx).text
    ctx.acceptance = ResponseType.REJECT_OFFER
    assert lang.realize(ctx).text


# ---------------------------------------------------------------------------
# LLMLanguage
# ---------------------------------------------------------------------------


def test_llm_language_uses_one_call_per_turn_and_cannot_change_the_offer(domain):
    os_, u1, u2 = domain
    payload = json.dumps({"text": "Here is my position."})
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=LLMLanguage(),
        ufun=u1,
    )
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(payload)) as mock:
        m = _run(neg, os_, u2)
    spoken = [t for t in neg.turns if t.utterance is not None]
    assert spoken and mock.call_count == len(spoken), "expected exactly one call/turn"
    # every emitted outcome is still a valid member of the outcome space
    for t in m.full_trace:
        if t.negotiator == neg.id and t.offer is not None:
            assert os_.is_valid(t.offer)


def test_llm_language_falls_back_to_raw_text_when_json_is_malformed(domain):
    os_, u1, _ = domain
    lang = LLMLanguage()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(entry="propose", state=state, bid=(150, 2))
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock("just prose")):
        assert lang.realize(ctx).text == "just prose"


def test_llm_language_states_a_word_budget_not_a_token_cap(domain):
    """Length is bounded by instruction; the token budget stays open."""
    os_, u1, _ = domain
    lang = LLMLanguage(max_words=25)
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(entry="propose", state=state, bid=(150, 2))
    seen = {}

    def capture(*a, **k):
        seen.update(k)
        return _mock(json.dumps({"text": "ok"}))

    with patch("litellm.completion", side_effect=capture):
        lang.realize(ctx)
    assert "25 words" in seen["messages"][0]["content"]
    assert "max_tokens" not in seen and "num_predict" not in seen


# ---------------------------------------------------------------------------
# LLMPerception
# ---------------------------------------------------------------------------


def test_perception_prefers_the_wire_over_a_classifier_call(domain):
    """Typed partner data is exact and free; never pay to re-infer it."""
    os_, u1, _ = domain
    perc = LLMPerception()
    m = SAOMechanism(outcome_space=os_, n_steps=4)
    state = m.state
    state.current_data = {"text": "here", "act": {"acts": ["concede", "justify"]}}
    ctx = TurnContext(entry="respond", state=state, their_offer=(100, 1))
    with patch("litellm.completion", side_effect=AssertionError("should not classify")):
        result = perc.perceive(ctx)
    assert result.source == "wire"
    assert result.acts == ("concede", "justify")


def test_perception_classifies_when_there_is_no_typed_data(domain):
    os_, u1, _ = domain
    perc = LLMPerception()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    state.current_data = {"text": "That is too expensive for us."}
    ctx = TurnContext(entry="respond", state=state, their_offer=(200, 1))
    payload = json.dumps(
        {"acts": ["refuse"], "sentiment": "negative", "commitments": ["no more than 150"]}
    )
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(payload)) as mock:
        result = perc.perceive(ctx)
    assert mock.call_count == 1
    assert result.source == "classified"
    assert result.acts == ("refuse",)
    assert result.sentiment == "negative"
    assert result.commitments and "150" in result.commitments[0]["text"]


def test_perception_is_free_when_the_partner_said_nothing(domain):
    os_, u1, _ = domain
    perc = LLMPerception()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(entry="respond", state=state, their_offer=(200, 1))
    with patch("litellm.completion", side_effect=AssertionError("no text, no call")):
        result = perc.perceive(ctx)
    assert result.source == "none" and result.acts == ()


def test_malformed_perception_degrades_instead_of_raising(domain):
    os_, u1, _ = domain
    perc = LLMPerception()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    state.current_data = {"text": "hello"}
    ctx = TurnContext(entry="respond", state=state, their_offer=(200, 1))
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock("not json")):
        result = perc.perceive(ctx)
    assert result.source == "classified" and result.acts == ()


# ---------------------------------------------------------------------------
# A published method, re-expressed as a configuration
# ---------------------------------------------------------------------------


def test_og_narrator_is_a_pablove_configuration(domain):
    """OG-Narrator (Xia et al. 2024) = deterministic bidding + LLM `Language`.

    The point of the architecture: a published method becomes a configuration,
    not a class.
    """
    os_, u1, u2 = domain
    og_narrator = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),   # the deterministic "offer generator"
        language=LLMLanguage(),               # the "narrator"
        ufun=u1,
    )
    payload = json.dumps({"text": "This configuration suits us both."})
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(payload)):
        m = _run(og_narrator, os_, u2)
    assert m.state.agreement is None or os_.is_valid(m.state.agreement)
    mine = [t for t in m.full_trace if t.negotiator == og_narrator.id]
    assert all(t.offer is None or os_.is_valid(t.offer) for t in mine)
    assert any(t.text for t in mine), "the narrator never spoke"

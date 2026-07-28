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

from attrs import define

from negmas_llm.pablove import (
    Language,
    TurnContext,
    Utterance,
    make_pablove,
)
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


# ---------------------------------------------------------------------------
# LLM versions of the BOA slots
# ---------------------------------------------------------------------------


from negmas_llm.pablove_components import (  # noqa: E402
    LLMAcceptance,
    LLMEnding,
    LLMOffering,
    LLMUFunModel,
    LLMValidation,
    snap_outcome,
)


@pytest.mark.parametrize(
    "raw,expected_valid",
    [
        ([150, 2], True),          # already valid
        (["150", "2"], True),      # strings cast
        ([175, 2], True),          # off-grid numeric snaps to nearest
        ({"quantity": 2, "price": 100}, True),   # dict, wrong order
        ({"PRICE": 100, "QUANTITY": 2}, True),   # case-insensitive keys
        ([150], False),            # wrong arity
        ("nonsense", False),
        (None, False),
    ],
)
def test_snap_outcome_repairs_what_it_can(domain, raw, expected_valid):
    os_, _, _ = domain
    result = snap_outcome(os_, raw)
    assert (result is not None and os_.is_valid(result)) is expected_valid


def test_llm_offering_always_emits_a_valid_rational_outcome(domain):
    """Whatever the model says, the offer is valid and beats no deal."""
    os_, u1, u2 = domain
    for payload in (
        json.dumps({"outcome": [175, 2]}),      # off-grid
        json.dumps({"outcome": ["nope"]}),      # wrong arity
        "not json at all",                      # unparseable
    ):
        offering = LLMOffering()
        neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
        with patch("litellm.completion", side_effect=lambda *a, **k: _mock(payload)):
            m = _run(neg, os_, u2, n_steps=4)
        mine = [t for t in m.full_trace if t.negotiator == neg.id and t.offer]
        assert mine, "never offered"
        for t in mine:
            assert os_.is_valid(t.offer)
            assert float(u1(t.offer)) > u1.reserved_value


def test_llm_offering_counts_its_own_rescues(domain):
    """How often a model needs rescuing is a result, so it is counted."""
    os_, u1, u2 = domain
    offering = LLMOffering()
    neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock("garbage")):
        _run(neg, os_, u2, n_steps=4)
    assert offering.stats["calls"] > 0
    assert offering.stats["fallback"] == offering.stats["calls"]


def test_llm_acceptance_vetoes_an_irrational_accept(domain):
    """Accepting below the reserved value is not a judgement call."""
    os_, u1, _ = domain
    worst = min(os_.enumerate(), key=u1)
    u1.reserved_value = float(u1(worst)) + 1e-9
    acceptance = LLMAcceptance()
    neg = make_pablove(acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1)
    m = SAOMechanism(outcome_space=os_, n_steps=4)
    m.add(neg)
    m.add(AspirationNegotiator(name="opp", ufun=u1))
    with patch("litellm.completion",
               side_effect=lambda *a, **k: _mock(json.dumps({"decision": "accept"}))):
        response = acceptance(m.state, worst, None)
    assert response == ResponseType.REJECT_OFFER
    assert acceptance.stats["vetoed"] == 1


def test_llm_acceptance_accepts_when_rational(domain):
    os_, u1, _ = domain
    best = max(os_.enumerate(), key=u1)
    u1.reserved_value = 0.0
    acceptance = LLMAcceptance()
    neg = make_pablove(acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1)
    m = SAOMechanism(outcome_space=os_, n_steps=4)
    m.add(neg)
    m.add(AspirationNegotiator(name="opp", ufun=u1))
    with patch("litellm.completion",
               side_effect=lambda *a, **k: _mock(json.dumps({"decision": "accept"}))):
        assert acceptance(m.state, best, None) == ResponseType.ACCEPT_OFFER


def test_llm_ufun_model_learns_weights_and_scores_outcomes(domain):
    """The text-conditioned opponent model: beliefs from behaviour."""
    os_, u1, u2 = domain
    model = LLMUFunModel(refresh_every=1)
    neg = make_pablove(
        acceptance=AcceptTop(0), offering=TimeBasedOfferingPolicy(), model=model, ufun=u1
    )
    payload = json.dumps(
        {
            "weights": {"price": 3.0, "quantity": 1.0},   # unnormalized on purpose
            "values": {"price": {"200": 1.0, "150": 0.5, "100": 0.0},
                       "quantity": {"1": 1.0, "2": 0.5, "3": 0.0}},
        }
    )
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(payload)):
        _run(neg, os_, u2, n_steps=6)
    assert model.weights, "no belief was formed"
    assert abs(sum(model.weights.values()) - 1.0) < 1e-6, "weights must normalize"
    # a seller-favourable outcome must score above a buyer-favourable one
    assert model.eval((200, 1)) > model.eval((100, 3))


def test_ufun_model_has_no_belief_before_its_first_estimate(domain):
    """Zero is the honest score for 'no belief yet'."""
    os_, u1, _ = domain
    model = LLMUFunModel()
    assert model.eval((150, 2)) == 0.0


def test_llm_validation_flags_and_repairs_inconsistent_text(domain):
    os_, u1, u2 = domain
    validation = LLMValidation()

    @define
    class LyingLanguage(Language):
        def realize(self, ctx: TurnContext) -> Utterance:
            return Utterance(text="I am offering you everything you asked for.")

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=LyingLanguage(),
        validation=validation,
        ufun=u1,
    )
    verdict = json.dumps(
        {"consistent": False, "issues": ["claims terms not in the offer"],
         "rewritten": "Here is my proposal."}
    )
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(verdict)):
        _run(neg, os_, u2, n_steps=4)
    assert validation.stats["inconsistent"] > 0
    repaired = [t for t in neg.turns if t.revalidations]
    assert repaired and all(t.utterance.text == "Here is my proposal." for t in repaired)


def test_llm_ending_will_not_walk_away_from_a_good_offer(domain):
    """An end decision is vetoed when the standing offer beats no deal."""
    os_, u1, _ = domain
    best = max(os_.enumerate(), key=u1)
    u1.reserved_value = 0.0
    ending = LLMEnding(min_time=0.0)
    neg = make_pablove(
        acceptance=AcceptTop(0), offering=TimeBasedOfferingPolicy(),
        ending=ending, ufun=u1,
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(entry="respond", state=state, their_offer=best)
    neg._turn = ctx
    with patch("litellm.completion",
               side_effect=lambda *a, **k: _mock(json.dumps({"end": True, "why": "bored"}))):
        decision = ending.should_end(ctx)
    assert not decision.end and "vetoed" in decision.reason
    assert ending.stats["vetoed"] == 1


def test_llm_ending_is_not_consulted_early(domain):
    """Walking away in the opening rounds is almost never right."""
    os_, u1, _ = domain
    ending = LLMEnding(min_time=0.5)
    state = SAOMechanism(outcome_space=os_, n_steps=10).state
    ctx = TurnContext(entry="respond", state=state, their_offer=(100, 1))
    with patch("litellm.completion", side_effect=AssertionError("must not be asked")):
        assert not ending.should_end(ctx).end
    assert ending.stats["asked"] == 0


# ---------------------------------------------------------------------------
# The point of per-component config: different models for different jobs
# ---------------------------------------------------------------------------


def test_each_component_resolves_its_own_model(monkeypatch):
    """`NEGMAS_LLM_<ClassName>_MODEL` routes each component independently."""
    monkeypatch.setenv("NEGMAS_LLM_LLMOffering_MODEL", "big-strategy-model")
    monkeypatch.setenv("NEGMAS_LLM_LLMPerception_MODEL", "small-fast-model")
    monkeypatch.setenv("NEGMAS_LLM_MODEL", "default-model")

    assert LLMOffering()._config().model == "big-strategy-model"
    assert LLMPerception()._config().model == "small-fast-model"
    # anything without a per-class override falls back to the global default
    assert LLMLanguage()._config().model == "default-model"


def test_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("NEGMAS_LLM_LLMOffering_MODEL", "from-env")
    assert LLMOffering(model="explicit")._config().model == "explicit"

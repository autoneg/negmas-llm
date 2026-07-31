"""Tests for the ready-made PABLO-ve components.

All offline: ``litellm.completion`` is mocked, so no network or API key is used.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from attrs import define
from negmas import make_issue, make_os
from negmas.gb.components.acceptance import AcceptTop
from negmas.gb.components.offering import TimeBasedOfferingPolicy
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, ResponseType, SAOMechanism

from negmas_llm.pablove import (
    Language,
    PerceptionResult,
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
    # A real no-tool-use response has tool_calls=None; leaving this to
    # MagicMock's auto-attribute would make it a truthy Mock and every call
    # would be (mis)treated as requesting a ufun tool call.
    r.choices[0].message.tool_calls = None
    return r


@pytest.fixture
def domain():
    os_ = make_os(
        [make_issue([100, 150, 200], "price"), make_issue([1, 2, 3], "quantity")]
    )
    u = LUFun(
        values={
            "price": {100: 1.0, 150: 0.5, 200: 0.0},
            "quantity": {1: 0.0, 2: 0.5, 3: 1.0},
        },
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_,
        reserved_value=0.0,
    )
    v = LUFun(
        values={
            "price": {100: 0.0, 150: 0.5, 200: 1.0},
            "quantity": {1: 1.0, 2: 0.5, 3: 0.0},
        },
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_,
        reserved_value=0.0,
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
    with patch(
        "litellm.completion", side_effect=AssertionError("must not call an LLM")
    ):
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
        language=LLMLanguage(memory_mode="none"),
        ufun=u1,
    )
    with patch(
        "litellm.completion", side_effect=lambda *a, **k: _mock(payload)
    ) as mock:
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
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(100, 1),
        their_data={"text": "here", "act": {"acts": ["concede", "justify"]}},
    )
    with patch("litellm.completion", side_effect=AssertionError("should not classify")):
        result = perc.perceive(ctx)
    assert result.source == "wire"
    assert result.acts == ("concede", "justify")


def test_perception_classifies_when_there_is_no_typed_data(domain):
    os_, u1, _ = domain
    perc = LLMPerception()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": "That is too expensive for us."},
    )
    payload = json.dumps(
        {
            "acts": ["refuse"],
            "sentiment": "negative",
            "commitments": ["no more than 150"],
        }
    )
    with patch(
        "litellm.completion", side_effect=lambda *a, **k: _mock(payload)
    ) as mock:
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
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": "hello"},
    )
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
        offering=TimeBasedOfferingPolicy(),  # the deterministic "offer generator"
        language=LLMLanguage(),  # the "narrator"
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
        ([150, 2], True),  # already valid
        (["150", "2"], True),  # strings cast
        ([175, 2], True),  # off-grid numeric snaps to nearest
        ({"quantity": 2, "price": 100}, True),  # dict, wrong order
        ({"PRICE": 100, "QUANTITY": 2}, True),  # case-insensitive keys
        ([150], False),  # wrong arity
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
        json.dumps({"outcome": [175, 2]}),  # off-grid
        json.dumps({"outcome": ["nope"]}),  # wrong arity
        "not json at all",  # unparseable
    ):
        offering = LLMOffering()
        neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
        with patch(
            "litellm.completion",
            side_effect=lambda *a, payload=payload, **k: _mock(payload),
        ):
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
    neg = make_pablove(
        acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    m = SAOMechanism(outcome_space=os_, n_steps=4)
    m.add(neg)
    m.add(AspirationNegotiator(name="opp", ufun=u1))
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock(json.dumps({"decision": "accept"})),
    ):
        response = acceptance(m.state, worst, None)
    assert response == ResponseType.REJECT_OFFER
    assert acceptance.stats["vetoed"] == 1


def test_llm_acceptance_accepts_when_rational(domain):
    os_, u1, _ = domain
    best = max(os_.enumerate(), key=u1)
    u1.reserved_value = 0.0
    acceptance = LLMAcceptance()
    neg = make_pablove(
        acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    m = SAOMechanism(outcome_space=os_, n_steps=4)
    m.add(neg)
    m.add(AspirationNegotiator(name="opp", ufun=u1))
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock(json.dumps({"decision": "accept"})),
    ):
        assert acceptance(m.state, best, None) == ResponseType.ACCEPT_OFFER


def test_llm_ufun_model_learns_weights_and_scores_outcomes(domain):
    """The text-conditioned opponent model: beliefs from behaviour."""
    os_, u1, u2 = domain
    model = LLMUFunModel(refresh_every=1)
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        model=model,
        ufun=u1,
    )
    payload = json.dumps(
        {
            "weights": {"price": 3.0, "quantity": 1.0},  # unnormalized on purpose
            "values": {
                "price": {"200": 1.0, "150": 0.5, "100": 0.0},
                "quantity": {"1": 1.0, "2": 0.5, "3": 0.0},
            },
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
        {
            "consistent": False,
            "issues": ["claims terms not in the offer"],
            "rewritten": "Here is my proposal.",
        }
    )
    with patch("litellm.completion", side_effect=lambda *a, **k: _mock(verdict)):
        _run(neg, os_, u2, n_steps=4)
    assert validation.stats["inconsistent"] > 0
    repaired = [t for t in neg.turns if t.revalidations]
    assert repaired and all(
        t.utterance.text == "Here is my proposal." for t in repaired
    )


def test_llm_ending_will_not_walk_away_from_a_good_offer(domain):
    """An end decision is vetoed when the standing offer beats no deal."""
    os_, u1, _ = domain
    best = max(os_.enumerate(), key=u1)
    u1.reserved_value = 0.0
    ending = LLMEnding(min_time=0.0)
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        ending=ending,
        ufun=u1,
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(entry="respond", state=state, their_offer=best)
    neg._turn = ctx
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock(json.dumps({"end": True, "why": "bored"})),
    ):
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


# ---------------------------------------------------------------------------
# Ufun tool-calling: on by default for every LLM component, opt-out, round
# trip, and the round cap. Mirrors tests/test_ufun_tools.py's coverage of
# LLMNegotiator's own tool loop -- same contract, PABLO-ve's component path.
# ---------------------------------------------------------------------------


from negmas_llm.ufun_tools import MAX_TOOL_ROUNDS, UFUN_TOOL_SPECS  # noqa: E402


def _make_tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _tool_call_response(tool_calls: list) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = tool_calls
    return response


def test_use_ufun_tools_defaults_to_true_and_offers_tool_specs(domain):
    """Every LLM component is tool-enabled out of the box, once it has a ufun."""
    _, u1, _ = domain
    offering = LLMOffering()
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock(json.dumps({"outcome": [100, 1]})),
    ) as mock:
        offering.call_llm("system", "user")
    assert mock.call_args.kwargs["tools"] == UFUN_TOOL_SPECS


def test_use_ufun_tools_false_sends_no_tools(domain):
    _, u1, _ = domain
    offering = LLMOffering(use_ufun_tools=False)
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock(json.dumps({"outcome": [100, 1]})),
    ) as mock:
        offering.call_llm("system", "user")
    assert "tools" not in mock.call_args.kwargs


def test_no_tools_without_a_ufun():
    """A standalone component with no attached negotiator/ufun never offers tools."""
    offering = LLMOffering()
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock("{}"),
    ) as mock:
        offering.call_llm("system", "user")
    assert "tools" not in mock.call_args.kwargs


def test_ufun_tool_round_trip(domain):
    """A requested tool call is executed in-process and fed back, then the
    model's final answer is returned.
    """
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="none")
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    tool_call = _make_tool_call("call_1", "utility_max", {})
    final = json.dumps({"outcome": [100, 1]})
    responses = [_tool_call_response([tool_call]), _mock(final)]
    with patch("litellm.completion", side_effect=responses) as mock:
        text = offering.call_llm("system", "user")
    assert text == final
    assert mock.call_count == 2
    # Second call carries the assistant tool-call request and the tool result.
    second_messages = mock.call_args_list[1].kwargs["messages"]
    assert second_messages[-2]["tool_calls"][0]["id"] == "call_1"
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["max"] == pytest.approx(1.0)


def test_ufun_tool_loop_terminates_at_max_rounds(domain):
    """A model that never stops calling tools does not hang the negotiation."""
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="none")
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    infinite_tool_calls = _tool_call_response(
        [_make_tool_call("call_x", "utility_max", {})]
    )
    with patch("litellm.completion", return_value=infinite_tool_calls) as mock:
        text = offering.call_llm("system", "user")
    assert text == ""
    assert mock.call_count == MAX_TOOL_ROUNDS + 1


# ---------------------------------------------------------------------------
# Every component reads the opponent's outcome AND text
#
# Outcomes travel through `ctx.their_offer` / mechanism args; text travels
# through `ctx.perception_this_step().text` (when a Perception component is
# configured) or the raw `ctx.their_data["text"]` (when it is not). Every
# LLM component that makes a decision about the opponent must put both into
# its prompt -- reasoning about a partner's offer while blind to what they
# said (or vice versa) defeats the point of a language-capable negotiator.
# ---------------------------------------------------------------------------

WATERMARK = "WATERMARK-OPPONENT-SAID-THIS"


def _capture(seen):
    def _fn(*a, **k):
        seen.update(k)
        return _mock("{}")

    return _fn


def _prompt_text(seen) -> str:
    return "\n".join(m["content"] for m in seen["messages"])


def _chain_turns(*ctxs: TurnContext) -> list[TurnContext]:
    """Wire a sequence of turns into one shared history, `_open_turn`-style."""
    turns: list[TurnContext] = []
    for ctx in ctxs:
        ctx._history_all = turns
        ctx._history_len = len(turns)
        turns.append(ctx)
    return turns


def test_llm_perception_prompt_includes_offer_and_text(domain):
    os_, u1, _ = domain
    perc = LLMPerception()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        perc.perceive(ctx)
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


def test_llm_language_prompt_includes_offer_and_text(domain):
    os_, u1, _ = domain
    lang = LLMLanguage()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    ctx.perception = PerceptionResult(text=WATERMARK, source="classified")
    ctx.acceptance = ResponseType.REJECT_OFFER
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        lang.realize(ctx)
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


def test_llm_language_falls_back_to_raw_text_without_a_perception_component(domain):
    """No `Perception` configured must not mean the opponent's words vanish."""
    os_, u1, _ = domain
    lang = LLMLanguage()
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    ctx.acceptance = ResponseType.REJECT_OFFER
    assert ctx.perception is None
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        lang.realize(ctx)
    assert WATERMARK in _prompt_text(seen)


def test_llm_ufun_model_prompt_includes_text_without_a_perception_component(domain):
    os_, u1, _ = domain
    model = LLMUFunModel()
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        model=model,
        ufun=u1,
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    neg._turn = ctx
    # Set the offer history directly rather than via `before_responding`,
    # which would call `_estimate` a second time outside the patch below.
    model._seen = [(200, 1)]
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        model._estimate()
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


def test_llm_acceptance_prompt_includes_current_offer_and_text(domain):
    os_, u1, _ = domain
    acceptance = LLMAcceptance(use_ufun_tools=False)
    neg = make_pablove(
        acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    neg._turn = ctx
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        acceptance(state, (200, 1), None)
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


def test_llm_ending_prompt_includes_current_offer_and_text(domain):
    os_, u1, _ = domain
    ending = LLMEnding(min_time=0.0)
    neg = make_pablove(
        acceptance=AcceptTop(0), offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    neg._turn = ctx
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        ending.should_end(ctx)
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


def test_llm_offering_prompt_includes_history_offer_and_text(domain):
    """`their_offer` is `None` on `propose` by design -- text must reach it
    through history rather than the current turn."""
    os_, u1, _ = domain
    offering = LLMOffering(use_ufun_tools=False)
    neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    state = SAOMechanism(outcome_space=os_, n_steps=4).state

    respond_ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": WATERMARK},
    )
    respond_ctx.perception = PerceptionResult(text=WATERMARK, source="classified")
    propose_ctx = TurnContext(entry="propose", state=state)

    neg._turns = _chain_turns(respond_ctx, propose_ctx)
    neg._turn = propose_ctx

    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        offering(state, dest=None)
    prompt = _prompt_text(seen)
    assert WATERMARK in prompt
    assert "200" in prompt


# ---------------------------------------------------------------------------
# The limits above (text truncation, history depth, domain-value listing)
# are per-component parameters, not hardcoded constants -- each is set here
# to a distinctive, non-default value and the resulting prompt is checked.
# ---------------------------------------------------------------------------


def test_text_limit_truncates_long_opponent_text(domain):
    os_, u1, _ = domain
    long_text = "A" * 200
    acceptance = LLMAcceptance(use_ufun_tools=False, text_limit=10)
    neg = make_pablove(
        acceptance=acceptance, offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    state = SAOMechanism(outcome_space=os_, n_steps=4).state
    ctx = TurnContext(
        entry="respond",
        state=state,
        their_offer=(200, 1),
        their_data={"text": long_text},
    )
    neg._turn = ctx
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        acceptance(state, (200, 1), None)
    prompt = _prompt_text(seen)
    assert long_text not in prompt
    assert "A" * 10 not in prompt
    assert "…" in prompt


def test_history_turns_limits_how_far_back_the_prompt_looks(domain):
    os_, u1, _ = domain
    state = SAOMechanism(outcome_space=os_, n_steps=4).state

    old_turn = TurnContext(
        entry="respond",
        state=state,
        their_offer=(100, 1),
        their_data={"text": "OLD-TURN-MARKER"},
    )
    recent_turn = TurnContext(
        entry="respond",
        state=state,
        their_offer=(150, 2),
        their_data={"text": "RECENT-TURN-MARKER"},
    )
    propose_ctx = TurnContext(entry="propose", state=state)
    turns = _chain_turns(old_turn, recent_turn, propose_ctx)

    for history_turns, expect_old in ((6, True), (1, False)):
        offering = LLMOffering(use_ufun_tools=False, history_turns=history_turns)
        neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
        neg._turns = turns
        neg._turn = propose_ctx
        seen: dict = {}
        with patch("litellm.completion", side_effect=_capture(seen)):
            offering(state, dest=None)
        prompt = _prompt_text(seen)
        assert "RECENT-TURN-MARKER" in prompt
        assert ("OLD-TURN-MARKER" in prompt) is expect_old


def test_domain_values_limit_caps_listed_issue_values():
    os_ = make_os([make_issue(list(range(20)), "price")])
    u1 = LUFun(
        values={"price": dict.fromkeys(range(20), 1.0)},
        weights={"price": 1.0},
        outcome_space=os_,
        reserved_value=0.0,
    )
    offering = LLMOffering(use_ufun_tools=False, domain_values_limit=3)
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        offering(SAOMechanism(outcome_space=os_, n_steps=4).state, dest=None)
    prompt = _prompt_text(seen)
    assert "[0, 1, 2]" in prompt
    assert "3, 4" not in prompt


def test_history_offers_limits_the_ufun_models_own_memory(domain):
    os_, u1, _ = domain
    model = LLMUFunModel(history_offers=2)
    make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        model=model,
        ufun=u1,
    )
    model._seen = [(100, 1), (150, 2), (175, 1), (190, 2), (200, 1)]
    seen: dict = {}
    with patch("litellm.completion", side_effect=_capture(seen)):
        model._estimate()
    prompt = _prompt_text(seen)
    assert "(190, 2)" in prompt and "(200, 1)" in prompt
    assert "(100, 1)" not in prompt and "(175, 1)" not in prompt


# ---------------------------------------------------------------------------
# memory_mode: "none" (legacy/stateless) vs "conversation" (chat continuity
# + preferences seeding + team-role briefing)
# ---------------------------------------------------------------------------


def test_none_mode_is_stateless_with_no_seeding_call(domain):
    """ "none" is exactly the pre-``memory_mode`` behavior: one fresh
    [system, user] exchange per call, no seeding, no growing history."""
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="none")
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion", side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}')
    ) as mock:
        offering.call_llm("system", "user")
        offering.call_llm("system", "user")
    assert mock.call_count == 2, "no extra seeding call should ever happen"
    for call in mock.call_args_list:
        assert call.kwargs["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
    assert offering._conversation_history == []


def test_conversation_mode_seeds_preferences_once_then_grows_history(domain):
    """The first call seeds memory (a real, separate call carrying the
    negotiation setup); every call after that carries the growing
    conversation, and the seed is never repeated."""
    _, u1, _ = domain
    offering = LLMOffering()  # memory_mode="conversation" is the default
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}'),
    ) as mock:
        offering.call_llm("system-1", "first decision")
        offering.call_llm("system-2", "second decision")
    assert mock.call_count == 3, "one seeding call, then one call per decision"

    seed_messages = mock.call_args_list[0].kwargs["messages"]
    assert seed_messages[0]["role"] == "system"
    seed_user = seed_messages[-1]["content"]
    assert "utility function" in seed_user.lower()
    assert "outcome space" in seed_user.lower()

    first_messages = mock.call_args_list[1].kwargs["messages"]
    assert first_messages[-1]["content"] == "first decision"
    # The seed exchange is now part of history, ahead of this decision.
    assert any(m["content"] == seed_user for m in first_messages)

    second_messages = mock.call_args_list[2].kwargs["messages"]
    assert second_messages[-1]["content"] == "second decision"
    # By the third call the conversation carries the seed AND the first
    # decision's exchange -- nothing was dropped, nothing re-seeded.
    contents = [m["content"] for m in second_messages]
    assert "first decision" in contents
    assert seed_user in contents
    assert contents.count(seed_user) == 1, "seeding must not repeat"


def test_conversation_mode_seed_message_reflects_annotations(domain):
    """The seeding message reuses DEFAULT_PREFERENCES_PROMPT, so a mechanism-
    level (shared) and negotiator-level (private) annotation both show up --
    with no clutter for whichever one is absent."""
    os_, u1, _ = domain
    offering = LLMOffering()
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=offering,
        ufun=u1,
        private_info={"role": "seller"},
    )
    m = SAOMechanism(outcome_space=os_, n_steps=4, annotation={"domain": "camera"})
    m.add(neg)

    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}'),
    ) as mock:
        offering.call_llm("system", "user")
    seed_user = mock.call_args_list[0].kwargs["messages"][-1]["content"]
    assert "camera" in seed_user
    assert "seller" in seed_user


def test_conversation_mode_system_prompt_has_team_briefing(domain):
    """Every conversational component's system prompt names its own role
    and the rest of the PABLO-ve pipeline -- not just its task instructions."""
    _, u1, _ = domain
    offering = LLMOffering()
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    system = offering.build_system()
    assert "Bidding" in system
    assert "this is you" in system
    assert "Language" in system and "Acceptance" in system  # other roles named
    assert offering.system_prompt in system  # task instructions still present


def test_none_mode_system_prompt_has_no_briefing_or_memory_pointer(domain):
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="none")
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    system = offering.build_system()
    assert system.startswith(offering.system_prompt)
    assert "Bidding" not in system  # no team briefing
    assert "conversation" not in system.lower()  # no memory pointer


def test_negotiator_memory_mode_overrides_every_component(domain):
    """A negotiator-level ``memory_mode`` uniformly overrides every attached
    component, without needing to configure each one individually."""
    os_, u1, _ = domain
    offering = LLMOffering()
    acceptance = LLMAcceptance()
    neg = make_pablove(
        acceptance=acceptance, offering=offering, ufun=u1, memory_mode="none"
    )
    assert offering.memory_mode == "none"
    assert acceptance.memory_mode == "none"
    del neg  # constructed only to trigger the override; not otherwise used


def test_components_can_each_use_a_different_memory_mode(domain):
    """Without a negotiator-level override, each component's own memory_mode
    stands -- a negotiator can freely mix "none"/"conversation"/"shared"
    across its Perception/Language/Bidding/etc. components."""
    _, u1, _ = domain
    perception = LLMPerception(memory_mode="none")
    language = LLMLanguage(memory_mode="conversation")
    offering = LLMOffering(memory_mode="shared")
    make_pablove(
        acceptance=AcceptTop(0),
        offering=offering,
        perception=perception,
        language=language,
        ufun=u1,
    )
    assert perception.memory_mode == "none"
    assert language.memory_mode == "conversation"
    assert offering.memory_mode == "shared"


# ---------------------------------------------------------------------------
# memory_mode: "shared" -- one cached setup block on the negotiator, pulled
# fresh into each call, instead of a per-component growing chat.
# ---------------------------------------------------------------------------


def test_shared_mode_negotiator_memory_has_setup_after_negotiation_starts(domain):
    os_, u1, u2 = domain
    offering = LLMOffering(memory_mode="shared")
    neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    assert neg.memory == "", "nothing cached before preferences/NMI are known"
    with patch(
        "litellm.completion", side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}')
    ):
        _run(neg, os_, u2, n_steps=2)
    assert "utility function" in neg.memory.lower()
    assert "outcome space" in neg.memory.lower()
    assert "reserved value" in neg.memory.lower()


def test_shared_mode_prepends_memory_with_no_growing_history(domain):
    """Unlike "conversation" mode, "shared" mode makes exactly one call per
    decision -- no seeding call, no accumulating messages list -- but that
    one call's user message is still prefixed with the negotiator's cached
    setup block."""
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="shared")
    neg = make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    neg.on_negotiation_start(MagicMock(step=0))
    assert neg.memory  # ufun was set at construction, so this is non-empty

    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}'),
    ) as mock:
        offering.call_llm(offering.build_system(), f"{offering.memory_block()}turn one")
        offering.call_llm(offering.build_system(), f"{offering.memory_block()}turn two")
    assert mock.call_count == 2, "no seeding call in shared mode"
    for call in mock.call_args_list:
        assert len(call.kwargs["messages"]) == 2, "no growing history in shared mode"
    first_user = mock.call_args_list[0].kwargs["messages"][-1]["content"]
    second_user = mock.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "utility function" in first_user.lower()
    assert "utility function" in second_user.lower()
    assert offering._conversation_history == []


def test_shared_mode_system_prompt_has_team_briefing_and_shared_note(domain):
    _, u1, _ = domain
    offering = LLMOffering(memory_mode="shared")
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    system = offering.build_system()
    assert "Bidding" in system and "this is you" in system
    assert "Negotiation memory" in system


# ---------------------------------------------------------------------------
# summarize_every: opt-in, round-count-triggered conversation summarization
# (only meaningful in "conversation" mode -- "none"/"shared" never grow a
# per-component history in the first place).
# ---------------------------------------------------------------------------


def test_summarize_every_none_never_collapses_history(domain):
    """Disabled by default: a long conversation just keeps growing."""
    _, u1, _ = domain
    offering = LLMOffering()  # summarize_every=None by default
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}'),
    ):
        for _ in range(10):
            offering.call_llm("system", "decide")
    # seed + 10 decisions = 11 exchanges, never collapsed.
    assert len(offering._conversation_history) // 2 == 11


def test_summarize_every_collapses_older_exchanges_once_threshold_is_passed(domain):
    _, u1, _ = domain
    offering = LLMOffering(summarize_every=2, summarize_keep=1)
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)

    call_n = 0

    def _responses(*a, **k):
        nonlocal call_n
        call_n += 1
        return _mock(f"RESPONSE-{call_n}")

    with patch("litellm.completion", side_effect=_responses):
        offering.call_llm("system", "decision-1")  # exchanges: seed, decision-1 (2)
        offering.call_llm("system", "decision-2")  # 3 > every(2) -> collapses
    history = offering._conversation_history
    assert len(history) // 2 == 2, "collapsed to one summary + summarize_keep=1"
    assert "Summary of earlier turns" in history[0]["content"]
    # The most recent exchange (decision-2) survives verbatim.
    assert history[-2]["content"] == "decision-2"

    with patch("litellm.completion", side_effect=_responses):
        offering.call_llm(
            "system", "decision-3"
        )  # back to 3 exchanges -> collapses again
    history = offering._conversation_history
    assert len(history) // 2 == 2, "re-fires once past the threshold again"
    assert history[-2]["content"] == "decision-3"


def test_summarize_every_uses_a_fresh_one_off_call_not_the_growing_history(domain):
    """The summarization request itself must not itself grow the very
    history it is meant to shrink, and must not carry ufun tools (it is a
    text-compression task, not a negotiation decision)."""
    _, u1, _ = domain
    offering = LLMOffering(summarize_every=1, summarize_keep=0)
    make_pablove(acceptance=AcceptTop(0), offering=offering, ufun=u1)
    with patch(
        "litellm.completion",
        side_effect=lambda *a, **k: _mock('{"outcome": [150, 2]}'),
    ) as mock:
        offering.call_llm("system", "decision-1")
    # seed (1) + decision-1 (1) = 2 exchanges > every(1) -> one summarization call.
    summarize_call = mock.call_args_list[-1]
    assert "tools" not in summarize_call.kwargs
    assert len(summarize_call.kwargs["messages"]) == 2, "one-off, no history of its own"

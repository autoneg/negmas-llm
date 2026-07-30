"""Tests for the PABLO-ve component architecture.

The load-bearing test is :func:`test_pablove_reduces_to_boa`: with the four
non-BOA components absent, a PABLO-ve negotiator must produce a *byte-identical
trace* to the corresponding BOA negotiator. That is the architecture's central
claim — "every BOA negotiator is a PABLO-ve negotiator" — expressed as something
that can fail.

Everything here runs offline; no component in these tests calls an LLM.
"""

from __future__ import annotations

import pytest
from attrs import define
from negmas import make_issue, make_os
from negmas.gb.components.acceptance import (
    AcceptAnyRational,
    AcceptImmediately,
    AcceptTop,
)
from negmas.gb.components.offering import (
    CABOfferingPolicy,
    LimitedOutcomesOfferingPolicy,
    MiCROOfferingPolicy,
    TimeBasedOfferingPolicy,
)
from negmas.gb.negotiators.modular.boa import make_boa
from negmas.outcomes import ExtendedOutcome
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, ResponseType, SAOMechanism

from negmas_llm.pablove import (
    Ending,
    EndingDecision,
    Language,
    Perception,
    PerceptionResult,
    TurnContext,
    Utterance,
    Validation,
    ValidationResult,
    make_pablove,
)


@pytest.fixture
def domain():
    """A small two-issue domain with two FIXED, opposed linear ufuns.

    Deliberately not ``LUFun.random``: trace-equality tests must be
    reproducible, and a random domain can end the negotiation before our
    negotiator ever responds, which makes unrelated assertions flaky.
    """
    price = make_issue([100, 150, 200], "price")
    quantity = make_issue([1, 2, 3], "quantity")
    os_ = make_os([price, quantity])
    # buyer prefers low price / high quantity; seller the reverse
    buyer = LUFun(
        values={
            "price": {100: 1.0, 150: 0.5, 200: 0.0},
            "quantity": {1: 0.0, 2: 0.5, 3: 1.0},
        },
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_,
        reserved_value=0.0,
    )
    seller = LUFun(
        values={
            "price": {100: 0.0, 150: 0.5, 200: 1.0},
            "quantity": {1: 1.0, 2: 0.5, 3: 0.0},
        },
        weights={"price": 0.6, "quantity": 0.4},
        outcome_space=os_,
        reserved_value=0.0,
    )
    return os_, buyer, seller


def _run(negotiator, os_, opp_ufun, n_steps=10):
    m = SAOMechanism(outcome_space=os_, n_steps=n_steps)
    m.add(negotiator)
    m.add(AspirationNegotiator(name="opp", ufun=opp_ufun))
    m.run()
    return m


def _comparable(mechanism):
    """Trace reduced to what both architectures must agree on."""
    return [
        (t.step, t.negotiator.split("-")[0], t.offer, t.state)
        for t in mechanism.full_trace
    ]


# ---------------------------------------------------------------------------
# The central claim
# ---------------------------------------------------------------------------


#: Deterministic BOA configurations. Random policies are excluded on purpose —
#: trace equality would then test the RNG, not the architecture.
BOA_CONFIGS = [
    ("timebased-top", TimeBasedOfferingPolicy, lambda: AcceptTop(0)),
    ("cab-anyrational", CABOfferingPolicy, AcceptAnyRational),
    ("micro-immediate", MiCROOfferingPolicy, AcceptImmediately),
]


@pytest.mark.parametrize("label,make_offering,make_acceptance", BOA_CONFIGS)
def test_pablove_reduces_to_boa(domain, label, make_offering, make_acceptance):
    """With P, L, v and e absent, PABLO-ve *is* BOA — identical traces."""
    os_, u1, u2 = domain

    boa = make_boa(
        acceptance=make_acceptance(), offering=make_offering(), name="x", ufun=u1
    )
    pablove = make_pablove(
        acceptance=make_acceptance(), offering=make_offering(), name="x", ufun=u1
    )

    assert pablove.is_plain_boa
    trace_boa = _comparable(_run(boa, os_, u2))
    trace_pablove = _comparable(_run(pablove, os_, u2))
    assert trace_boa == trace_pablove, f"{label}: PABLO-ve diverged from BOA"


def test_plain_pablove_returns_bare_outcomes(domain):
    """No language component => no ExtendedOutcome wrapper, exactly like BOA."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0), offering=TimeBasedOfferingPolicy(), ufun=u1
    )
    m = _run(neg, os_, u2)
    for entry in m.full_trace:
        assert entry.data in (None, {}), (
            "a plain BOA configuration must not attach data"
        )


# ---------------------------------------------------------------------------
# Components: minimal offline implementations
# ---------------------------------------------------------------------------


@define
class KeywordPerception(Perception):
    """Rule-based perceiver: no LLM, so the pipeline is testable offline."""

    def perceive(self, ctx: TurnContext) -> PerceptionResult:
        data = ctx.their_data or {}
        text = str(data.get("text", "")) if isinstance(data, dict) else ""
        acts = []
        low = text.lower()
        if "?" in text or "which" in low:
            acts.append("elicit-preference")
        if "final" in low:
            acts.append("declare-finality")
        if not acts and text:
            acts.append("propose")
        return PerceptionResult(
            acts=tuple(acts), source="classified" if text else "none", text=text or None
        )


@define
class TemplateLanguage(Language):
    """Deterministic verbalizer — the no-LLM baseline."""

    def realize(self, ctx: TurnContext) -> Utterance:
        if ctx.entry == "propose":
            return Utterance(text=f"I propose {ctx.bid}.", data={"entry": "propose"})
        return Utterance(
            text=f"Responding {ctx.acceptance}.", data={"entry": "respond"}
        )


@define
class HonestyValidation(Validation):
    """Rejects an utterance that names an outcome we are not offering."""

    def validate(self, ctx: TurnContext) -> ValidationResult:
        u = ctx.utterance
        if u is None or ctx.entry != "propose" or ctx.bid is None:
            return ValidationResult(ok=True)
        if str(ctx.bid) in u.text:
            return ValidationResult(ok=True)
        return ValidationResult(
            ok=False,
            issues=("utterance does not match the bid",),
            revised=Utterance(text=f"I propose {ctx.bid}.", data=u.data),
        )


@define
class NeverRepairValidation(Validation):
    """Always fails and offers no repair — exercises the resolution policy."""

    def validate(self, ctx: TurnContext) -> ValidationResult:
        return ValidationResult(ok=False, issues=("always fails",), revised=None)


@define
class AlwaysEnding(Ending):
    """Ends immediately; used to test both placements."""

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        return EndingDecision(end=True, reason="test")


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------


def test_language_attaches_text_to_offers_and_responses(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        ufun=u1,
    )
    m = _run(neg, os_, u2)
    mine = [t for t in m.full_trace if t.negotiator == neg.id]
    assert mine, "our negotiator never offered"
    assert any(t.text and "I propose" in t.text for t in mine)


def test_perception_runs_on_respond_and_is_visible_to_propose(domain):
    os_, u1, u2 = domain

    @define
    class RecordingLanguage(Language):
        def realize(self, ctx: TurnContext) -> Utterance:
            p = ctx.perception_this_step()
            return Utterance(text="ok", data={"heard": list(p.acts) if p else []})

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        perception=KeywordPerception(),
        language=RecordingLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    respond_turns = [t for t in neg.turns if t.entry == "respond"]
    assert respond_turns, "respond never ran"
    assert all(t.perception is not None for t in respond_turns), (
        "perception must run on every respond"
    )
    # propose reuses the same step's perception rather than re-perceiving
    propose_turns = [t for t in neg.turns if t.entry == "propose"]
    assert all(t.perception is None for t in propose_turns)


def test_validation_repairs_a_dishonest_utterance(domain):
    os_, u1, u2 = domain

    @define
    class LyingLanguage(Language):
        def realize(self, ctx: TurnContext) -> Utterance:
            return Utterance(text="I propose something else entirely.")

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=LyingLanguage(),
        validation=HonestyValidation(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    proposals = [t for t in neg.turns if t.entry == "propose" and t.bid is not None]
    assert proposals
    for t in proposals:
        # ``validation`` holds the FINAL verdict, which passes once repaired;
        # ``revalidations`` is what records that a repair was needed.
        assert t.revalidations >= 1, "validator never repaired the utterance"
        assert t.validation is not None and t.validation.ok
        assert str(t.bid) in t.utterance.text, "repaired text does not match the bid"


def test_validation_strict_drops_irreparable_utterance(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        validation=NeverRepairValidation(),
        resolution="strict",
        ufun=u1,
    )
    _run(neg, os_, u2)
    assert all(t.utterance.text == "" for t in neg.turns if t.utterance is not None)


def test_validation_log_sends_anyway(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        validation=NeverRepairValidation(),
        resolution="log",
        ufun=u1,
    )
    _run(neg, os_, u2)
    spoken = [t for t in neg.turns if t.utterance is not None]
    assert spoken and all(t.utterance.text for t in spoken)
    assert all(not t.validation.ok for t in spoken)


def test_validation_is_bounded(domain):
    """A validator that never accepts must not loop forever."""
    os_, u1, u2 = domain
    calls = {"n": 0}

    @define
    class CountingValidation(Validation):
        def validate(self, ctx: TurnContext) -> ValidationResult:
            calls["n"] += 1
            return ValidationResult(
                ok=False, issues=("no",), revised=Utterance(text="another try")
            )

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        validation=CountingValidation(),
        max_revalidations=2,
        ufun=u1,
    )
    _run(neg, os_, u2, n_steps=4)
    turns_with_text = len([t for t in neg.turns if t.utterance is not None])
    assert calls["n"] <= turns_with_text * 3, "validation was not bounded"


@pytest.mark.parametrize("when", ["early", "late"])
def test_ending_component_terminates(domain, when):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        ending=AlwaysEnding(),
        ending_when=when,
        ufun=u1,
    )
    m = _run(neg, os_, u2)
    assert m.state.agreement is None
    assert m.state.step <= 2, "ending component did not stop the negotiation promptly"


def test_turn_context_history_accumulates(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    assert len(neg.turns) >= 2
    assert neg.turn is None, "turn context must be closed outside a turn"
    assert len(neg.turns[-1].history) == len(neg.turns) - 1


def test_turn_context_carries_mechanism_limits(domain):
    """n_steps/time_limit come from the NMI, not the (limit-less) GBState."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2, n_steps=7)
    assert len(neg.turns) >= 1
    for t in neg.turns:
        assert t.n_steps == 7
        assert t.time_limit == float("inf"), "negmas reports unset time_limit as inf"


def test_their_offer_and_bid_are_separate_fields(domain):
    """The two must never be conflated — that is a known bug class."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=TemplateLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    for t in neg.turns:
        if t.entry == "propose":
            assert t.their_offer is None
        else:
            assert t.bid is None


def test_dialogue_only_turn_is_expressible(domain):
    """Bidding may return None while Language still speaks."""
    os_, u1, u2 = domain

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=LimitedOutcomesOfferingPolicy(outcomes=[]),
        language=TemplateLanguage(),
        ufun=u1,
    )
    ctx = neg._open_turn("propose", SAOMechanism(outcome_space=os_, n_steps=3).state)
    ctx.bid = None
    neg._run_language(ctx)
    attached = neg._attach(None, ctx)
    neg._close_turn()
    assert isinstance(attached, ExtendedOutcome)
    assert attached.outcome is None and attached.data["text"]


def test_components_receive_lifecycle_callbacks(domain):
    """New components must be registered like any other GBComponent."""
    os_, u1, u2 = domain
    seen = {"start": 0}

    @define
    class LifecycleLanguage(Language):
        def on_negotiation_start(self, state):  # noqa: D102
            seen["start"] += 1

        def realize(self, ctx: TurnContext) -> Utterance:
            return Utterance(text="hi")

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=LifecycleLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    assert seen["start"] == 1, "component did not get on_negotiation_start"


# ---------------------------------------------------------------------------
# Patterns found in the HAN-2026 taxonomy (see HAN_AGENTS_TAXONOMY.md)
# ---------------------------------------------------------------------------


def test_shared_state_survives_across_turns(domain):
    """Phase controllers need cross-turn state; TurnContext is turn-scoped."""
    os_, u1, u2 = domain

    @define
    class PhaseLanguage(Language):
        def realize(self, ctx: TurnContext) -> Utterance:
            n = self.negotiator.shared.get("turns", 0) + 1
            self.negotiator.shared["turns"] = n
            self.negotiator.shared["phase"] = "explore" if n < 3 else "concede"
            return Utterance(text=f"phase={self.negotiator.shared['phase']}")

    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=PhaseLanguage(),
        ufun=u1,
    )
    _run(neg, os_, u2)
    assert neg.shared["turns"] >= 3
    assert neg.shared["phase"] == "concede"


def test_joint_policy_idiom_computes_the_bid_once(domain):
    """A fused bid+accept policy fills both slots with one forward pass."""
    os_, u1, u2 = domain
    passes = {"n": 0}

    @define
    class JointOffering(TimeBasedOfferingPolicy):
        """Stands in for a learned policy emitting bid and accept together.

        The forward pass is ``__call__``; ``propose`` is the *cached* wrapper
        around it, so counting calls to ``propose`` would measure the wrong
        thing — which is exactly the point of the idiom.
        """

        def __call__(self, state, dest=None):
            passes["n"] += 1
            return super().__call__(state, dest)

    joint = JointOffering()

    @define
    class ReadsTheJointBid(AcceptTop):
        def __call__(self, state, offer, source):
            # negmas caches propose() per (step, thread), so this reuses the
            # same forward pass rather than triggering a second one.
            mine = joint.propose(state)
            if (
                offer is not None
                and mine is not None
                and self.negotiator.ufun is not None
                and float(self.negotiator.ufun(offer))
                >= float(self.negotiator.ufun(mine))
            ):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

    neg = make_pablove(acceptance=ReadsTheJointBid(0), offering=joint, ufun=u1)
    m = _run(neg, os_, u2, n_steps=6)
    steps = len({t.step for t in m.full_trace if t.negotiator == neg.id})
    assert passes["n"] <= steps + 1, (
        f"joint policy recomputed: {passes['n']} passes for {steps} steps"
    )

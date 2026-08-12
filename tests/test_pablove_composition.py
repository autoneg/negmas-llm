"""Tests for composing PABLO-ve slots: chaining with ``|`` and building from types.

Two features, one file, because they are two spellings of the same thing: a slot
takes several components (``a | b``, or the list ``[a, b]``), and it takes types
plus parameters instead of ready-made objects. ``a | b`` and ``[A, B]`` must
produce the *same* negotiator, which is asserted directly here.

Everything runs offline; no component in these tests calls an LLM.
"""

from __future__ import annotations

import pytest
from attrs import define, field
from negmas import make_issue, make_os
from negmas.gb.components.acceptance import AcceptTop
from negmas.gb.components.models.ufun import UFunModel
from negmas.gb.components.offering import TimeBasedOfferingPolicy
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import AspirationNegotiator, ResponseType, SAOMechanism

from negmas_llm.pablove import (
    Ending,
    EndingChain,
    EndingDecision,
    Language,
    LanguageChain,
    PABLOveNegotiator,
    Perception,
    PerceptionChain,
    PerceptionResult,
    TurnContext,
    Utterance,
    Validation,
    ValidationChain,
    ValidationResult,
    component_leaves,
    make_component,
    make_pablove,
    resolve_component_type,
)
from negmas_llm.pablove_components import TemplateLanguage


@pytest.fixture
def domain():
    """The same fixed, opposed two-issue domain ``test_pablove.py`` uses."""
    price = make_issue([100, 150, 200], "price")
    quantity = make_issue([1, 2, 3], "quantity")
    os_ = make_os([price, quantity])
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


def _ctx(**kwargs) -> TurnContext:
    """A bare turn context; no mechanism needed for a component unit test."""
    from negmas.gb.common import GBState

    kwargs.setdefault("entry", "respond")
    kwargs.setdefault("state", GBState())
    return TurnContext(**kwargs)


# ---------------------------------------------------------------------------
# Offline components
# ---------------------------------------------------------------------------


@define
class FixedPerception(Perception):
    """Reports a canned reading, so merge order is exactly observable."""

    result: PerceptionResult = field(factory=PerceptionResult)
    seen: list[str] = field(factory=list)

    def perceive(self, ctx: TurnContext) -> PerceptionResult:
        self.seen.append("perceive")
        return self.result


@define
class PrefixLanguage(Language):
    """Prepends its ``prefix`` to whatever draft it is handed (if any)."""

    prefix: str = "a"

    def realize(self, ctx: TurnContext) -> Utterance:
        draft = ctx.utterance
        base = draft.text if draft is not None else ""
        return Utterance(text=f"{self.prefix}{base}", data={"last": self.prefix})


@define
class ReplacingLanguage(Language):
    """Ignores any draft — the honest reading of "does not revise"."""

    text: str = "replaced"

    def realize(self, ctx: TurnContext) -> Utterance:
        return Utterance(text=self.text)


@define
class BannedWordValidation(Validation):
    """Fails when the utterance contains ``word``, and rewrites it out."""

    word: str = "bad"
    repair: bool = True
    calls: list[str] = field(factory=list)

    def validate(self, ctx: TurnContext) -> ValidationResult:
        text = ctx.utterance.text if ctx.utterance else ""
        self.calls.append(text)
        if self.word not in text:
            return ValidationResult(ok=True)
        return ValidationResult(
            ok=False,
            issues=(f"contains {self.word!r}",),
            revised=Utterance(text=text.replace(self.word, ""))
            if self.repair
            else None,
        )


@define
class RewritingValidation(Validation):
    """Reports ``ok`` yet still hands back a rewrite — see the chain's contract."""

    def validate(self, ctx: TurnContext) -> ValidationResult:
        text = ctx.utterance.text if ctx.utterance else ""
        return ValidationResult(ok=True, revised=Utterance(text=text.upper()))


@define
class FixedEnding(Ending):
    """Answers a canned decision, and records that it was asked."""

    end: bool = False
    calls: list[bool] = field(factory=list)

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        self.calls.append(self.end)
        return EndingDecision(end=self.end, reason=f"fixed:{self.end}")


@define
class ZeroModel(UFunModel):
    """A minimal opponent model — only needs to be a `Model` for the slot."""

    label: str = "z"

    def eval(self, offer):  # noqa: D102 - BaseUtilityFunction contract
        return 0.0


# ---------------------------------------------------------------------------
# `|` — building chains
# ---------------------------------------------------------------------------


def test_or_builds_a_chain_of_the_right_kind():
    assert isinstance(FixedPerception() | FixedPerception(), PerceptionChain)
    assert isinstance(PrefixLanguage() | PrefixLanguage(), LanguageChain)
    assert isinstance(BannedWordValidation() | BannedWordValidation(), ValidationChain)
    assert isinstance(FixedEnding() | FixedEnding(), EndingChain)


def test_or_is_flat_and_associative():
    """``(a|b)|c`` and ``a|(b|c)`` are the same three-element chain."""
    a, b, c = FixedPerception(), FixedPerception(), FixedPerception()
    left, right = (a | b) | c, a | (b | c)
    assert [id(x) for x in left.components] == [id(a), id(b), id(c)]
    assert [id(x) for x in right.components] == [id(a), id(b), id(c)]


def test_or_does_not_mutate_its_operands():
    a, b, c = FixedPerception(), FixedPerception(), FixedPerception()
    ab = a | b
    _ = ab | c
    assert len(ab.components) == 2, "chaining must build a new chain, not extend one"


def test_or_across_kinds_is_a_type_error():
    """``perception | language`` is a mistake, not a two-element chain."""
    with pytest.raises(TypeError):
        _ = FixedPerception() | PrefixLanguage()
    with pytest.raises(TypeError):
        _ = BannedWordValidation() | FixedEnding()


# ---------------------------------------------------------------------------
# PerceptionChain — merge semantics
# ---------------------------------------------------------------------------


def test_perception_chain_merges_and_keeps_act_order():
    """Acts are unioned in order with duplicates dropped — never a `set`.

    Perceived acts are joined straight into LLM prompts, so set ordering would
    make those prompts differ run to run for no reason.
    """
    first = FixedPerception(
        result=PerceptionResult(acts=("propose", "concede"), source="classified")
    )
    second = FixedPerception(
        result=PerceptionResult(acts=("concede", "ask"), source="classified")
    )
    merged = (first | second).perceive(_ctx())
    assert merged.acts == ("propose", "concede", "ask")


def test_perception_chain_first_non_empty_wins_for_single_valued_fields():
    first = FixedPerception(
        result=PerceptionResult(source="none", extracted={"k": "from-first"})
    )
    second = FixedPerception(
        result=PerceptionResult(
            sentiment="positive",
            text="they said this",
            extracted={"k": "from-second", "other": 1},
            source="classified",
        )
    )
    merged = (first | second).perceive(_ctx())
    assert merged.sentiment == "positive", "an empty field must not shadow a full one"
    assert merged.text == "they said this", "text must survive for _partner_text"
    assert merged.extracted == {"k": "from-first", "other": 1}


def test_perception_chain_reports_the_best_source():
    """``wire`` beats ``classified`` beats ``none``, whatever the order."""
    wire = FixedPerception(result=PerceptionResult(source="wire"))
    classified = FixedPerception(result=PerceptionResult(source="classified"))
    none = FixedPerception(result=PerceptionResult(source="none"))
    assert (none | classified).perceive(_ctx()).source == "classified"
    assert (wire | classified).perceive(_ctx()).source == "wire"
    assert (classified | wire).perceive(_ctx()).source == "wire"
    assert (none | none).perceive(_ctx()).source == "none"


def test_perception_chain_deduplicates_commitments():
    same = {"text": "we will ship by Friday"}
    first = FixedPerception(result=PerceptionResult(commitments=(same,)))
    second = FixedPerception(
        result=PerceptionResult(commitments=(dict(same), {"text": "and no more"}))
    )
    merged = (first | second).perceive(_ctx())
    assert merged.commitments == (same, {"text": "and no more"})


def test_perception_chain_runs_every_member():
    first, second = FixedPerception(), FixedPerception()
    (first | second).perceive(_ctx())
    assert first.seen and second.seen, "perceivers observe; none may be skipped"


# ---------------------------------------------------------------------------
# LanguageChain — progressive drafting
# ---------------------------------------------------------------------------


def test_language_chain_passes_the_draft_along():
    chain = PrefixLanguage(prefix="1") | PrefixLanguage(prefix="2")
    assert chain.realize(_ctx()).text == "21", "later links must see the draft"


def test_language_chain_restores_ctx_utterance():
    """The chain looks exactly like one component from the outside.

    It writes drafts into ``ctx.utterance`` so later links can read them, but
    installing the result is the *pipeline's* job — leaving the last draft
    behind would make the chain indistinguishable from having already been
    installed.
    """
    ctx = _ctx()
    ctx.utterance = Utterance(text="original")
    result = (PrefixLanguage(prefix="1") | PrefixLanguage(prefix="2")).realize(ctx)
    assert result.text == "21original"
    assert ctx.utterance is not None and ctx.utterance.text == "original"


def test_language_chain_link_may_ignore_the_draft():
    chain = PrefixLanguage(prefix="1") | ReplacingLanguage(text="fresh")
    assert chain.realize(_ctx()).text == "fresh"


def test_language_chain_attaches_the_last_links_data(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=PrefixLanguage(prefix="x") | PrefixLanguage(prefix="y"),
        ufun=u1,
    )
    m = _run(neg, os_, u2, n_steps=4)
    said = [t.utterance for t in neg.turns if t.utterance is not None]
    assert said, "the chain must still produce an utterance"
    assert all(u.text == "yx" for u in said), "both links ran, in order"
    assert all(u.data == {"last": "y"} for u in said), "the last link's payload wins"
    on_wire = [t.data for t in m.full_trace if t.negotiator == neg.id and t.data]
    assert on_wire and all("yx" in d["text"] for d in on_wire), (
        "and it reaches the wire"
    )


# ---------------------------------------------------------------------------
# ValidationChain — all checks must pass, rewrites compose
# ---------------------------------------------------------------------------


def test_validation_chain_composes_rewrites():
    """Each check sees the previous one's rewrite, so fixes stack."""
    ctx = _ctx()
    ctx.utterance = Utterance(text="bad and worse")
    chain = BannedWordValidation(word="bad") | BannedWordValidation(word="worse")
    result = chain.validate(ctx)
    assert not result.ok
    assert result.revised is not None
    assert result.revised.text == " and "
    assert result.issues == ("contains 'bad'", "contains 'worse'")


def test_validation_chain_runs_every_check_and_reports_all_issues():
    """A failure does not short-circuit: the caller learns about every problem."""
    ctx = _ctx()
    ctx.utterance = Utterance(text="bad and worse")
    second = BannedWordValidation(word="worse", repair=False)
    chain = BannedWordValidation(word="bad", repair=False) | second
    result = chain.validate(ctx)
    assert second.calls, "the second check must run even after the first failed"
    assert result.issues == ("contains 'bad'", "contains 'worse'")
    assert result.revised is None


def test_validation_chain_reports_not_ok_when_it_has_a_rewrite():
    """A rewrite must never be smuggled out under ``ok=True``.

    ``PABLOveNegotiator._run_validation`` returns the moment a result is ``ok``,
    without installing ``revised`` — so a chain that produced a rewrite has to
    say ``ok=False`` or the rewrite is silently dropped.
    """
    ctx = _ctx()
    ctx.utterance = Utterance(text="quiet")
    result = (RewritingValidation() | RewritingValidation()).validate(ctx)
    assert result.revised is not None and result.revised.text == "QUIET"
    assert not result.ok, "a chain holding a rewrite must not report ok"
    assert result.issues == ("rewritten",)


def test_validation_chain_no_op_rewrite_is_not_a_rewrite():
    """A rewrite identical to the input must not be reported as one.

    Otherwise the pipeline reinstalls the same text, re-runs the chain, and
    burns through ``max_revalidations`` — see the strict-mode test below.
    """
    ctx = _ctx()
    ctx.utterance = Utterance(text="ALREADY LOUD")
    result = (RewritingValidation() | RewritingValidation()).validate(ctx)
    assert result.ok and result.revised is None


def test_validation_chain_never_blanks_an_unobjected_message(domain):
    """Chaining must not turn a passing message into silence under ``strict``.

    A lone ``ok=True``-with-a-rewrite validator leaves the message intact, so a
    chain of two must not instead loop to the revalidation cap and have
    ``resolution="strict"`` blank a message no member ever objected to.
    """
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=ReplacingLanguage(text="hello"),
        validation=RewritingValidation() | RewritingValidation(),
        ufun=u1,
        resolution="strict",
    )
    _run(neg, os_, u2, n_steps=4)
    said = [t.utterance for t in neg.turns if t.utterance is not None]
    assert said, "expected an utterance every turn"
    assert all(u.text == "HELLO" for u in said), "the rewrite applies, once"
    assert all(t.revalidations == 1 for t in neg.turns), "and then reaches a fixpoint"


def test_validation_chain_restores_ctx_utterance():
    ctx = _ctx()
    ctx.utterance = Utterance(text="bad")
    (BannedWordValidation(word="bad") | BannedWordValidation(word="x")).validate(ctx)
    assert ctx.utterance is not None and ctx.utterance.text == "bad"


def test_validation_chain_all_ok_is_ok():
    ctx = _ctx()
    ctx.utterance = Utterance(text="fine")
    result = (
        BannedWordValidation(word="bad") | BannedWordValidation(word="x")
    ).validate(ctx)
    assert result.ok and result.revised is None and result.issues == ()


def test_validation_chain_rewrite_reaches_the_wire(domain):
    """End to end: the pipeline installs a chain's composed rewrite."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        language=ReplacingLanguage(text="bad and worse"),
        validation=[
            (BannedWordValidation, {"word": "bad"}),
            (BannedWordValidation, {"word": "worse"}),
        ],
        ufun=u1,
        max_revalidations=3,
    )
    _run(neg, os_, u2, n_steps=4)
    said = [t.utterance for t in neg.turns if t.utterance is not None]
    assert said, "expected an utterance every turn"
    assert all(u.text == " and " for u in said), "both rewrites were installed"
    assert all(t.revalidations == 1 for t in neg.turns), (
        "one chain pass repaired both problems, so only one re-validation"
    )


# ---------------------------------------------------------------------------
# EndingChain
# ---------------------------------------------------------------------------


def test_ending_chain_any_short_circuits():
    second = FixedEnding(end=False)
    chain = FixedEnding(end=True) | second
    decision = chain.should_end(_ctx())
    assert decision.end and decision.reason == "fixed:True"
    assert not second.calls, "'any' must not consult the rest once one says stop"


def test_ending_chain_any_needs_one_yes():
    assert not (FixedEnding(end=False) | FixedEnding(end=False)).should_end(_ctx()).end
    assert (FixedEnding(end=False) | FixedEnding(end=True)).should_end(_ctx()).end


def test_ending_chain_all_requires_unanimity():
    chain = EndingChain(components=(FixedEnding(end=True), FixedEnding(end=False)))
    chain.mode = "all"
    assert not chain.should_end(_ctx()).end
    unanimous = EndingChain(
        components=(FixedEnding(end=True), FixedEnding(end=True)), mode="all"
    )
    assert unanimous.should_end(_ctx()).end


def test_empty_ending_chain_never_ends():
    assert not EndingChain().should_end(_ctx()).end


def test_ending_chain_terminates_a_real_negotiation(domain):
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        ending=FixedEnding(end=False) | FixedEnding(end=True),
        ufun=u1,
    )
    m = _run(neg, os_, u2, n_steps=10)
    assert m.agreement is None
    assert m.state.step < 10, "the chain should have ended it early"


# ---------------------------------------------------------------------------
# Types and parameters
# ---------------------------------------------------------------------------


def test_type_is_constructed_with_its_params():
    neg = make_pablove(
        language=TemplateLanguage, language_params={"reject_template": "nope"}
    )
    assert isinstance(neg._language, TemplateLanguage)
    assert neg._language.reject_template == "nope"


def test_type_without_params_uses_defaults():
    neg = make_pablove(language=TemplateLanguage)
    assert neg._language.reject_template == "That does not work for me."


def test_tuple_spec_carries_its_own_params():
    neg = make_pablove(language=(TemplateLanguage, {"reject_template": "tuple"}))
    assert neg._language.reject_template == "tuple"


def test_tuple_params_override_slot_params():
    neg = make_pablove(
        language=(TemplateLanguage, {"reject_template": "inner"}),
        language_params={"reject_template": "outer", "end_template": "kept"},
    )
    assert neg._language.reject_template == "inner"
    assert neg._language.end_template == "kept", "outer params still apply otherwise"


def test_string_specs_resolve_bare_and_dotted_names():
    assert resolve_component_type("TemplateLanguage") is TemplateLanguage
    assert (
        resolve_component_type("negmas_llm.pablove_components.TemplateLanguage")
        is TemplateLanguage
    )
    neg = make_pablove(
        language="TemplateLanguage", language_params={"end_template": "e"}
    )
    assert isinstance(neg._language, TemplateLanguage)
    assert neg._language.end_template == "e"


def test_string_spec_resolves_a_negmas_component():
    neg = make_pablove(offering="TimeBasedOfferingPolicy", acceptance="AcceptTop")
    assert isinstance(neg._offering, TimeBasedOfferingPolicy)
    assert isinstance(neg._acceptance, AcceptTop)


def test_unknown_name_is_an_error():
    with pytest.raises(ValueError, match="Cannot find a component class"):
        make_pablove(language="NoSuchLanguageComponent")


def test_wrong_kind_for_a_slot_is_an_error():
    with pytest.raises(ValueError, match="expected a Language"):
        make_pablove(language=FixedPerception())


def test_params_for_an_instance_are_an_error():
    """Silently dropping them would leave the caller with a mis-built component."""
    with pytest.raises(ValueError, match="already-constructed"):
        make_pablove(
            language=TemplateLanguage(), language_params={"reject_template": "x"}
        )


def test_list_spec_equals_the_or_spelling():
    """``[A, B]`` and ``a | b`` must build the same thing."""
    from_list = make_pablove(perception=[FixedPerception, FixedPerception])._perception
    from_or = make_pablove(perception=FixedPerception() | FixedPerception())._perception
    assert type(from_list) is type(from_or) is PerceptionChain
    assert len(from_list.components) == len(from_or.components) == 2


def test_union_of_types_builds_a_chain():
    """``A | B`` over *types* is a `types.UnionType`, not a chain — until we fix it.

    Without this, the two features would quietly disagree: the same ``|`` that
    chains instances would produce a type annotation for types.
    """

    # Two *different* classes: Python collapses `A | A` to `A` before we see it.
    @define
    class OtherPerception(FixedPerception):
        pass

    neg = make_pablove(perception=FixedPerception | OtherPerception)
    assert isinstance(neg._perception, PerceptionChain)
    assert [type(c) for c in neg._perception.components] == [
        FixedPerception,
        OtherPerception,
    ]


def test_list_params_are_matched_element_wise():
    neg = make_pablove(
        language=[TemplateLanguage, TemplateLanguage],
        language_params=[{"reject_template": "first"}, {"reject_template": "second"}],
    )
    assert [c.reject_template for c in neg._language.components] == ["first", "second"]


def test_one_param_dict_applies_to_every_element():
    neg = make_pablove(
        language=[TemplateLanguage, TemplateLanguage],
        language_params={"reject_template": "both"},
    )
    assert [c.reject_template for c in neg._language.components] == ["both", "both"]


def test_mismatched_param_list_length_is_an_error():
    with pytest.raises(ValueError, match="parameter dicts"):
        make_pablove(
            language=[TemplateLanguage, TemplateLanguage],
            language_params=[{"reject_template": "only-one"}],
        )


def test_single_element_list_is_not_wrapped_in_a_chain():
    neg = make_pablove(language=[TemplateLanguage])
    assert isinstance(neg._language, TemplateLanguage)


def test_unchainable_slot_rejects_several_components():
    with pytest.raises(ValueError, match="cannot combine"):
        make_pablove(offering=[TimeBasedOfferingPolicy, TimeBasedOfferingPolicy])


def test_malformed_tuple_spec_is_an_error():
    with pytest.raises(ValueError, match="must be"):
        make_pablove(language=(TemplateLanguage, TemplateLanguage))


def test_model_slot_accepts_a_list_of_models():
    """negmas' own ``models=`` takes a list, so this needs no chain type."""
    neg = make_pablove(model=[ZeroModel, (ZeroModel, {"label": "second"})])
    labels = [c.label for c in neg.components if isinstance(c, ZeroModel)]
    assert labels == ["z", "second"]


def test_make_component_directly():
    """The resolver is public: a project can use the same grammar elsewhere."""
    assert make_component(None) is None
    built = make_component("TemplateLanguage", Language, {"end_template": "x"})
    assert isinstance(built, TemplateLanguage) and built.end_template == "x"


# ---------------------------------------------------------------------------
# Chains and the negotiator: registration, lifecycle, configuration
# ---------------------------------------------------------------------------


def test_component_leaves_flattens_recursively():
    a, b, c = FixedPerception(), FixedPerception(), FixedPerception()
    nested = PerceptionChain(components=(PerceptionChain(components=(a, b)), c))
    assert [id(x) for x in component_leaves(nested)] == [id(a), id(b), id(c)]
    assert component_leaves(None) == []
    assert [id(x) for x in component_leaves(a)] == [id(a)]


def test_leaves_are_registered_not_the_chain():
    """Leaves get the lifecycle, chains do not — a chain owns no state.

    Registering the chain instead would give its members no ``negotiator``, no
    ufun/NMI sharing and none of the per-component configuration below.
    """
    first, second = FixedPerception(), FixedPerception()
    neg = make_pablove(perception=first | second)
    registered = [id(c) for c in neg.components]
    assert id(first) in registered and id(second) in registered
    assert id(neg._perception) not in registered, "the chain itself is not a component"
    assert first.negotiator is neg and second.negotiator is neg


def test_chained_leaves_get_distinct_component_names():
    """`ModularNegotiator`'s component map is a dict — a repeated name shadows."""
    neg = make_pablove(perception=[FixedPerception, FixedPerception])
    names = neg._ModularNegotiator__component_map  # noqa: SLF001 - the map is private
    assert "perception[0]" in names and "perception[1]" in names


def test_single_component_keeps_its_plain_slot_name():
    neg = make_pablove(perception=FixedPerception())
    assert "perception" in neg._ModularNegotiator__component_map  # noqa: SLF001


def test_memory_mode_override_reaches_every_leaf():
    """The uniform override walks registered components — so leaves must be there."""
    from negmas_llm.pablove_components import LLMPerception

    neg = PABLOveNegotiator(
        perception=[LLMPerception, LLMPerception], memory_mode="none"
    )
    assert [c.memory_mode for c in neg._perception.components] == ["none", "none"]


def test_chained_leaves_receive_lifecycle_callbacks(domain):
    os_, u1, u2 = domain

    @define
    class CountingPerception(Perception):
        counts: dict = field(factory=lambda: {"start": 0, "prefs": 0})

        def perceive(self, ctx: TurnContext) -> PerceptionResult:
            return PerceptionResult(source="none")

        def on_negotiation_start(self, state):
            self.counts["start"] += 1

        def on_preferences_changed(self, changes):
            self.counts["prefs"] += 1

    first, second = CountingPerception(), CountingPerception()
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        perception=first | second,
        ufun=u1,
    )
    _run(neg, os_, u2, n_steps=4)
    for comp in (first, second):
        assert comp.counts["start"] >= 1, "each leaf gets on_negotiation_start"
        assert comp.counts["prefs"] >= 1, "each leaf gets on_preferences_changed"


def test_chained_perception_reaches_the_pipeline(domain):
    """A chain in the slot behaves as one perceiver from the pipeline's view."""
    os_, u1, u2 = domain
    first = FixedPerception(result=PerceptionResult(acts=("a",), source="classified"))
    second = FixedPerception(result=PerceptionResult(acts=("b",), source="wire"))
    neg = make_pablove(
        acceptance=AcceptTop(0),
        offering=TimeBasedOfferingPolicy(),
        perception=first | second,
        ufun=u1,
    )
    _run(neg, os_, u2, n_steps=4)
    perceived = [t.perception for t in neg.turns if t.perception is not None]
    assert perceived, "perception must still run"
    assert all(p.acts == ("a", "b") and p.source == "wire" for p in perceived)


def test_specs_do_not_disturb_the_boa_reduction(domain):
    """Types-instead-of-objects must not smuggle a component into a plain BOA."""
    os_, u1, u2 = domain
    neg = make_pablove(
        acceptance="AcceptTop", offering="TimeBasedOfferingPolicy", ufun=u1
    )
    assert neg.is_plain_boa
    m = _run(neg, os_, u2, n_steps=6)
    for entry in m.full_trace:
        assert entry.data in (None, {})
    assert m.state.step > 0 and ResponseType is not None

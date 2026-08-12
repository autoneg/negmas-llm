"""PABLO-ve: a component architecture for language-capable negotiators.

**PABLO-ve** — **P**erception, **A**cceptance, **B**idding, **L**anguage,
**O**pponent-modeling, plus optional **v**alidation and **e**nding — generalizes
negmas' BOA (Bidding / Opponent model / Acceptance) to negotiators that *read*
and *produce* natural language.

BOA has no slot for understanding what the partner said, for deciding to stop, or
for producing what we say back. PABLO-ve adds those as ordinary negmas
components, so they inherit lifecycle callbacks, ufun/NMI sharing and registry
support for free:

===========  ==========================  ==============================
Letter       Component                   negmas equivalent
===========  ==========================  ==============================
``P``        :class:`Perception`         -- (new)
``A``        ``AcceptancePolicy``        unchanged
``B``        ``OfferingPolicy``          unchanged (the *bidding* slot)
``L``        :class:`Language`           -- (new)
``O``        ``Model``                   unchanged
``v``        :class:`Validation`         -- (new, auxiliary)
``e``        :class:`Ending`             -- (new, auxiliary)
===========  ==========================  ==============================

*Bidding, not "yielding"*: an offering policy does not always concede — Boulware
holds, tit-for-tat can escalate, hardheaded never moves.

**Every component is optional.** With ``perception``, ``language``,
``validation`` and ``ending`` all ``None``, a :class:`PABLOveNegotiator` is
behaviourally identical to the corresponding BOA negotiator — the same offers,
the same responses, the same trace. That equivalence is asserted by
``tests/test_pablove.py::test_pablove_reduces_to_boa`` rather than merely
claimed, and it is what makes "every BOA negotiator is a PABLO-ve negotiator"
a fact about the code.

Inter-step data flows through a turn-scoped :class:`TurnContext` reachable at
``self.negotiator.turn``, *not* through the component signatures — widening
``AcceptancePolicy.__call__`` or ``OfferingPolicy.propose`` would break every
existing policy in negmas. Old components ignore the context; new ones read it.
"""

from __future__ import annotations

import importlib
import types
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from attrs import define
from attrs import field as afield
from negmas.common import MechanismState, PreferencesChange
from negmas.gb.common import ExtendedResponseType, GBState
from negmas.gb.components import AcceptancePolicy, GBComponent, Model, OfferingPolicy
from negmas.gb.negotiators.modular.mapneg import MAPNegotiator
from negmas.helpers import get_class
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.sao import ResponseType

from negmas_llm.negotiator import (
    DEFAULT_PREFERENCES_CHANGED_PROMPT,
    DEFAULT_PREFERENCES_PROMPT,
)
from negmas_llm.tags import process_prompt

if TYPE_CHECKING:
    pass

__all__ = [
    "TurnContext",
    "PerceptionResult",
    "Utterance",
    "EndingDecision",
    "ValidationResult",
    "Perception",
    "Language",
    "Ending",
    "Validation",
    "ComponentChain",
    "PerceptionChain",
    "LanguageChain",
    "ValidationChain",
    "EndingChain",
    "ComponentSpec",
    "make_component",
    "component_leaves",
    "resolve_component_type",
    "PABLOveNegotiator",
    "make_pablove",
]

#: What any PABLO-ve slot accepts, in place of a ready-made component:
#:
#: * a component **instance** — used as-is (today's only option);
#: * a **type** — instantiated with that slot's ``*_params`` (or ``{}``);
#: * a **string** — a class name resolved with :func:`resolve_component_type`
#:   (bare names are looked up in ``negmas_llm`` and negmas, dotted paths
#:   anywhere), then treated as a type;
#: * a ``(spec, params)`` **tuple** — any of the above plus the keyword
#:   arguments to build it with, which override that slot's ``*_params``;
#: * a **list** of specs — combined into a chain (see :class:`ComponentChain`),
#:   exactly as ``a | b | c`` would;
#: * ``A | B`` where both are *types* — Python evaluates this to a
#:   ``types.UnionType`` before we ever see it, so it is treated as a list of
#:   the two types, making ``|`` mean the same thing for types as for
#:   instances;
#: * ``None`` — the slot stays empty.
ComponentSpec = Any


# =============================================================================
# Values passed between steps
# =============================================================================


@dataclass(frozen=True)
class PerceptionResult:
    """What the partner's move *was*, as structure rather than prose.

    Deliberately open: a speech-act perceiver, a sentiment classifier and a
    keyword matcher are all valid `Perception` components, so consumers should
    read the fields they need and degrade gracefully when they are empty.

    Attributes:
        acts: Illocutionary/dialogue acts attributed to the partner's utterance.
        commitments: Structured obligations the partner took on, if any.
        sentiment: Free-form affect label, when the perceiver produces one.
        extracted: Any other structured content recovered from the utterance.
        source: Where the perception came from — ``"wire"`` when the partner
            published typed data, ``"classified"`` when we inferred it,
            ``"none"`` when nothing was available.
        text: The raw utterance perceived (kept for logging, not for policies).
    """

    acts: tuple[str, ...] = ()
    commitments: tuple[dict[str, Any], ...] = ()
    sentiment: str | None = None
    extracted: dict[str, Any] = field(default_factory=dict)
    source: Literal["wire", "classified", "none"] = "none"
    text: str | None = None


@dataclass(frozen=True)
class Utterance:
    """What we say, and any structured payload that travels with it."""

    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_data(self) -> dict[str, Any]:
        """The payload to attach to an outcome/response (text included)."""
        out = dict(self.data)
        if self.text:
            out["text"] = self.text
        return out


@dataclass(frozen=True)
class EndingDecision:
    """Whether to terminate, and why (the reason is logged, never sent)."""

    end: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ValidationResult:
    """Whether the utterance is true of the decision that was taken.

    Attributes:
        ok: True when the utterance is consistent with the decision.
        issues: Human-readable descriptions of each inconsistency found.
        revised: A repaired utterance, when the validator could produce one.
    """

    ok: bool = True
    issues: tuple[str, ...] = ()
    revised: Utterance | None = None


# =============================================================================
# The turn context — how steps see each other's output
# =============================================================================


@dataclass
class TurnContext:
    """Everything produced during one ``propose`` or ``respond`` entry.

    Created at the top of each entry point and reachable from any component as
    ``self.negotiator.turn``. Fields are ``None`` until the step that fills them
    has run, so a component can tell "not computed yet" from "computed as empty".

    ``their_offer`` is the partner's offer and is populated on ``respond`` only;
    ``bid`` is *our* outcome and is populated on ``propose``. They are separate
    fields on purpose: collapsing them into one "outcome" whose meaning depends
    on the entry point is a known source of subtle bugs.

    ``their_offer`` is always a plain ``Outcome``, never an ``ExtendedOutcome``:
    negmas unbundles the two once an offer reaches the mechanism, storing them
    as separate ``ThreadState.new_offer`` / ``ThreadState.new_data`` fields.
    ``their_data`` is that second half — the partner's raw payload (their
    ``"text"`` key included, when they sent one) — populated on ``respond``
    from ``state.threads[source].new_data``.

    ``history`` is every earlier turn, oldest first. It is a *view* over the
    negotiator's own growing turn log, not a per-turn copy: materializing a
    fresh tuple of everything-so-far on every single turn would cost O(n) time
    and space each time, i.e. O(n^2) over an n-turn negotiation. Instead each
    ``TurnContext`` stores a reference to the same underlying list plus the
    length it had at creation time, so building the tuple is deferred to
    whoever actually reads ``history`` (and ``last``/``perception_this_step``
    below walk the list directly instead of going through ``history`` at all).

    ``step`` and ``relative_time`` are read live from ``state`` rather than
    copied, so they always reflect the state this turn was opened with.
    ``n_steps``/``time_limit`` come from the NMI instead, since ``GBState``
    itself carries no mechanism limits; both are captured once, in
    ``PABLOveNegotiator._open_turn``. Any component can reach all four here
    without going through ``self.negotiator.nmi`` itself, which is ``None``
    outside a live negotiation.
    """

    entry: Literal["propose", "respond"]
    state: GBState
    source: str | None = None
    dest: str | None = None
    their_offer: Outcome | None = None
    their_data: dict[str, Any] | None = None
    #: Mechanism limits, from the NMI at the time the turn opened. ``None``
    #: means unlimited, not "unknown" — the NMI reports it that way too.
    n_steps: int | None = None
    time_limit: float | None = None

    perception: PerceptionResult | None = None
    beliefs: Any = None
    ending: EndingDecision | None = None
    bid: Outcome | ExtendedOutcome | None = None
    acceptance: ResponseType | None = None
    utterance: Utterance | None = None
    validation: ValidationResult | None = None
    #: How many times the utterance was regenerated after a failed validation.
    #: ``validation`` holds only the *final* verdict, so this is what records
    #: that a repair happened at all.
    revalidations: int = 0

    #: Reference to the owning negotiator's ``_turns`` list (shared, not
    #: copied) plus the length it had when this turn was opened. Together
    #: they define the ``history`` prefix without duplicating it per turn.
    _history_all: list[TurnContext] | None = field(
        default=None, repr=False, compare=False, init=False
    )
    _history_len: int = field(default=0, repr=False, compare=False, init=False)

    @property
    def history(self) -> tuple[TurnContext, ...]:
        """Every turn before this one, oldest first."""
        if self._history_all is None:
            return ()
        return tuple(self._history_all[: self._history_len])

    @property
    def step(self) -> int:
        """The mechanism step this turn belongs to."""
        return int(getattr(self.state, "step", 0) or 0)

    @property
    def relative_time(self) -> float:
        """Mechanism relative time in ``[0, 1]``."""
        return float(getattr(self.state, "relative_time", 0.0) or 0.0)

    def last(
        self, entry: Literal["propose", "respond"] | None = None
    ) -> TurnContext | None:
        """The most recent previous turn, optionally of a given entry point."""
        if self._history_all is None:
            return None
        for i in range(self._history_len - 1, -1, -1):
            ctx = self._history_all[i]
            if entry is None or ctx.entry == entry:
                return ctx
        return None

    def perception_this_step(self) -> PerceptionResult | None:
        """Perception from this step, whichever entry point produced it.

        ``respond`` perceives and ``propose`` reuses, so a proposing component
        can still see what the partner just said.
        """
        if self.perception is not None:
            return self.perception
        if self._history_all is None:
            return None
        for i in range(self._history_len - 1, -1, -1):
            ctx = self._history_all[i]
            if ctx.step != self.step:
                break
            if ctx.perception is not None:
                return ctx.perception
        return None


# =============================================================================
# New component base classes
# =============================================================================


class ComponentChain:
    """Marker for a component that is really several components in a row.

    A chain *is* a component of its own kind — a :class:`PerceptionChain` is a
    :class:`Perception` — so every slot keeps its single-component type and
    nothing downstream needs to know whether it holds one component or five.

    Chains are built with ``a | b`` on each chainable base class. Chaining is
    *kind-checked*: ``perception | language`` is a mistake, not a two-element
    chain, so ``__or__`` returns ``NotImplemented`` and Python raises
    ``TypeError`` as it would for any other unsupported operand pair. Neither
    operand is ever mutated, and nesting is flattened, so ``(a | b) | c`` and
    ``a | (b | c)`` build the same three-element chain.

    Empty ``__slots__`` on purpose: every chain's other base is a slotted
    ``attrs`` class, and a second base that adds layout would raise the same
    conflict documented in
    :class:`~negmas_llm.pablove_components.LLMComponent`.

    Attributes:
        components: The chained components, in the order they run.
    """

    __slots__ = ()

    components: tuple[Any, ...]


def _leaves_of(comp: Any) -> tuple[Any, ...]:
    """``comp`` flattened to the components that actually do the work."""
    if isinstance(comp, ComponentChain):
        out: tuple[Any, ...] = ()
        for child in comp.components:
            out += _leaves_of(child)
        return out
    return (comp,)


def component_leaves(comp: Any) -> list[GBComponent]:
    """Every real component inside ``comp``, chains flattened recursively.

    A chain is not registered with the negotiator; its leaves are (see
    :class:`PABLOveNegotiator`), so each one gets exactly the lifecycle
    callbacks, ufun/NMI sharing and configuration overrides it would get as a
    standalone component.

    Args:
        comp: A component, a chain, or ``None``.

    Returns:
        The leaf components, in run order (empty for ``None``).
    """
    return [] if comp is None else list(_leaves_of(comp))


@define
class Perception(GBComponent):
    """Turns the partner's utterance into structure. Decides nothing.

    Always the first step of a turn. Runs on ``respond`` (where the partner's
    message arrives); ``propose`` reuses the same step's result.

    ``a | b`` builds a :class:`PerceptionChain` that runs both and merges what
    they saw.
    """

    def perceive(self, ctx: TurnContext) -> PerceptionResult:
        """Read the partner's move.

        Args:
            ctx: The current turn context; ``ctx.their_offer`` and the partner's
                data are available, nothing downstream is.

        Returns:
            The structured observation.
        """
        raise NotImplementedError

    def __or__(self, other: Perception) -> Perception:
        """Run this perceiver and ``other``, merging both readings."""
        if not isinstance(other, Perception):
            return NotImplemented
        return PerceptionChain(components=_leaves_of(self) + _leaves_of(other))


@define
class Language(GBComponent):
    """Turns a decision into an utterance. May not change the decision.

    ``a | b`` builds a :class:`LanguageChain` in which ``b`` revises ``a``'s
    draft.
    """

    def realize(self, ctx: TurnContext) -> Utterance:
        """Say what was decided.

        Args:
            ctx: Turn context with ``bid`` / ``acceptance`` already filled in.

        Returns:
            The utterance to send.
        """
        raise NotImplementedError

    def __or__(self, other: Language) -> Language:
        """Draft with this component, then hand the draft to ``other``."""
        if not isinstance(other, Language):
            return NotImplemented
        return LanguageChain(components=_leaves_of(self) + _leaves_of(other))


@define
class Ending(GBComponent):
    """Decides whether to stop, before or after the rest of the turn.

    ``a | b`` builds an :class:`EndingChain` that stops when *either* says so.
    """

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        """Whether to terminate the negotiation now."""
        raise NotImplementedError

    def __or__(self, other: Ending) -> Ending:
        """Stop when this component or ``other`` says to."""
        if not isinstance(other, Ending):
            return NotImplemented
        return EndingChain(components=_leaves_of(self) + _leaves_of(other))


@define
class Validation(GBComponent):
    """Checks that the utterance is true of the decision. Always last.

    A validator may replace the *utterance*. It may not change the outcome or
    the response type — a component that rewrites the outcome is a second
    bidding policy in disguise and belongs in that slot.

    ``a | b`` builds a :class:`ValidationChain`: every check must pass, and
    each one sees the previous one's rewrite.
    """

    def validate(self, ctx: TurnContext) -> ValidationResult:
        """Check the utterance against the decision."""
        raise NotImplementedError

    def __or__(self, other: Validation) -> Validation:
        """Require this check *and* ``other`` to pass."""
        if not isinstance(other, Validation):
            return NotImplemented
        return ValidationChain(components=_leaves_of(self) + _leaves_of(other))


# =============================================================================
# Chains — several components in one slot
# =============================================================================


def _ordered_unique(items: Any) -> tuple[Any, ...]:
    """``items`` with duplicates dropped, **order preserved**.

    Not a ``set``: perceived acts are joined straight into prompts (see
    `~negmas_llm.pablove_components.LLMOffering.__call__`), and set iteration
    order would make those prompts differ run to run for no reason.
    """
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        key = item if isinstance(item, str) else repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


#: How informative each `PerceptionResult.source` is. A chain reports the best
#: source any of its members achieved: if one perceiver read typed data off the
#: wire, the whole reading is at least partly exact, and downstream analyses
#: that split results by perception quality should see that.
_SOURCE_RANK: dict[str, int] = {"none": 0, "classified": 1, "wire": 2}


@define
class PerceptionChain(ComponentChain, Perception):
    """Several perceivers, one merged reading.

    Every member runs — they observe rather than decide, so there is nothing to
    arbitrate — and their results are unioned: acts and commitments are
    concatenated with duplicates dropped, and for the single-valued fields
    (``sentiment``, ``text``, and each key of ``extracted``) the **first**
    member to produce a value wins. Order therefore expresses priority: put the
    perceiver you trust most first.

    Attributes:
        components: The perceivers, in priority order.
    """

    components: tuple[Perception, ...] = afield(factory=tuple, converter=tuple)

    def perceive(self, ctx: TurnContext) -> PerceptionResult:
        """Run every member and merge what they saw."""
        acts: list[str] = []
        commitments: list[dict[str, Any]] = []
        extracted: dict[str, Any] = {}
        sentiment: str | None = None
        text: str | None = None
        source: str = "none"
        for comp in self.components:
            result = comp.perceive(ctx)
            acts.extend(result.acts)
            commitments.extend(result.commitments)
            for key, value in result.extracted.items():
                extracted.setdefault(key, value)
            if sentiment is None:
                sentiment = result.sentiment
            if text is None:
                text = result.text
            if _SOURCE_RANK.get(result.source, 0) > _SOURCE_RANK.get(source, 0):
                source = result.source
        return PerceptionResult(
            acts=_ordered_unique(acts),
            commitments=_ordered_unique(commitments),
            sentiment=sentiment,
            extracted=extracted,
            source=source,  # type: ignore[arg-type]
            text=text,
        )


@define
class LanguageChain(ComponentChain, Language):
    """Several language components, each revising the last one's draft.

    The first member writes the message; every later one is handed that draft
    through ``ctx.utterance`` and returns its own version, so a chain reads as
    "write it, then make it more concrete, then soften it". This is the same
    field the pipeline itself fills in after `realize` returns, so a component
    that already knows how to revise a draft (as
    `~negmas_llm.pablove_components.LLMLanguage` does) needs no chain-specific
    code — and one that ignores ``ctx.utterance`` simply overwrites the draft,
    which is the honest reading of "it does not revise".

    ``ctx.utterance`` is restored before returning, so the pipeline installs the
    result exactly once and a chain is indistinguishable from a single
    component from the outside.

    Attributes:
        components: The language components, in the order they write.
    """

    components: tuple[Language, ...] = afield(factory=tuple, converter=tuple)

    def realize(self, ctx: TurnContext) -> Utterance:
        """Draft, then revise, returning the last member's version."""
        original = ctx.utterance
        utterance = original
        try:
            for comp in self.components:
                utterance = comp.realize(ctx)
                ctx.utterance = utterance
        finally:
            ctx.utterance = original
        return utterance if utterance is not None else Utterance()


@define
class ValidationChain(ComponentChain, Validation):
    """Several checks, all of which must pass.

    Every member runs even after one fails, so the caller learns about *all* the
    problems rather than only the first; issues are concatenated in order. Each
    member sees the previous member's rewrite through ``ctx.utterance``, so
    fixes compose instead of overwriting each other, and ``ctx.utterance`` is
    restored before returning — the pipeline, not the chain, decides whether to
    install a rewrite.

    A chain that produced a rewrite always reports ``ok=False``, even when no
    member actually complained: `PABLOveNegotiator._run_validation` returns
    immediately on an ``ok`` result and would otherwise discard the rewrite.
    A rewrite that changed nothing is not a rewrite, though — reporting one
    would make the pipeline re-run the chain against the same text until it
    ran out of attempts and (under ``resolution="strict"``) blank a message no
    member ever objected to, which is not what a single such member does.

    Attributes:
        components: The checks, in the order they run.
    """

    components: tuple[Validation, ...] = afield(factory=tuple, converter=tuple)

    def validate(self, ctx: TurnContext) -> ValidationResult:
        """Run every check, composing any rewrites they produce."""
        original = ctx.utterance
        issues: list[str] = []
        revised: Utterance | None = None
        ok = True
        try:
            for comp in self.components:
                result = comp.validate(ctx)
                if not result.ok:
                    ok = False
                    issues.extend(result.issues)
                if result.revised is not None:
                    revised = result.revised
                    ctx.utterance = revised
        finally:
            ctx.utterance = original
        if revised is not None and original is not None and revised == original:
            revised = None
        if revised is not None:
            return ValidationResult(
                ok=False, issues=tuple(issues) or ("rewritten",), revised=revised
            )
        return ValidationResult(ok=ok, issues=tuple(issues))


@define
class EndingChain(ComponentChain, Ending):
    """Several termination checks, combined by ``mode``.

    ``"any"`` (the default) stops as soon as one member says to and does not
    consult the rest — walking away is the conservative-to-trigger decision, and
    each member is already individually guarded (see
    `~negmas_llm.pablove_components.LLMEnding`). ``"all"`` requires unanimity
    and therefore runs every member, which is the right mode when members are
    cheap heuristics that each veto rather than each decide.

    Attributes:
        components: The termination checks, in the order they run.
        mode: ``"any"`` — stop if any member says stop (short-circuits);
            ``"all"`` — stop only if every member does.
    """

    components: tuple[Ending, ...] = afield(factory=tuple, converter=tuple)
    mode: Literal["any", "all"] = afield(default="any")

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        """Whether to walk away, per :attr:`mode`."""
        if not self.components:
            return EndingDecision(end=False, reason="no ending components")
        reasons: list[str] = []
        for comp in self.components:
            decision = comp.should_end(ctx)
            if decision.reason:
                reasons.append(decision.reason)
            if decision.end and self.mode == "any":
                return EndingDecision(end=True, reason=decision.reason)
            if not decision.end and self.mode == "all":
                return EndingDecision(end=False, reason=decision.reason)
        return EndingDecision(end=self.mode == "all", reason="; ".join(reasons))


#: Which chain fills which slot. A slot whose kind is not here (acceptance,
#: offering, opponent model — all negmas' own types) cannot be chained, since
#: "combine two bidding strategies" has no single obvious meaning, unlike
#: "merge two readings" or "apply both checks".
_CHAIN_TYPES: dict[type, type] = {
    Perception: PerceptionChain,
    Language: LanguageChain,
    Validation: ValidationChain,
    Ending: EndingChain,
}


# =============================================================================
# Building components from types, names and parameters
# =============================================================================


def resolve_component_type(name: str) -> type:
    """The class named ``name``.

    A dotted path is imported as written. A bare name is looked up in
    ``negmas_llm``'s own namespace first, then negmas' component and top-level
    namespaces — so ``"LLMPerception"`` and ``"AspirationOfferingPolicy"`` both
    work without the caller spelling out a module.

    Args:
        name: A class name, bare or fully qualified.

    Returns:
        The class.

    Raises:
        ValueError: If no class by that name can be found.
    """
    if "." not in name:
        for module_name in (
            "negmas_llm",
            "negmas.gb.components",
            "negmas.gb.components.models.ufun",
            "negmas",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:  # pragma: no cover - negmas is a hard dependency
                continue
            found = getattr(module, name, None)
            if isinstance(found, type):
                return found
        raise ValueError(
            f"Cannot find a component class named {name!r} in negmas_llm or negmas. "
            "Pass a fully qualified name (e.g. 'mypkg.mymod.MyPerception')."
        )
    try:
        found = get_class(name)
    except Exception as e:  # noqa: BLE001 - re-raised with a better message
        raise ValueError(f"Cannot import the component class {name!r}: {e}") from e
    if not isinstance(found, type):
        raise ValueError(f"{name!r} names a {type(found).__name__}, not a class.")
    return found


def make_component(
    spec: ComponentSpec,
    kind: type | None = None,
    params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    slot: str = "component",
) -> Any:
    """Build one PABLO-ve component from a :data:`ComponentSpec`.

    This is what lets every slot accept a *type* (with parameters) instead of a
    ready-made object, the way negmas' own ``*_type`` / ``*_params`` pairs do,
    and what makes ``[a, b]`` mean the same as ``a | b``.

    Args:
        spec: The specification — see :data:`ComponentSpec` for the grammar.
        kind: The base class this slot requires. Used to pick the chain type
            and to reject a component that belongs in another slot; ``None``
            skips both checks.
        params: Keyword arguments for a type/name spec. Against a list spec,
            either one dict applied to every element or a list of dicts (or
            ``None`` s) matched element-wise. A ``(spec, params)`` tuple's own
            parameters override these.
        slot: The slot's name, for error messages only.

    Returns:
        The component, a chain of them, or ``None`` when ``spec`` is ``None``.

    Raises:
        ValueError: For an unusable spec — an unchainable slot given several
            components, parameters supplied for an already-built instance, a
            per-element parameter list of the wrong length, or a component of
            the wrong kind.
    """
    if spec is None:
        return None
    # `A | B` over *types* is evaluated by Python into a union before it ever
    # reaches us; treat it as the list it was meant to be.
    if isinstance(spec, types.UnionType):
        spec = list(typing.get_args(spec))
    if isinstance(spec, tuple):
        if len(spec) != 2 or not isinstance(spec[1], (dict, type(None))):
            raise ValueError(
                f"{slot}: a tuple spec must be (spec, params), got {spec!r}. "
                "Use a list to chain several components."
            )
        inner, own = spec
        merged = dict(params) if isinstance(params, dict) else {}
        merged.update(own or {})
        return make_component(inner, kind, merged or None, slot)
    if isinstance(spec, list):
        each = _params_for_each(spec, params, slot)
        parts = [
            make_component(s, kind, p, slot)
            for s, p in zip(spec, each, strict=True)
            if s is not None
        ]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        chain_type = _CHAIN_TYPES.get(kind) if kind is not None else None
        if chain_type is None:
            raise ValueError(
                f"{slot}: cannot combine {len(parts)} components — only "
                "perception, language, validation and ending can be chained."
            )
        return chain_type(components=tuple(parts))
    if isinstance(spec, str):
        spec = resolve_component_type(spec)
    if isinstance(spec, type):
        kwargs: dict[str, Any] = dict(params) if isinstance(params, dict) else {}
        component = spec(**kwargs)
    else:
        if params:
            raise ValueError(
                f"{slot}: parameters were given for an already-constructed "
                f"{type(spec).__name__}. Pass the type instead of an instance, "
                "or build the instance with those parameters yourself."
            )
        component = spec
    if kind is not None and not isinstance(component, kind):
        raise ValueError(
            f"{slot}: expected a {kind.__name__}, got {type(component).__name__}."
        )
    return component


def _params_for_each(
    spec: list[Any],
    params: dict[str, Any] | list[dict[str, Any] | None] | None,
    slot: str,
) -> list[dict[str, Any] | None]:
    """Line up ``params`` with the elements of a list spec."""
    if params is None or isinstance(params, dict):
        return [params] * len(spec)  # type: ignore[list-item]
    if len(params) != len(spec):
        raise ValueError(
            f"{slot}: got {len(params)} parameter dicts for {len(spec)} components. "
            "Pass one dict per component, or a single dict to use for all of them."
        )
    return list(params)


def _make_components(
    spec: ComponentSpec,
    kind: type,
    params: dict[str, Any] | list[dict[str, Any] | None] | None,
    slot: str,
) -> list[Any]:
    """:func:`make_component`, but a list spec yields *several* components.

    For the opponent-model slot, which negmas itself accepts a list for
    (``MAPNegotiator(models=[...])``) — so "two models" needs no chain type.
    """
    if isinstance(spec, types.UnionType):
        spec = list(typing.get_args(spec))
    if isinstance(spec, list):
        each = _params_for_each(spec, params, slot)
        built = [
            make_component(s, kind, p, slot) for s, p in zip(spec, each, strict=True)
        ]
        return [c for c in built if c is not None]
    component = make_component(spec, kind, params, slot)
    return [] if component is None else [component]


# =============================================================================
# The negotiator
# =============================================================================


class PABLOveNegotiator(MAPNegotiator):
    """A negotiator assembled from PABLO-ve components.

    Extends :class:`~negmas.gb.negotiators.modular.mapneg.MAPNegotiator` — the
    class ``BOANegotiator`` itself extends — so the BOA core (model, acceptance,
    offering) behaves exactly as before and the new components are registered
    through the same path, keeping lifecycle callbacks, ``share_ufun`` /
    ``share_nmi`` and registry support intact.

    Every slot takes a :data:`ComponentSpec`, not just a built component: a
    type (built with that slot's ``*_params``), a class name, a
    ``(type, params)`` pair, or a list of any of those, which is combined into
    a chain exactly as ``|`` would. So these three are the same negotiator::

        PABLOveNegotiator(perception=LLMPerception() | WirePerception())
        PABLOveNegotiator(perception=[LLMPerception, WirePerception])
        PABLOveNegotiator(
            perception=["LLMPerception", "WirePerception"],
            perception_params=[dict(model="gpt-4o"), None],
        )

    Args:
        acceptance: ``A`` — acceptance policy (BOA, unchanged).
        offering: ``B`` — offering/bidding policy (BOA, unchanged).
        model: ``O`` — opponent model (BOA, unchanged). A list gives several
            models, as ``MAPNegotiator`` already allows — note that only the
            first ``UFunModel`` among them becomes ``opponent_ufun``, and
            negmas warns when there is more than one.
        perception: ``P`` — partner-utterance understanding.
        language: ``L`` — utterance generation.
        validation: ``v`` — consistency check over the utterance.
        ending: ``e`` — termination decision.
        acceptance_params: Keyword arguments used when ``acceptance`` is given
            as a type or a name (ignored for an already-built component, which
            raises instead of silently dropping them). The other ``*_params``
            behave identically for their own slot; against a list spec, pass
            either one dict for all of them or one dict per component.
        offering_params: See ``acceptance_params``.
        model_params: See ``acceptance_params``.
        perception_params: See ``acceptance_params``.
        language_params: See ``acceptance_params``.
        validation_params: See ``acceptance_params``.
        ending_params: See ``acceptance_params``.
        ending_when: Run ``ending`` before the decision (``"early"``) or after
            it (``"late"``). Ignored when no ending component is given.
        resolution: What to do when validation fails. ``"text"`` regenerates the
            utterance (bounded by ``max_revalidations``), ``"strict"`` drops to a
            safe empty utterance, ``"log"`` records and sends anyway.
        max_revalidations: Cap on regeneration attempts; unbounded would loop.
        acceptance_first: Run order of acceptance and offering (BOA, unchanged).
        memory_mode: When given, uniformly overrides ``memory_mode`` on every
            attached :class:`~negmas_llm.pablove_components.LLMComponent` --
            ``"none"`` (today's original stateless behavior), ``"conversation"``
            (each component keeps its own persistent chat -- the default a
            component picks for itself if this stays ``None``), or ``"shared"``
            (a single setup block cached here on the negotiator -- see
            :attr:`memory`, :meth:`on_preferences_changed` -- and a bounded
            recap pulled fresh into each component's call, rather than a
            per-component growing chat). ``None`` (default) leaves each
            component's own setting untouched, so a mix of modes across
            components is possible when built explicitly.
        **kwargs: Forwarded to ``MAPNegotiator``.

    Remarks:
        With ``perception``, ``language``, ``validation`` and ``ending`` all
        ``None`` this negotiator is behaviourally identical to
        ``make_boa(acceptance=..., offering=..., model=...)``.

        **Joint bidding/acceptance policies** (a learned policy that emits both
        from one forward pass) are supported without a new slot: put the policy
        in ``offering``, let it write its decision into ``ctx``, and have the
        acceptance policy read ``ctx.bid`` — or simply call ``propose()``, which
        negmas caches per ``(step, thread)`` for exactly this purpose. One
        forward pass, two slots.

        **Cross-turn state** (phase variables and similar) belongs in
        :attr:`shared`, not in :class:`TurnContext`, which is turn-scoped.
    """

    def __init__(
        self,
        *args,
        acceptance: ComponentSpec = None,
        offering: ComponentSpec = None,
        model: ComponentSpec = None,
        perception: ComponentSpec = None,
        language: ComponentSpec = None,
        validation: ComponentSpec = None,
        ending: ComponentSpec = None,
        acceptance_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        offering_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        model_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        perception_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        language_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        validation_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        ending_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        ending_when: Literal["early", "late"] = "early",
        resolution: Literal["text", "strict", "log"] = "text",
        max_revalidations: int = 1,
        acceptance_first: bool = True,
        extra_components: list[GBComponent] | None = None,
        extra_component_names: list[str] | None = None,
        memory_mode: Literal["none", "conversation", "shared"] | None = None,
        **kwargs,
    ):
        # Specs become components before anything else: `is_plain_boa`, the
        # `models=` path and the `memory_mode` override all read real objects.
        acceptance = make_component(
            acceptance, AcceptancePolicy, acceptance_params, "acceptance"
        )
        offering = make_component(offering, OfferingPolicy, offering_params, "offering")
        models = _make_components(model, Model, model_params, "model")
        perception = make_component(
            perception, Perception, perception_params, "perception"
        )
        language = make_component(language, Language, language_params, "language")
        validation = make_component(
            validation, Validation, validation_params, "validation"
        )
        ending = make_component(ending, Ending, ending_params, "ending")

        extra = list(extra_components) if extra_components else []
        names = (
            list(extra_component_names)
            if extra_component_names
            else [f"extra{i}" for i in range(len(extra))]
        )
        # Registered as ordinary components so they receive every lifecycle
        # callback; the pipeline calls them explicitly at the right moment.
        # A chain is *not* registered — its leaves are, individually, so each
        # one is treated exactly as it would be standing alone (own lifecycle
        # callbacks, own `memory_mode` override, one memory-seeding call).
        # Names are indexed because `ModularNegotiator`'s component map is a
        # dict, where a repeated name would silently shadow.
        for comp, name in (
            (perception, "perception"),
            (language, "language"),
            (validation, "validation"),
            (ending, "ending"),
        ):
            leaves = component_leaves(comp)
            for i, leaf in enumerate(leaves):
                extra.append(leaf)
                names.append(name if len(leaves) == 1 else f"{name}[{i}]")

        self._perception = perception
        self._language = language
        self._validation = validation
        self._ending = ending
        self._ending_when = ending_when
        self._resolution = resolution
        self._max_revalidations = max_revalidations

        self._turns: list[TurnContext] = []
        self._turn: TurnContext | None = None
        #: Cross-turn state that no single component owns — phase variables,
        #: counters, anything that outlives a turn. Deliberately separate from
        #: `TurnContext`, which is strictly turn-scoped; conflating the two is
        #: how "which turn was that from?" bugs start.
        self.shared: dict[str, Any] = {}
        #: Rescued in `respond_`, read in `respond`: see that method.
        self._pending_their_offer: Outcome | None = None
        self._pending_their_data: dict[str, Any] | None = None
        #: Cached "shared" mode setup block; see :attr:`memory`. Unused by
        #: components in "none"/"conversation" mode.
        self._memory_text: str | None = None

        super().__init__(
            *args,
            acceptance=acceptance,
            offering=offering,
            models=models or None,
            model_names=[f"model{i}" if i else "model" for i in range(len(models))]
            or None,
            extra_components=extra or None,
            extra_component_names=names or None,
            acceptance_first=acceptance_first,
            **kwargs,
        )
        if memory_mode is not None:
            for comp in self._components:
                if hasattr(comp, "memory_mode"):
                    comp.memory_mode = memory_mode

    # -- context ----------------------------------------------------------

    @property
    def turn(self) -> TurnContext | None:
        """The turn currently being executed (``None`` outside a turn)."""
        return self._turn

    @property
    def turns(self) -> tuple[TurnContext, ...]:
        """Every completed turn, oldest first."""
        return tuple(self._turns)

    @property
    def is_plain_boa(self) -> bool:
        """True when no PABLO-ve-only component is configured."""
        return not any(
            (self._perception, self._language, self._validation, self._ending)
        )

    @property
    def opponent_model(self) -> Model | None:
        """Alias ``self.opponent_ufun`` under the name some policies expect.

        `MAPNegotiator.__init__` already filters ``models=`` for `UFunModel`
        instances and registers the first one as ``self.private_info["opponent_ufun"]``
        (exposed via the base `Negotiator.opponent_ufun` property) -- so a
        `PABLOveNegotiator` constructed with a `UFunModel` ``model=`` already
        has this, correctly, with no extra wiring. The only real gap: some of
        negmas's own offering/acceptance policies (e.g.
        ``NiceTitForTatOfferingPolicy``) read ``self.negotiator.opponent_model``
        instead of ``opponent_ufun`` -- a naming difference from
        `NiceTitForTatNegotiator`'s own property, not a missing mechanism.
        This is that one-line bridge, not a reimplementation.
        """
        return self.opponent_ufun

    def _open_turn(
        self,
        entry: Literal["propose", "respond"],
        state: GBState,
        source: str | None = None,
        dest: str | None = None,
        their_offer: Outcome | None = None,
        their_data: dict[str, Any] | None = None,
    ) -> TurnContext:
        nmi = getattr(self, "nmi", None)
        ctx = TurnContext(
            entry=entry,
            state=state,
            source=source,
            dest=dest,
            their_offer=their_offer,
            their_data=their_data,
            n_steps=getattr(nmi, "n_steps", None),
            time_limit=getattr(nmi, "time_limit", None),
        )
        # Shared reference, not a copy: see TurnContext.history.
        ctx._history_all = self._turns
        ctx._history_len = len(self._turns)
        self._turn = ctx
        return ctx

    def _close_turn(self) -> None:
        if self._turn is not None:
            self._turns.append(self._turn)
        self._turn = None

    # -- pipeline steps ---------------------------------------------------

    def _run_perception(self, ctx: TurnContext) -> None:
        if self._perception is None:
            return
        ctx.perception = self._perception.perceive(ctx)

    def _run_ending(self, ctx: TurnContext, when: str) -> bool:
        """Consult the ending component; return True when it says stop."""
        if self._ending is None or self._ending_when != when:
            return False
        decision = self._ending.should_end(ctx)
        ctx.ending = decision
        return bool(decision.end)

    def _run_language(self, ctx: TurnContext) -> None:
        if self._language is None:
            return
        ctx.utterance = self._language.realize(ctx)

    def _run_validation(self, ctx: TurnContext) -> None:
        """Validate the utterance, repairing it up to ``max_revalidations``."""
        if self._validation is None or ctx.utterance is None:
            return
        for _ in range(max(0, self._max_revalidations) + 1):
            result = self._validation.validate(ctx)
            ctx.validation = result
            if result.ok:
                return
            if self._resolution == "log":
                return
            if result.revised is not None:
                ctx.utterance = result.revised
                ctx.revalidations += 1
                continue
            break
        if self._resolution == "strict" and not (ctx.validation and ctx.validation.ok):
            # Irreparable: say nothing rather than say something untrue.
            ctx.utterance = Utterance()

    # -- SAO entry points -------------------------------------------------

    def propose(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Bid, then say it.

        Runs ``ending`` (early), the BOA core, ``language``, ``validation`` and
        ``ending`` (late). With no PABLO-ve components this is exactly
        ``MAPNegotiator.propose``.
        """
        if self.is_plain_boa:
            return super().propose(state, dest=dest)

        ctx = self._open_turn("propose", state, dest=dest)
        try:
            if self._run_ending(ctx, "early"):
                return None
            offer = super().propose(state, dest=dest)
            ctx.bid = offer
            if self._run_ending(ctx, "late"):
                return None
            self._run_language(ctx)
            self._run_validation(ctx)
            return self._attach(offer, ctx)
        finally:
            self._close_turn()

    def respond_(
        self, state: Any, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Rescue the partner's offer and payload before they can be dropped.

        When this negotiator runs inside an ``SAOMechanism`` (the common
        case: PABLO-ve components are ordinary GB negotiators wrapped for the
        classic alternating-offers protocol), the mechanism calls
        ``respond_`` with a real ``SAOState`` whose ``current_offer`` /
        ``current_data`` hold the partner's move. ``GBNegotiator.respond_``
        (the base class) then converts it into the ``GBState`` that
        :meth:`respond` sees via ``_gb_state_from_sao_state`` — but that
        conversion builds ``state.threads`` from ``self.nmi.negotiator_ids``,
        which is empty for this adapter, so ``state.threads`` ends up with no
        entries at all. Both fields are lost unless captured here first.
        Under a native ``GBMechanism`` the mechanism calls :meth:`respond`
        directly, bypassing this method entirely, and
        ``state.threads[source].new_offer`` / ``.new_data`` (read in
        :meth:`respond`) are already correct.
        """
        self._pending_their_offer = getattr(state, "current_offer", None)
        self._pending_their_data = getattr(state, "current_data", None)
        return super().respond_(state, source=source)

    def respond(
        self, state: GBState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Perceive, decide, then say it.

        Perception runs here because this is where the partner's offer and
        message arrive; ``propose`` reuses the same step's result via
        :meth:`TurnContext.perception_this_step`.
        """
        if self.is_plain_boa:
            return super().respond(state, source=source)

        # GBState has no `current_offer` (that's an SAOState/STState thing);
        # the offer and its accompanying payload normally arrive as two
        # separate fields on the sender's thread. Under the SAO-mechanism
        # compatibility shim that thread is missing entirely (see
        # `respond_`), so fall back to what was rescued there.
        thread = state.threads.get(source) if source else None
        their_offer = (thread.new_offer if thread else None) or (
            self._pending_their_offer
        )
        their_data = (thread.new_data if thread else None) or (self._pending_their_data)
        ctx = self._open_turn(
            "respond",
            state,
            source=source,
            their_offer=their_offer,
            their_data=their_data,
        )
        try:
            self._run_perception(ctx)
            if self._run_ending(ctx, "early"):
                return ResponseType.END_NEGOTIATION
            response = super().respond(state, source=source)
            rtype = (
                response.response
                if isinstance(response, ExtendedResponseType)
                else response
            )
            ctx.acceptance = rtype
            if self._run_ending(ctx, "late"):
                return ResponseType.END_NEGOTIATION
            self._run_language(ctx)
            self._run_validation(ctx)
            return self._attach_response(response, ctx)
        finally:
            self._close_turn()

    # -- attaching the utterance -----------------------------------------

    def _attach(
        self, offer: Outcome | ExtendedOutcome | None, ctx: TurnContext
    ) -> Outcome | ExtendedOutcome | None:
        """Merge the utterance into the proposal's data payload."""
        data = ctx.utterance.as_data() if ctx.utterance else {}
        if not data:
            return offer
        if isinstance(offer, ExtendedOutcome):
            return ExtendedOutcome(
                outcome=offer.outcome, data={**(offer.data or {}), **data}
            )
        # A None outcome with data is a legitimate dialogue-only turn: negmas
        # keeps the negotiation running when a message carries no offer.
        return ExtendedOutcome(outcome=offer, data=data)

    def _attach_response(
        self, response: ResponseType | ExtendedResponseType, ctx: TurnContext
    ) -> ResponseType | ExtendedResponseType:
        """Merge the utterance into the response's data payload."""
        data = ctx.utterance.as_data() if ctx.utterance else {}
        if isinstance(response, ExtendedResponseType):
            merged = {**(response.data or {}), **data}
            if not merged:
                return response.response
            return ExtendedResponseType(response=response.response, data=merged)
        if not data:
            return response
        return ExtendedResponseType(response=response, data=data)

    # -- "shared" memory-mode setup block ----------------------------------

    @property
    def memory(self) -> str:
        """The cached "shared" mode setup block: NMI, outcome space, own
        utility function/reserved value, and the opponent's utility function
        when known.

        Rendered once, from `~negmas_llm.negotiator.DEFAULT_PREFERENCES_PROMPT`
        via the same `negmas_llm.tags.process_prompt` renderer
        `~negmas_llm.negotiator.LLMNegotiator` uses for the same content, so
        this is genuinely the same information, not a re-derived copy. Empty
        until preferences are known (see :meth:`on_preferences_changed`); no
        component *requires* this to be non-empty -- ``"shared"`` mode
        components degrade to an empty block, same as no memory at all.
        """
        return self._memory_text or ""

    def _refresh_memory(self, changes: list[PreferencesChange] | None) -> None:
        if self.ufun is None:
            return
        if changes:
            change_types = ", ".join(c.type.name for c in changes) or "unspecified"
            template = DEFAULT_PREFERENCES_CHANGED_PROMPT.format(
                change_types=change_types
            )
        else:
            template = DEFAULT_PREFERENCES_PROMPT
        self._memory_text = process_prompt(template, self, None)

    # -- lifecycle --------------------------------------------------------

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Dispatch to every component (base behavior), then refresh :attr:`memory`."""
        was_known = self._memory_text is not None
        super().on_preferences_changed(changes)
        self._refresh_memory(changes if was_known else None)

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Reset the turn log, and make sure :attr:`memory` reflects the
        final NMI/outcome space now that this negotiator has joined a
        mechanism -- `on_preferences_changed` may have run before that, when
        NMI-dependent tags had nothing to render yet."""
        super().on_negotiation_start(state)
        self._turns = []
        self._turn = None
        self.shared = {}
        self._pending_their_offer = None
        self._pending_their_data = None
        self._refresh_memory(None)


def make_pablove(
    acceptance: ComponentSpec = None,
    offering: ComponentSpec = None,
    model: ComponentSpec = None,
    perception: ComponentSpec = None,
    language: ComponentSpec = None,
    validation: ComponentSpec = None,
    ending: ComponentSpec = None,
    **kwargs,
) -> PABLOveNegotiator:
    """Assemble a PABLO-ve negotiator.

    Mirrors ``negmas.gb.negotiators.modular.boa.make_boa`` and reduces to it
    when only the BOA slots are given.

    Args:
        acceptance: ``A`` acceptance policy.
        offering: ``B`` offering/bidding policy.
        model: ``O`` opponent model.
        perception: ``P`` partner-utterance understanding.
        language: ``L`` utterance generation.
        validation: ``v`` utterance/decision consistency check.
        ending: ``e`` termination decision.
        **kwargs: Forwarded to :class:`PABLOveNegotiator` — including every
            ``*_params`` argument, since each slot here takes a full
            :data:`ComponentSpec` (an instance, a type, a name, a
            ``(spec, params)`` pair, or a list of them) rather than only a
            built component.

    Returns:
        The assembled negotiator.
    """
    return PABLOveNegotiator(
        acceptance=acceptance,
        offering=offering,
        model=model,
        perception=perception,
        language=language,
        validation=validation,
        ending=ending,
        type_name="PABLOveNegotiator",
        **kwargs,
    )

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from attrs import define
from negmas.common import MechanismState
from negmas.gb.common import ExtendedResponseType, GBState
from negmas.gb.components import AcceptancePolicy, GBComponent, Model, OfferingPolicy
from negmas.gb.negotiators.modular.mapneg import MAPNegotiator
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.sao import ResponseType

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
    "PABLOveNegotiator",
    "make_pablove",
]


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
    """

    entry: Literal["propose", "respond"]
    state: GBState
    source: str | None = None
    dest: str | None = None
    their_offer: Outcome | None = None

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

    history: tuple[TurnContext, ...] = ()

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
        for ctx in reversed(self.history):
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
        for ctx in reversed(self.history):
            if ctx.step != self.step:
                break
            if ctx.perception is not None:
                return ctx.perception
        return None


# =============================================================================
# New component base classes
# =============================================================================


@define
class Perception(GBComponent):
    """Turns the partner's utterance into structure. Decides nothing.

    Always the first step of a turn. Runs on ``respond`` (where the partner's
    message arrives); ``propose`` reuses the same step's result.
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


@define
class Language(GBComponent):
    """Turns a decision into an utterance. May not change the decision."""

    def realize(self, ctx: TurnContext) -> Utterance:
        """Say what was decided.

        Args:
            ctx: Turn context with ``bid`` / ``acceptance`` already filled in.

        Returns:
            The utterance to send.
        """
        raise NotImplementedError


@define
class Ending(GBComponent):
    """Decides whether to stop, before or after the rest of the turn."""

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        """Whether to terminate the negotiation now."""
        raise NotImplementedError


@define
class Validation(GBComponent):
    """Checks that the utterance is true of the decision. Always last.

    A validator may replace the *utterance*. It may not change the outcome or
    the response type — a component that rewrites the outcome is a second
    bidding policy in disguise and belongs in that slot.
    """

    def validate(self, ctx: TurnContext) -> ValidationResult:
        """Check the utterance against the decision."""
        raise NotImplementedError


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

    Args:
        acceptance: ``A`` — acceptance policy (BOA, unchanged).
        offering: ``B`` — offering/bidding policy (BOA, unchanged).
        model: ``O`` — opponent model (BOA, unchanged).
        perception: ``P`` — partner-utterance understanding.
        language: ``L`` — utterance generation.
        validation: ``v`` — consistency check over the utterance.
        ending: ``e`` — termination decision.
        ending_when: Run ``ending`` before the decision (``"early"``) or after
            it (``"late"``). Ignored when no ending component is given.
        resolution: What to do when validation fails. ``"text"`` regenerates the
            utterance (bounded by ``max_revalidations``), ``"strict"`` drops to a
            safe empty utterance, ``"log"`` records and sends anyway.
        max_revalidations: Cap on regeneration attempts; unbounded would loop.
        acceptance_first: Run order of acceptance and offering (BOA, unchanged).
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
        acceptance: AcceptancePolicy | None = None,
        offering: OfferingPolicy | None = None,
        model: Model | None = None,
        perception: Perception | None = None,
        language: Language | None = None,
        validation: Validation | None = None,
        ending: Ending | None = None,
        ending_when: Literal["early", "late"] = "early",
        resolution: Literal["text", "strict", "log"] = "text",
        max_revalidations: int = 1,
        acceptance_first: bool = True,
        extra_components: list[GBComponent] | None = None,
        extra_component_names: list[str] | None = None,
        **kwargs,
    ):
        extra = list(extra_components) if extra_components else []
        names = (
            list(extra_component_names)
            if extra_component_names
            else [f"extra{i}" for i in range(len(extra))]
        )
        # Registered as ordinary components so they receive every lifecycle
        # callback; the pipeline calls them explicitly at the right moment.
        for comp, name in (
            (perception, "perception"),
            (language, "language"),
            (validation, "validation"),
            (ending, "ending"),
        ):
            if comp is not None:
                extra.append(comp)
                names.append(name)

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

        super().__init__(
            *args,
            acceptance=acceptance,
            offering=offering,
            models=[model] if model else None,
            model_names=["model"] if model else None,
            extra_components=extra or None,
            extra_component_names=names or None,
            acceptance_first=acceptance_first,
            **kwargs,
        )

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

    def _open_turn(
        self,
        entry: Literal["propose", "respond"],
        state: GBState,
        source: str | None = None,
        dest: str | None = None,
        their_offer: Outcome | None = None,
    ) -> TurnContext:
        ctx = TurnContext(
            entry=entry,
            state=state,
            source=source,
            dest=dest,
            their_offer=their_offer,
            history=tuple(self._turns),
        )
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

        their_offer = getattr(state, "current_offer", None)
        ctx = self._open_turn("respond", state, source=source, their_offer=their_offer)
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

    # -- lifecycle --------------------------------------------------------

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Reset the turn log."""
        super().on_negotiation_start(state)
        self._turns = []
        self._turn = None
        self.shared = {}


def make_pablove(
    acceptance: AcceptancePolicy | None = None,
    offering: OfferingPolicy | None = None,
    model: Model | None = None,
    perception: Perception | None = None,
    language: Language | None = None,
    validation: Validation | None = None,
    ending: Ending | None = None,
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
        **kwargs: Forwarded to :class:`PABLOveNegotiator`.

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

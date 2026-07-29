"""Ready-made PABLO-ve components, with and without an LLM.

The architecture in :mod:`negmas_llm.pablove` defines the *slots*; this module
fills the two that are new and generic enough to belong in the library:

* :class:`TemplateLanguage` — deterministic verbalization, **no LLM**. The
  baseline that isolates what an LLM actually contributes: any claim of the form
  "the LLM improves outcomes" has to beat this, not silence.
* :class:`LLMLanguage` — one LLM call turns the decision into an utterance.
* :class:`LLMPerception` — one LLM call turns the partner's utterance into
  structure.

Components are ``GBComponent`` s, not negotiators, so they have none of the
provider/model plumbing. :class:`LLMComponent` supplies it, resolving
configuration through the same precedence rules as every negotiator (explicit
argument, then ``NEGMAS_LLM_<ClassName>_<VAR>``, then ``NEGMAS_LLM_<VAR>``, then
the built-in default) so a component can be re-pointed at another model by
environment alone.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import litellm
from attrs import define, field
from negmas.gb.components import AcceptancePolicy, OfferingPolicy
from negmas.gb.components.models.ufun import UFunModel
from negmas.outcomes import ExtendedOutcome

from negmas_llm.common import (
    apply_effort,
    apply_max_tokens,
    apply_temperature,
    litellm_model_string,
    resolve_max_words,
    word_limit_instruction,
)
from negmas_llm.config import DEFAULT_PROVIDER, resolve_llm_config
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
)
from negmas_llm.token_usage import TokenUsage

__all__ = [
    "LLMComponent",
    "TemplateLanguage",
    "LLMLanguage",
    "LLMPerception",
    "LLMOffering",
    "LLMAcceptance",
    "LLMUFunModel",
    "LLMValidation",
    "LLMEnding",
    "snap_outcome",
    "outcome_space_of",
    "DEFAULT_LANGUAGE_PROMPT",
    "DEFAULT_PERCEPTION_PROMPT",
]


# =============================================================================
# LLM plumbing for components
# =============================================================================


# ``slots=False`` is required: ``GBComponent`` is itself a slotted attrs class,
# and combining two independently-slotted bases raises a layout conflict.
@define(slots=False)
class LLMComponent:
    """Mixin giving a `GBComponent` the library's LLM configuration and call path.

    Attributes:
        provider: LLM provider; ``None`` resolves from the environment.
        model: Model name; ``None`` resolves from the environment.
        temperature: Sampling temperature; ``None`` picks a model-appropriate one.
        max_tokens: Hard token ceiling; ``None`` sends no cap, so a reasoning
            model is never cut off mid-thought.
        max_words: Approximate length of the generated text, stated in the
            prompt. This — not ``max_tokens`` — is how answer length is bounded.
        verbose: Print prompts and responses.
        llm_kwargs: Extra keyword arguments for ``litellm.completion``.
    """

    provider: str | None = field(default=None, kw_only=True)
    model: str | None = field(default=None, kw_only=True)
    effort: str | None = field(default=None, kw_only=True)
    api_key: str | None = field(default=None, kw_only=True)
    api_base: str | None = field(default=None, kw_only=True)
    temperature: float | None = field(default=None, kw_only=True)
    max_tokens: int | None = field(default=None, kw_only=True)
    max_words: int | None = field(default=None, kw_only=True)
    timeout: float | int | None = field(default=None, kw_only=True)
    num_retries: int | None = field(default=None, kw_only=True)
    verbose: bool = field(default=False, kw_only=True)
    llm_kwargs: dict[str, Any] = field(factory=dict, kw_only=True)
    token_usage: TokenUsage = field(factory=TokenUsage, init=False)
    _resolved: Any = field(default=None, init=False)

    def _config(self):
        """Resolve (and cache) this component's LLM configuration."""
        if self._resolved is None:
            self._resolved = resolve_llm_config(
                type(self).__name__,
                provider=self.provider,
                model=self.model,
                effort=self.effort,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_words=self.max_words,
                timeout=self.timeout,
                num_retries=self.num_retries,
                default_provider=DEFAULT_PROVIDER,
            )
        return self._resolved

    @property
    def word_budget(self) -> int | None:
        """Approximate word budget for generated text."""
        return resolve_max_words(self._config().max_words)

    def call_llm(self, system: str, user: str) -> str:
        """Send one system/user exchange and return the raw response text.

        Args:
            system: The system prompt.
            user: The user message.

        Returns:
            The model's text response (empty string if it returned none).
        """
        cfg = self._config()
        kwargs: dict[str, Any] = {
            "model": litellm_model_string(cfg.provider, cfg.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.llm_kwargs,
        }
        apply_temperature(kwargs, cfg.provider, cfg.model, cfg.temperature)
        apply_max_tokens(kwargs, cfg.provider, cfg.model, cfg.max_tokens)
        apply_effort(kwargs, cfg.effort)
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.timeout is not None:
            kwargs["timeout"] = cfg.timeout
        if cfg.num_retries is not None:
            kwargs["num_retries"] = cfg.num_retries

        start = time.perf_counter()
        response = litellm.completion(**kwargs)
        elapsed = time.perf_counter() - start
        self.token_usage.add(response, seconds=elapsed)
        # Read structurally rather than by isinstance: litellm returns several
        # response types across providers, and an over-strict check would
        # silently yield an empty string instead of the model's answer.
        try:
            text = response.choices[0].message.content or ""  # type: ignore[index,union-attr]
        except (AttributeError, IndexError, TypeError):
            text = ""
        if self.verbose:
            print(
                f"[{type(self).__name__} {cfg.provider}/{cfg.model} "
                f"{elapsed:.1f}s]\n  << {user[-300:]}\n  >> {text[:300]}"
            )
        return text

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from a model response.

        Returns an empty dict rather than raising: a component that cannot parse
        should degrade, not abort the negotiation.
        """
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _outcome_of(bid: Any) -> Any:
    """The plain outcome behind a bid that may be an `ExtendedOutcome`."""
    return bid.outcome if isinstance(bid, ExtendedOutcome) else bid


# =============================================================================
# Language
# =============================================================================


@define
class TemplateLanguage(Language):
    """Deterministic verbalization — the no-LLM baseline.

    Exists to be beaten. Comparing an LLM narrator against *no text at all*
    conflates "language helps" with "this model's language helps"; comparing it
    against a template isolates the second.

    Attributes:
        propose_template: Format string with ``{outcome}`` and ``{step}``.
        accept_template: Format string with ``{outcome}``.
        reject_template: Format string with ``{outcome}``.
        end_template: Format string, no fields required.
    """

    propose_template: str = "I propose {outcome}."
    accept_template: str = "Agreed — {outcome} works for me."
    reject_template: str = "That does not work for me."
    end_template: str = "I do not think we can reach a deal."

    def realize(self, ctx: TurnContext) -> Utterance:
        """Fill the template matching this turn's decision."""
        from negmas.sao import ResponseType

        if ctx.entry == "propose":
            return Utterance(
                text=self.propose_template.format(
                    outcome=_outcome_of(ctx.bid), step=ctx.step
                )
            )
        if ctx.acceptance == ResponseType.ACCEPT_OFFER:
            return Utterance(text=self.accept_template.format(outcome=ctx.their_offer))
        if ctx.acceptance == ResponseType.END_NEGOTIATION:
            return Utterance(text=self.end_template)
        return Utterance(text=self.reject_template.format(outcome=ctx.their_offer))


DEFAULT_LANGUAGE_PROMPT = """You put a negotiation decision into words.

A separate strategy has already decided the action and the exact terms. Your job
is only to express that decision.

Rules:
    1. Never change the decision and never invent terms, prices or extras.
    2. Never reveal utilities, reserved values, or that a strategy chose this.
    3. Be natural and professional.

Respond with ONLY this JSON: {"text": "your message"}"""


@define(slots=False)
class LLMLanguage(LLMComponent, Language):
    """One LLM call turns the decision into an utterance.

    The decision is an *input*: this component cannot change the offer or the
    response type, which is what keeps agreements valid regardless of what the
    model writes.

    Attributes:
        system_prompt: Override the realization instructions.
        describe: Optional callable ``(ctx) -> str`` producing extra context for
            the prompt (elicited priorities, chosen speech acts, and so on).
    """

    system_prompt: str = DEFAULT_LANGUAGE_PROMPT
    describe: Any = field(default=None)

    def _system(self) -> str:
        limit = word_limit_instruction(self.word_budget)
        return f"{self.system_prompt}\n{limit}" if limit else self.system_prompt

    def _user(self, ctx: TurnContext) -> str:
        from negmas.sao import ResponseType

        parts = [f"Round {ctx.step} (time {ctx.relative_time:.0%})."]
        perception = ctx.perception_this_step()
        if perception is not None and perception.text:
            parts.append(f'They said: "{perception.text}"')
            if perception.acts:
                parts.append(f"Their move was: {', '.join(perception.acts)}.")
        if ctx.entry == "propose":
            parts.append(f"You are proposing {_outcome_of(ctx.bid)}.")
        elif ctx.acceptance == ResponseType.ACCEPT_OFFER:
            parts.append(f"You are ACCEPTING {ctx.their_offer}.")
        elif ctx.acceptance == ResponseType.END_NEGOTIATION:
            parts.append("You are ENDING the negotiation without agreement.")
        else:
            parts.append(f"You are REJECTING {ctx.their_offer}.")
        if self.describe is not None:
            extra = self.describe(ctx)
            if extra:
                parts.append(str(extra))
        return "\n".join(parts)

    def realize(self, ctx: TurnContext) -> Utterance:
        """Generate the utterance for this turn's decision."""
        raw = self.call_llm(self._system(), self._user(ctx))
        data = self.parse_json(raw)
        text = str(data.get("text") or "").strip() or raw.strip()
        return Utterance(text=text)


# =============================================================================
# Perception
# =============================================================================


DEFAULT_PERCEPTION_PROMPT = """You read one message from a negotiation partner
and report what it DID, as structure. You do not decide anything and you do not
reply.

Respond with ONLY this JSON:
{
    "acts": ["<what the message did, e.g. propose, concede, ask, refuse, threaten, agree>"],
    "sentiment": "positive|neutral|negative",
    "commitments": ["<any promise or limit they stated>"]
}"""


@define(slots=False)
class LLMPerception(LLMComponent, Perception):
    """One LLM call turns the partner's utterance into structure.

    Reads the partner's typed data off the wire when it is there (exact and
    free) and only calls the model otherwise, recording which happened in
    :attr:`PerceptionResult.source` so results can be split by perception
    quality.

    Attributes:
        system_prompt: Override the classification instructions.
        acts_key: Key under which a partner may publish its own typed acts.
    """

    system_prompt: str = DEFAULT_PERCEPTION_PROMPT
    acts_key: str = "act"

    def _from_wire(self, ctx: TurnContext) -> PerceptionResult | None:
        """Typed acts published by a cooperating partner, if any."""
        data = getattr(ctx.state, "current_data", None)
        if not isinstance(data, dict):
            return None
        for key, value in data.items():
            if not key.startswith(self.acts_key) or not isinstance(value, dict):
                continue
            acts = value.get("acts") or []
            if acts:
                return PerceptionResult(
                    acts=tuple(str(a) for a in acts),
                    source="wire",
                    text=str(data.get("text") or "") or None,
                )
        return None

    @staticmethod
    def _text_of(ctx: TurnContext) -> str:
        data = getattr(ctx.state, "current_data", None)
        if isinstance(data, dict) and data.get("text"):
            return str(data["text"])
        for entry in reversed(getattr(ctx.state, "new_data", None) or []):
            payload = entry[1] if isinstance(entry, tuple) else entry
            if isinstance(payload, dict) and payload.get("text"):
                return str(payload["text"])
        return ""

    def perceive(self, ctx: TurnContext) -> PerceptionResult:
        """Classify the partner's move, preferring typed data over inference."""
        wire = self._from_wire(ctx)
        if wire is not None:
            return wire
        text = self._text_of(ctx)
        if not text:
            return PerceptionResult(source="none")
        raw = self.call_llm(
            self.system_prompt,
            f'Their message: "{text}"\nTheir offer: {ctx.their_offer}',
        )
        data = self.parse_json(raw)
        acts = data.get("acts") or []
        commitments = data.get("commitments") or []
        return PerceptionResult(
            acts=tuple(str(a) for a in acts if a),
            commitments=tuple(
                {"text": str(c)} for c in commitments if c
            ),
            sentiment=str(data["sentiment"]) if data.get("sentiment") else None,
            source="classified",
            text=text,
        )


# =============================================================================
# Outcome validity — shared by every component that emits an outcome
# =============================================================================


def snap_outcome(outcome_space: Any, raw: Any, issue_index: dict[str, int] | None = None):
    """Coerce whatever a model returned into a valid outcome, or ``None``.

    Only structurally valid outcomes can ever be agreed on, so a near-miss
    should be repaired rather than discarded: numeric values snap to the
    nearest allowed value, categorical values match case-insensitively, and a
    dict keyed by issue name is reordered. Returns ``None`` when nothing
    salvageable remains.

    Args:
        outcome_space: The negotiation outcome space.
        raw: The model's proposed outcome — list, tuple or dict.
        issue_index: Optional precomputed issue-name -> position map.

    Returns:
        A valid outcome tuple, or ``None``.
    """
    issues = list(getattr(outcome_space, "issues", None) or [])
    if raw is None or not issues:
        return None
    if isinstance(raw, dict):
        index = issue_index or {str(i.name): n for n, i in enumerate(issues)}
        values: list[Any] = [None] * len(issues)
        for key, value in raw.items():
            pos = index.get(str(key))
            if pos is None:
                for name, p in index.items():
                    if name.lower() == str(key).lower():
                        pos = p
                        break
            if pos is not None:
                values[pos] = value
        raw = values
    if not isinstance(raw, (list, tuple)) or len(raw) != len(issues):
        return None
    snapped = tuple(
        _snap_value(issue, v) for issue, v in zip(issues, raw, strict=True)
    )
    if any(v is None for v in snapped):
        return None
    try:
        return snapped if outcome_space.is_valid(snapped) else None
    except Exception:  # noqa: BLE001
        return snapped


def _snap_value(issue: Any, value: Any) -> Any:
    """Coerce one value onto its issue's allowed values."""
    try:
        if issue.is_valid(value):
            return value
    except Exception:  # noqa: BLE001
        pass
    for cast in (int, float, str):
        try:
            candidate = cast(value)
            if issue.is_valid(candidate):
                return candidate
        except Exception:  # noqa: BLE001
            continue
    try:
        allowed = list(issue.all)
    except Exception:  # noqa: BLE001
        allowed = []
    if allowed:
        try:
            target = float(str(value))
            return min(allowed, key=lambda x: abs(float(str(x)) - target))
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            for candidate in allowed:
                if str(candidate).strip().lower() == text:
                    return candidate
            return allowed[0]
    try:
        return min(max(float(value), float(issue.min_value)), float(issue.max_value))
    except Exception:  # noqa: BLE001
        return None


def outcome_space_of(negotiator: Any):
    """The outcome space, from the NMI if joined or the ufun otherwise.

    The NMI is only available while attached to a mechanism, so a component
    that consults it alone silently sees nothing outside a live negotiation.
    """
    os_ = None
    if getattr(negotiator, "nmi", None) is not None:
        os_ = getattr(negotiator.nmi, "outcome_space", None)
    if os_ is None and getattr(negotiator, "ufun", None) is not None:
        os_ = getattr(negotiator.ufun, "outcome_space", None)
    return os_


def _describe_domain(negotiator: Any) -> str:
    """Issue names and allowed values, for a prompt."""
    os_ = outcome_space_of(negotiator)
    issues = list(getattr(os_, "issues", None) or [])
    if not issues:
        return ""
    lines = []
    for issue in issues:
        try:
            values = list(issue.all)[:12]
        except Exception:  # noqa: BLE001
            values = [f"{issue.min_value}..{issue.max_value}"]
        lines.append(f"  - {issue.name}: {values}")
    return "Issues, in order, with allowed values:\n" + "\n".join(lines)


def _history_lines(negotiator: Any, ctx: TurnContext, k: int = 6) -> str:
    """Recent offers with our utilities, for a prompt."""
    ufun = getattr(negotiator, "ufun", None)
    rows = []
    for prev in ctx.history[-k:]:
        if prev.entry == "propose" and prev.bid is not None:
            who, outcome = "you", _outcome_of(prev.bid)
        elif prev.their_offer is not None:
            who, outcome = "them", prev.their_offer
        else:
            continue
        try:
            u = f" (worth {float(ufun(outcome)):.2f} to you)" if ufun else ""
        except Exception:  # noqa: BLE001
            u = ""
        rows.append(f"  step {prev.step}: {who} offered {outcome}{u}")
    return "Recent offers:\n" + "\n".join(rows) if rows else ""


# =============================================================================
# B — Bidding
# =============================================================================


DEFAULT_OFFERING_PROMPT = """You choose the next offer in a negotiation.

Rules:
    1. "outcome" MUST be a JSON list with exactly one value per issue, in the
       issue order given, each value taken from that issue's allowed values.
       An invalid outcome cannot be accepted by anyone and wastes the round.
    2. Never offer something worth at or below your reserved value.
    3. Open near your best outcome and concede gradually as time runs out.

Respond with ONLY this JSON: {"outcome": [<one value per issue>], "why": "<brief>"}"""


@define(slots=False)
class LLMOffering(LLMComponent, OfferingPolicy):
    """``B`` — the LLM chooses the offer.

    Named for negmas' ``OfferingPolicy``, the slot it fills; the PABLO-ve letter
    is ``B`` because that is BOA's own name for the bidding strategy.

    Whatever the model returns is snapped onto the outcome space and checked
    for individual rationality, so this component cannot emit an offer that is
    invalid or worse than no deal. Both guards are counted in :attr:`stats`, so
    reliance on them is measurable rather than hidden — the rate at which a
    model needs rescuing is itself a result about that model.

    Attributes:
        system_prompt: Override the bidding instructions.
        enforce_rationality: Refuse outcomes at or below the reserved value.
        fallback: ``"aspiration"`` draws a time-appropriate outcome from the
            utility inverter when nothing usable survives; ``"best"`` repeats
            our best outcome; ``"none"`` returns ``None``.
    """

    system_prompt: str = DEFAULT_OFFERING_PROMPT
    enforce_rationality: bool = True
    fallback: str = "aspiration"
    concession_exponent: float = 0.3
    stats: dict[str, int] = field(factory=lambda: {"calls": 0, "invalid": 0, "fallback": 0})
    _inverter: Any = field(default=None, init=False)

    def _outcome_space(self):
        return outcome_space_of(self.negotiator)

    def _reserved(self) -> float:
        try:
            rv = self.negotiator.ufun.reserved_value
        except Exception:  # noqa: BLE001
            return 0.0
        return float(rv) if rv is not None and rv == rv else 0.0

    def _fallback_outcome(self, state):
        """A valid, individually rational offer when the model gave none."""
        self.stats["fallback"] += 1
        ufun = getattr(self.negotiator, "ufun", None)
        if ufun is None:
            return None
        if self.fallback == "none":
            return None
        if self.fallback == "best":
            try:
                return ufun.best()
            except Exception:  # noqa: BLE001
                return None
        try:
            if self._inverter is None:
                self._inverter = ufun.invert()
                self._inverter.init()
            t = float(getattr(state, "relative_time", 0.0) or 0.0)
            mn, mx = ufun.minmax()
            floor = (self._reserved() - mn) / (mx - mn) if mx > mn else 0.0
            target = floor + (1.0 - floor) * max(0.0, 1.0 - t) ** self.concession_exponent
            found = self._inverter.worst_in((target, 1.0), normalized=True)
            if found is not None:
                return found
        except Exception:  # noqa: BLE001
            pass
        try:
            return ufun.best()
        except Exception:  # noqa: BLE001
            return None

    def __call__(self, state, dest: str | None = None):
        """Ask the model for an offer, then make sure it is one."""
        ctx = getattr(self.negotiator, "turn", None)
        self.stats["calls"] += 1
        limit = word_limit_instruction(self.word_budget)
        system = f"{self.system_prompt}\n{limit}" if limit else self.system_prompt
        parts = [
            _describe_domain(self.negotiator),
            f"Round {getattr(state, 'step', 0)}, time "
            f"{float(getattr(state, 'relative_time', 0.0) or 0.0):.0%}.",
        ]
        if ctx is not None:
            parts.append(_history_lines(self.negotiator, ctx))
            perception = ctx.perception_this_step()
            if perception is not None and perception.acts:
                parts.append(f"Their last move was: {', '.join(perception.acts)}.")
        raw = self.call_llm(system, "\n".join(p for p in parts if p))
        proposed = self.parse_json(raw).get("outcome")
        outcome = snap_outcome(self._outcome_space(), proposed)
        if proposed is not None and outcome is None:
            self.stats["invalid"] += 1
        if outcome is not None and self.enforce_rationality:
            try:
                if float(self.negotiator.ufun(outcome)) <= self._reserved():
                    outcome = None
            except Exception:  # noqa: BLE001
                pass
        return outcome if outcome is not None else self._fallback_outcome(state)


# =============================================================================
# A — Acceptance
# =============================================================================


DEFAULT_ACCEPTANCE_PROMPT = """You decide whether to accept an offer.

Accept only when the offer is good enough given how much time is left and what
you could still get by continuing. Never accept anything worth at or below your
reserved value — no deal is better than a bad deal.

Respond with ONLY this JSON: {"decision": "accept" | "reject", "why": "<brief>"}"""


@define(slots=False)
class LLMAcceptance(LLMComponent, AcceptancePolicy):
    """``A`` — the LLM decides whether to accept.

    A rationality guard overrides an ``accept`` of an offer at or below the
    reserved value: that is not a judgement call, it is worse than walking away.

    Attributes:
        system_prompt: Override the acceptance instructions.
        enforce_rationality: Veto irrational accepts (default ``True``).
    """

    system_prompt: str = DEFAULT_ACCEPTANCE_PROMPT
    enforce_rationality: bool = True
    stats: dict[str, int] = field(factory=lambda: {"calls": 0, "vetoed": 0})

    def __call__(self, state, offer, source: str | None = None):
        """Accept or reject the standing offer."""
        from negmas.sao import ResponseType

        if offer is None:
            return ResponseType.REJECT_OFFER
        ufun = getattr(self.negotiator, "ufun", None)
        try:
            utility = float(ufun(offer)) if ufun else None
            reserved = float(ufun.reserved_value) if ufun else 0.0
        except Exception:  # noqa: BLE001
            utility, reserved = None, 0.0

        self.stats["calls"] += 1
        ctx = getattr(self.negotiator, "turn", None)
        limit = word_limit_instruction(self.word_budget)
        system = f"{self.system_prompt}\n{limit}" if limit else self.system_prompt
        parts = [
            f"Round {getattr(state, 'step', 0)}, time "
            f"{float(getattr(state, 'relative_time', 0.0) or 0.0):.0%}.",
            f"Their offer: {offer}"
            + (f" (worth {utility:.2f} to you; your reserved value is {reserved:.2f})"
               if utility is not None else ""),
        ]
        if ctx is not None:
            parts.append(_history_lines(self.negotiator, ctx))
        decision = str(
            self.parse_json(self.call_llm(system, "\n".join(p for p in parts if p)))
            .get("decision", "reject")
        ).lower()

        if decision.startswith("accept"):
            if (
                self.enforce_rationality
                and utility is not None
                and utility <= reserved
            ):
                self.stats["vetoed"] += 1
                return ResponseType.REJECT_OFFER
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


# =============================================================================
# O — Opponent model
# =============================================================================


DEFAULT_UFUN_MODEL_PROMPT = """You infer what a negotiation partner values.

From their offers (and anything they said), estimate how much each issue matters
to them and which values they prefer. Weights must be non-negative and sum to 1.

Respond with ONLY this JSON:
{
    "weights": {"<issue>": <0..1>, ...},
    "values":  {"<issue>": {"<value>": <0..1>, ...}, ...}
}"""


@define(slots=False)
class LLMUFunModel(LLMComponent, UFunModel):
    """``O`` — an opponent utility model inferred by an LLM from their behaviour.

    This is the "text-conditioned opponent model" that several HAN-2026 entries
    arrived at independently: unlike a frequency model it can use *what the
    partner said*, not only what they offered.

    Re-estimates every ``refresh_every`` observed offers rather than every turn,
    since beliefs move slowly and each estimate costs a call. Until the first
    estimate arrives, :meth:`eval` returns 0 for every outcome, which is the
    honest representation of "no belief yet".

    Attributes:
        system_prompt: Override the inference instructions.
        refresh_every: Re-estimate after this many new partner offers.
    """

    system_prompt: str = DEFAULT_UFUN_MODEL_PROMPT
    refresh_every: int = 3
    weights: dict[str, float] = field(factory=dict)
    values: dict[str, dict[str, float]] = field(factory=dict)
    _seen: list[Any] = field(factory=list, init=False)
    _last_estimate_at: int = field(default=-1, init=False)

    def eval(self, offer):  # noqa: D102 - BaseUtilityFunction contract
        if offer is None or not self.weights:
            return 0.0
        issues = list(
            getattr(outcome_space_of(self.negotiator), "issues", None) or []
        )
        if not issues:
            return 0.0
        total = 0.0
        for issue, value in zip(issues, offer, strict=False):
            name = str(issue.name)
            weight = float(self.weights.get(name, 0.0))
            per_value = self.values.get(name, {})
            total += weight * float(per_value.get(str(value), 0.0))
        return total

    def before_responding(self, state, offer, source: str | None = None):
        """Record the partner's offer and re-estimate when enough have arrived."""
        if offer is None:
            return
        self._seen.append(offer)
        if len(self._seen) - self._last_estimate_at < self.refresh_every:
            return
        self._last_estimate_at = len(self._seen)
        self._estimate()

    def _estimate(self) -> None:
        """One LLM call to refresh the belief; failures leave it unchanged."""
        ctx = getattr(self.negotiator, "turn", None)
        parts = [_describe_domain(self.negotiator)]
        parts.append(
            "Their offers so far, most recent last:\n"
            + "\n".join(f"  {o}" for o in self._seen[-8:])
        )
        if ctx is not None:
            perception = ctx.perception_this_step()
            if perception is not None and perception.text:
                parts.append(f'They most recently said: "{perception.text}"')
        data = self.parse_json(
            self.call_llm(self.system_prompt, "\n".join(p for p in parts if p))
        )
        weights = data.get("weights")
        values = data.get("values")
        if isinstance(weights, dict) and weights:
            try:
                total = sum(abs(float(v)) for v in weights.values()) or 1.0
                self.weights = {
                    str(k): abs(float(v)) / total for k, v in weights.items()
                }
            except (TypeError, ValueError):
                pass
        if isinstance(values, dict):
            cleaned: dict[str, dict[str, float]] = {}
            for issue, mapping in values.items():
                if isinstance(mapping, dict):
                    try:
                        cleaned[str(issue)] = {
                            str(k): float(v) for k, v in mapping.items()
                        }
                    except (TypeError, ValueError):
                        continue
            if cleaned:
                self.values = cleaned


# =============================================================================
# v — Validation
# =============================================================================


DEFAULT_VALIDATION_PROMPT = """You check that a negotiation message is TRUE of
the decision behind it.

A message is inconsistent if it announces terms that are not in the offer,
claims a concession that did not happen, promises something the offer does not
contain, or states an action other than the one being taken.

Respond with ONLY this JSON:
{
    "consistent": true | false,
    "issues": ["<short description>", ...],
    "rewritten": "<a corrected message, or empty if already consistent>"
}"""


@define(slots=False)
class LLMValidation(LLMComponent, Validation):
    """``v`` — an LLM checks the utterance against the decision.

    The generic honesty check: the words must be true of the offer and the
    action. It may replace the utterance and may not touch the decision.

    Attributes:
        system_prompt: Override the checking instructions.
    """

    system_prompt: str = DEFAULT_VALIDATION_PROMPT
    stats: dict[str, int] = field(factory=lambda: {"checked": 0, "inconsistent": 0})

    def validate(self, ctx: TurnContext) -> ValidationResult:
        """Check this turn's utterance for consistency with the decision."""
        from negmas.sao import ResponseType

        utterance = ctx.utterance
        if utterance is None or not utterance.text.strip():
            return ValidationResult(ok=True)
        self.stats["checked"] += 1
        if ctx.entry == "propose":
            action = f"proposing {_outcome_of(ctx.bid)}"
        elif ctx.acceptance == ResponseType.ACCEPT_OFFER:
            action = f"accepting {ctx.their_offer}"
        elif ctx.acceptance == ResponseType.END_NEGOTIATION:
            action = "ending the negotiation"
        else:
            action = f"rejecting {ctx.their_offer}"
        data = self.parse_json(
            self.call_llm(
                self.system_prompt,
                f'The action being taken: {action}.\nThe message: "{utterance.text}"',
            )
        )
        if data.get("consistent", True):
            return ValidationResult(ok=True)
        self.stats["inconsistent"] += 1
        rewritten = str(data.get("rewritten") or "").strip()
        return ValidationResult(
            ok=False,
            issues=tuple(str(i) for i in (data.get("issues") or ())) or ("inconsistent",),
            revised=Utterance(text=rewritten, data=utterance.data) if rewritten else None,
        )


# =============================================================================
# e — Ending
# =============================================================================


DEFAULT_ENDING_PROMPT = """You decide whether to walk away from a negotiation.

Walk away only when continuing is worse than no deal at all — the partner will
clearly not offer anything above your reserved value before time runs out.
Ending early forfeits every remaining chance, so the bar is high.

Respond with ONLY this JSON: {"end": true | false, "why": "<brief>"}"""


@define(slots=False)
class LLMEnding(LLMComponent, Ending):
    """``e`` — an LLM decides whether to terminate.

    Guarded on both sides: it is never consulted before ``min_time`` (walking
    away in the opening rounds is almost never right), and an ``end`` decision
    is overridden when the standing offer is already better than no deal.

    Attributes:
        system_prompt: Override the instructions.
        min_time: Relative time before which ending is not even considered.
    """

    system_prompt: str = DEFAULT_ENDING_PROMPT
    min_time: float = 0.5
    stats: dict[str, int] = field(factory=lambda: {"asked": 0, "ended": 0, "vetoed": 0})

    def should_end(self, ctx: TurnContext) -> EndingDecision:
        """Whether to walk away now."""
        if ctx.relative_time < self.min_time:
            return EndingDecision(end=False, reason="too early to consider ending")
        ufun = getattr(self.negotiator, "ufun", None)
        utility = reserved = None
        if ufun is not None and ctx.their_offer is not None:
            try:
                utility = float(ufun(ctx.their_offer))
                reserved = float(ufun.reserved_value)
            except Exception:  # noqa: BLE001
                utility = reserved = None
        self.stats["asked"] += 1
        data = self.parse_json(
            self.call_llm(
                self.system_prompt,
                f"Time is {ctx.relative_time:.0%}.\n"
                f"Their offer: {ctx.their_offer}"
                + (
                    f" (worth {utility:.2f}; your reserved value is {reserved:.2f})"
                    if utility is not None and reserved is not None
                    else ""
                )
                + f"\n{_history_lines(self.negotiator, ctx)}",
            )
        )
        if not data.get("end"):
            return EndingDecision(end=False, reason=str(data.get("why", "")))
        if utility is not None and reserved is not None and utility > reserved:
            self.stats["vetoed"] += 1
            return EndingDecision(
                end=False, reason="vetoed: the standing offer beats no deal"
            )
        self.stats["ended"] += 1
        return EndingDecision(end=True, reason=str(data.get("why", "")))

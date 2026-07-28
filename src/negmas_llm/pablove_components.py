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
    Language,
    Perception,
    PerceptionResult,
    TurnContext,
    Utterance,
)

__all__ = [
    "LLMComponent",
    "TemplateLanguage",
    "LLMLanguage",
    "LLMPerception",
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
                f"{time.perf_counter() - start:.1f}s]\n  << {user[-300:]}\n  >> {text[:300]}"
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

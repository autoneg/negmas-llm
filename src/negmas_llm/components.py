"""LLM-based components for negmas modular negotiators.

This module provides LLM-based acceptance policies, offering policies,
and other components that can be used with MAPNegotiator and other
modular negotiators in negmas.
"""

from __future__ import annotations

import json
import re
import textwrap
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, cast

import litellm
from attrs import define, field
from litellm import ModelResponse
from negmas.gb import GBState, ResponseType
from negmas.gb.components import AcceptancePolicy, GBComponent, OfferingPolicy
from negmas.inout import serialize
from negmas.outcomes import Outcome

from negmas_llm.common import apply_max_tokens
from negmas_llm.tags import process_prompt

if TYPE_CHECKING:
    from litellm.types.utils import Choices
    from negmas.negotiators import Negotiator


def _dedent(text: str) -> str:
    """Dedent a multi-line string, stripping the first line if empty."""
    if text.startswith("\n"):
        text = text[1:]
    return textwrap.dedent(text)


# =============================================================================
# Default Prompts for Components
# =============================================================================

DEFAULT_ACCEPTANCE_SYSTEM_PROMPT = _dedent("""
    You are an acceptance policy. Decide ACCEPT / REJECT / END for offers.

    {{outcome-space}}
    {{utility-function}}
    """)

DEFAULT_ACCEPTANCE_RESPONSE_INSTRUCTIONS = _dedent("""
    Respond with JSON only, in this shape.
    {
        "decision": "accept" | "reject" | "end",
        "reasoning": "brief explanation"
    }

    The decision field takes one of these values.
        1. "accept" means accept the current offer.
        2. "reject" means reject the offer; a counter-offer is generated separately.
        3. "end" means end the negotiation without agreement.
    """)

DEFAULT_OFFERING_SYSTEM_PROMPT = _dedent("""
    You are an offering policy. Generate strategic offers that advance your
    interests while seeking acceptable agreements.

    {{outcome-space}}
    {{utility-function}}
    """)

DEFAULT_OFFERING_RESPONSE_INSTRUCTIONS = _dedent("""
    Respond with JSON only, in this shape.
    {
        "outcome": [value1, value2, ...],
        "text": "optional message to opponent",
        "reasoning": "brief explanation"
    }

    The outcome field is a list of values in issue order.
    """)

DEFAULT_SUPPORTER_SYSTEM_PROMPT = _dedent("""
    You generate concise, professional text for negotiation actions.
    Be brief, support the action, maintain a professional tone.
    Respond with ONLY the text message (no JSON, no formatting).
    """)

DEFAULT_VALIDATOR_PROMPT = _dedent("""
    You check consistency between negotiation text and actions.

    Respond with JSON only, in this shape.
    {
        "consistent": true | false,
        "issues": ["..."],
        "suggested_text": "corrected text if inconsistent",
        "suggested_action": "accept" | "reject" | "end"
    }
    """)


# =============================================================================
# Base mixin for LLM functionality
# =============================================================================


class LLMComponentMixin(ABC):
    """Mixin providing common LLM functionality for components.

    This mixin provides the core LLM interaction logic including:
    - LLM configuration (provider, model, API settings)
    - Message building and LLM calling
    - Response parsing

    Note:
        This is a mixin class that provides methods but no attrs fields.
        Fields must be defined on the concrete component classes.

    Expected attributes on subclasses:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        _conversation_history: List to store conversation history.
    """

    # Type hints for expected attributes (defined on subclasses)
    provider: str
    model: str
    api_key: str | None
    api_base: str | None
    temperature: float
    max_tokens: int
    llm_kwargs: dict[str, Any]
    _conversation_history: list[dict[str, str]]

    def get_model_string(self) -> str:
        """Get the model string for litellm."""
        return f"{self.provider}/{self.model}"

    def _process_prompt(
        self,
        prompt: str,
        negotiator: Negotiator | None,
        state: GBState | None = None,
    ) -> str:
        """Process a prompt, replacing all tags with their values.

        Args:
            prompt: The prompt string containing tags.
            negotiator: The negotiator instance.
            state: The current negotiation state.

        Returns:
            The processed prompt with tags replaced.
        """
        if negotiator is None:
            return prompt
        # Convert GBState to SAOState-like for process_prompt
        return process_prompt(prompt, negotiator, state)  # type: ignore[arg-type]

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        negotiator: Negotiator | None = None,
        state: GBState | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call the LLM and get a response.

        All message contents are processed with process_prompt before sending.

        Args:
            messages: The conversation messages.
            negotiator: The negotiator instance (for tag processing).
            state: The current negotiation state (for tag processing).
            max_tokens: Per-call override for the output token cap. If None,
                uses ``self.max_tokens``. A provider-specific alias in
                ``self.llm_kwargs`` always takes precedence.

        Returns:
            The LLM response text.
        """
        # Process all message contents with process_prompt
        processed_messages = []
        for msg in messages:
            processed_content = self._process_prompt(msg["content"], negotiator, state)
            processed_messages.append({**msg, "content": processed_content})

        kwargs: dict[str, Any] = {
            "model": self.get_model_string(),
            "messages": processed_messages,
            "temperature": self.temperature,
            **self.llm_kwargs,
        }
        apply_max_tokens(
            kwargs,
            self.provider,
            self.model,
            max_tokens if max_tokens is not None else self.max_tokens,
        )

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = litellm.completion(**kwargs)
        model_response = cast(ModelResponse, response)
        choices = cast(list["Choices"], model_response.choices)
        return choices[0].message.content or ""

    def _format_outcome(self, outcome: Outcome, negotiator: Negotiator | None) -> str:
        """Format an outcome for display."""
        if negotiator is not None and negotiator.nmi is not None:
            outcome_space = negotiator.nmi.outcome_space
            if outcome_space is not None:
                try:
                    issues = outcome_space.issues  # type: ignore[attr-defined]
                    if issues:
                        parts = []
                        for i, value in enumerate(outcome):
                            if i < len(issues):
                                parts.append(f"{issues[i].name}={value}")
                            else:
                                parts.append(str(value))
                        return "{" + ", ".join(parts) + "}"
                except AttributeError:
                    pass
        return str(outcome)

    def format_outcome_space(self, negotiator: Negotiator | None) -> str:
        """Format the outcome space for the LLM."""
        if negotiator is None or negotiator.nmi is None:
            return ""
        outcome_space = negotiator.nmi.outcome_space
        if outcome_space is None:
            return ""

        try:
            os_dict = serialize(outcome_space)
            os_dict.pop("__python_class__", None)

            parts = [
                "The Outcome Space follows.",
                "",
                f"```json\n{json.dumps(os_dict, indent=2, default=str)}\n```",
                "",
                "Each outcome is a tuple of values, one per issue.",
            ]
            return "\n".join(parts)
        except Exception:
            return f"The Outcome Space follows.\n\n{outcome_space}\n"

    def format_own_ufun(self, negotiator: Negotiator | None) -> str:
        """Format the utility function for the LLM."""
        if negotiator is None or negotiator.ufun is None:
            return (
                "Your Utility Function is not provided. You do NOT have a "
                "utility function. Negotiate using general principles and "
                "any instructions provided.\n"
            )

        try:
            ufun_dict = serialize(negotiator.ufun)
            ufun_dict.pop("__python_class__", None)
            reserved = negotiator.reserved_value
            ufun_str = str(negotiator.ufun)

            parts = [
                "Your Utility Function follows.",
                "",
                "Higher utility is better for you.",
                "",
                f"Your utility function is {ufun_str}.",
                f"Your reserved value (utility of no agreement) is {reserved}.",
                "",
                "Full specification follows.",
                f"```json\n{json.dumps(ufun_dict, indent=2, default=str)}\n```",
            ]
            return "\n".join(parts)
        except Exception:
            return (
                "Your Utility Function follows.\n\n"
                f"Your utility function is {negotiator.ufun}.\n"
                f"Your reserved value is {negotiator.reserved_value}.\n"
            )

    def format_state(
        self,
        state: GBState,
        offer: Outcome | None,
        negotiator: Negotiator | None,
    ) -> str:
        """Format the negotiation state for the LLM."""
        parts = ["Current state follows.", ""]
        parts.append(f"    Step is {state.step}.")
        parts.append(f"    Relative time is {state.relative_time:.2%}.")

        if offer is not None:
            offer_str = self._format_outcome(offer, negotiator)
            parts.append(f"    Current offer is {offer_str}.")

            if negotiator is not None and negotiator.ufun is not None:
                utility = negotiator.ufun(offer)
                parts.append(
                    f"    Your utility for the current offer is {utility:.4f}."
                )
        else:
            parts.append("    No current offer is on the table.")

        parts.append(f"    Running is {state.running}.")
        if state.broken:
            parts.append("    Status is BROKEN.")
        if state.timedout:
            parts.append("    Status is TIMED OUT.")

        parts.append("")
        return "\n".join(parts)

    def on_negotiation_start(self, state: GBState) -> None:
        """Reset conversation history when negotiation starts."""
        self._conversation_history = []


# =============================================================================
# LLM Acceptance Policy
# =============================================================================


@define
class LLMAcceptancePolicy(AcceptancePolicy, LLMComponentMixin):
    """An acceptance policy that uses an LLM to decide whether to accept offers.

    This component can be used with MAPNegotiator to provide LLM-based
    acceptance decisions while using a different offering policy.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        system_prompt: Custom system prompt. Supports tags like {{outcome-space}}.
        response_instructions: Custom response format instructions.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.

    Example:
        >>> from negmas.gb.negotiators.modular import MAPNegotiator
        >>> from negmas.gb.components.offering import RandomOfferingPolicy
        >>> from negmas_llm import LLMAcceptancePolicy
        >>>
        >>> negotiator = MAPNegotiator(
        ...     acceptance=LLMAcceptancePolicy(
        ...         provider="openai",
        ...         model="gpt-4o",
        ...     ),
        ...     offering=RandomOfferingPolicy(),
        ... )
    """

    # LLM configuration fields
    provider: str = field()
    model: str = field()
    api_key: str | None = field(default=None)
    api_base: str | None = field(default=None)
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1024)
    raise_on_parsing_error: bool = field(default=False)
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Configurable prompts
    system_prompt: str = field(default=DEFAULT_ACCEPTANCE_SYSTEM_PROMPT)
    response_instructions: str = field(default=DEFAULT_ACCEPTANCE_RESPONSE_INSTRUCTIONS)

    def format_response_instructions(self) -> str:
        """Format the response instructions for acceptance decisions."""
        return self.response_instructions

    def build_system_prompt(self, state: GBState) -> str:
        """Build the system prompt for acceptance decisions.

        Override this method for complete control over the system prompt.

        Args:
            state: The current negotiation state.

        Returns:
            The system prompt string.
        """
        # Combine system prompt with response instructions
        return f"{self.system_prompt}\n\n{self.response_instructions}"

    def build_user_message(
        self,
        state: GBState,
        offer: Outcome | None,
        source: str | None,
    ) -> str:
        """Build the user message for acceptance decisions.

        Override this method to customize how offers are presented to the LLM.

        Args:
            state: The current negotiation state.
            offer: The offer to evaluate.
            source: The ID of the negotiator who made the offer.

        Returns:
            The user message string.
        """
        parts = [f"Round {state.step}.", ""]
        parts.append(self.format_state(state, offer, self.negotiator))

        if offer is not None:
            parts.append(f"Offer from {source or 'opponent'}. Accept, reject, or end?")
        else:
            parts.append("No offer received. Continue (reject) or end?")

        parts.append("")
        parts.append("Respond with JSON.")
        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> ResponseType:
        """Parse the LLM response into a ResponseType."""
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return ResponseType.REJECT_OFFER

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ResponseType.REJECT_OFFER

        decision = data.get("decision", "reject").lower()
        decision_map = {
            "accept": ResponseType.ACCEPT_OFFER,
            "reject": ResponseType.REJECT_OFFER,
            "end": ResponseType.END_NEGOTIATION,
        }
        return decision_map.get(decision, ResponseType.REJECT_OFFER)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Evaluate an offer and return the acceptance decision.

        Args:
            state: The current negotiation state.
            offer: The offer to evaluate.
            source: The ID of the negotiator who made the offer.

        Returns:
            ResponseType indicating accept, reject, or end.
        """
        system_prompt = self.build_system_prompt(state)
        user_message = self.build_user_message(state, offer, source)

        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages, self.negotiator, state)

        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        return self._parse_response(response_text)


# =============================================================================
# LLM Offering Policy
# =============================================================================


@define
class LLMOfferingPolicy(OfferingPolicy, LLMComponentMixin):
    """An offering policy that uses an LLM to generate offers.

    This component can be used with MAPNegotiator to provide LLM-based
    offer generation while using a different acceptance policy.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        system_prompt: Custom system prompt. Supports tags like {{outcome-space}}.
        response_instructions: Custom response format instructions.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.

    Example:
        >>> from negmas.gb.negotiators.modular import MAPNegotiator
        >>> from negmas.gb.components.acceptance import AcceptAnyRational
        >>> from negmas_llm import LLMOfferingPolicy
        >>>
        >>> negotiator = MAPNegotiator(
        ...     acceptance=AcceptAnyRational(),
        ...     offering=LLMOfferingPolicy(
        ...         provider="openai",
        ...         model="gpt-4o",
        ...     ),
        ... )
    """

    # LLM configuration fields
    provider: str = field()
    model: str = field()
    api_key: str | None = field(default=None)
    api_base: str | None = field(default=None)
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1024)
    raise_on_parsing_error: bool = field(default=False)
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Configurable prompts
    system_prompt: str = field(default=DEFAULT_OFFERING_SYSTEM_PROMPT)
    response_instructions: str = field(default=DEFAULT_OFFERING_RESPONSE_INSTRUCTIONS)

    def format_response_instructions(self) -> str:
        """Format the response instructions for offer generation."""
        return self.response_instructions

    def build_system_prompt(self, state: GBState) -> str:
        """Build the system prompt for offer generation.

        Override this method for complete control over the system prompt.

        Args:
            state: The current negotiation state.

        Returns:
            The system prompt string.
        """
        # Combine system prompt with response instructions
        return f"{self.system_prompt}\n\n{self.response_instructions}"

    def build_user_message(self, state: GBState, dest: str | None) -> str:
        """Build the user message for offer generation.

        Override this method to customize how the state is presented.

        Args:
            state: The current negotiation state.
            dest: The destination negotiator ID.

        Returns:
            The user message string.
        """
        parts = [f"Round {state.step}.", ""]
        parts.append(self.format_state(state, None, self.negotiator))

        if state.step == 0:
            parts.append("Opening round. Make your first offer.")
        else:
            parts.append("Generate your next offer.")

        if dest:
            parts.append(f"Recipient: {dest}")

        parts.append("")
        parts.append("Respond with JSON.")
        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> tuple[Outcome | None, str | None]:
        """Parse the LLM response into an outcome and optional text."""
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return None, None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None, None

        outcome: Outcome | None = None
        outcome_data = data.get("outcome")
        if outcome_data is not None:
            if isinstance(outcome_data, list):
                outcome = tuple(outcome_data)
            elif isinstance(outcome_data, dict):
                # LLM returned dict - convert to tuple in correct issue order
                outcome = self._dict_to_outcome_tuple(outcome_data)

        # Validate the outcome
        if outcome is not None:
            outcome = self._validate_outcome(outcome)

        text = data.get("text")
        return outcome, text

    def _validate_outcome(self, outcome: Outcome) -> Outcome | None:
        """Validate an outcome to ensure it has no None values and is valid.

        Args:
            outcome: The outcome to validate.

        Returns:
            The outcome if valid, or None if invalid.

        Raises:
            ValueError: If raise_on_parsing_error is True and the outcome is invalid.
        """
        if outcome is None:
            return None

        # Check for None values
        if any(v is None for v in outcome):
            msg = (
                f"LLM returned outcome with None values: {outcome}. "
                "Setting outcome to None (information-only message)."
            )
            if self.raise_on_parsing_error:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
            return None

        # Validate against outcome space if available
        if self.negotiator is not None:
            nmi = getattr(self.negotiator, "nmi", None)
            if nmi is not None and nmi.outcome_space is not None:
                try:
                    if not nmi.outcome_space.is_valid(outcome):  # type: ignore[attr-defined]
                        msg = (
                            f"LLM returned invalid outcome: {outcome}. "
                            "Not valid in outcome space. "
                            "Setting to None (information-only message)."
                        )
                        if self.raise_on_parsing_error:
                            raise ValueError(msg)
                        warnings.warn(msg, stacklevel=2)
                        return None
                except (AttributeError, TypeError):
                    pass

        return outcome

    def _dict_to_outcome_tuple(self, outcome_dict: dict[str, Any]) -> Outcome:
        """Convert a dict outcome to a tuple in the correct issue order.

        Args:
            outcome_dict: Dict mapping issue names to values.

        Returns:
            Outcome tuple with values in the correct issue order.
        """
        if self.negotiator is None:
            return tuple(outcome_dict.values())

        nmi = getattr(self.negotiator, "nmi", None)
        if nmi is None or nmi.outcome_space is None:
            return tuple(outcome_dict.values())

        try:
            issues = nmi.outcome_space.issues  # type: ignore[attr-defined]
            if not issues:
                return tuple(outcome_dict.values())

            values = []
            for issue in issues:
                issue_name = issue.name
                if issue_name in outcome_dict:
                    values.append(outcome_dict[issue_name])
                else:
                    # Try case-insensitive match
                    found = False
                    for key, val in outcome_dict.items():
                        if key.lower() == issue_name.lower():
                            values.append(val)
                            found = True
                            break
                    if not found:
                        # Can't match all issues, fall back to dict values order
                        return tuple(outcome_dict.values())

            if len(values) == len(issues):
                return tuple(values)
            return tuple(outcome_dict.values())
        except AttributeError:
            return tuple(outcome_dict.values())

    def __call__(self, state: GBState, dest: str | None = None) -> Outcome | None:
        """Generate an offer using the LLM.

        Args:
            state: The current negotiation state.
            dest: The destination negotiator ID.

        Returns:
            The proposed outcome or None.
        """
        system_prompt = self.build_system_prompt(state)
        user_message = self.build_user_message(state, dest)

        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages, self.negotiator, state)

        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        outcome, _ = self._parse_response(response_text)
        return outcome


# =============================================================================
# LLM Negotiation Supporter
# =============================================================================


@define
class LLMNegotiationSupporter(GBComponent, LLMComponentMixin):
    """A component that generates supporting text for negotiation actions.

    This component wraps another negotiator's decisions and uses an LLM
    to generate persuasive text to accompany offers and responses.
    It does not make decisions itself - it only adds text support.

    The supporter hooks into the after_proposing and after_responding
    callbacks to generate text after actions are taken.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.

    Example:
        >>> from negmas.gb.negotiators.modular import MAPNegotiator
        >>> from negmas.gb.components.acceptance import AcceptAnyRational
        >>> from negmas.gb.components.offering import TimeBasedOfferingPolicy
        >>> from negmas_llm import LLMNegotiationSupporter
        >>>
        >>> negotiator = MAPNegotiator(
        ...     acceptance=AcceptAnyRational(),
        ...     offering=TimeBasedOfferingPolicy(),
        ...     extra_components=[
        ...         LLMNegotiationSupporter(
        ...             provider="openai",
        ...             model="gpt-4o",
        ...         ),
        ...     ],
        ... )
    """

    # LLM configuration fields
    provider: str = field()
    model: str = field()
    api_key: str | None = field(default=None)
    api_base: str | None = field(default=None)
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1024)
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Configurable prompt
    system_prompt: str = field(default=DEFAULT_SUPPORTER_SYSTEM_PROMPT)

    # Store generated text for retrieval
    _last_generated_text: str | None = field(default=None, init=False)

    def build_system_prompt(self) -> str:
        """Build the system prompt for text generation."""
        return self.system_prompt

    def generate_offer_text(
        self,
        state: GBState,
        offer: Outcome | None,
        dest: str | None,
    ) -> str:
        """Generate text to accompany an offer.

        Override this method to customize offer text generation.

        Args:
            state: The current negotiation state.
            offer: The offer being made.
            dest: The destination negotiator ID.

        Returns:
            Generated text to accompany the offer.
        """
        if offer is None:
            return ""

        offer_str = self._format_outcome(offer, self.negotiator)
        utility = None
        if self.negotiator is not None and self.negotiator.ufun is not None:
            utility = self.negotiator.ufun(offer)

        user_message = (
            "Generate a brief, persuasive message for this offer.\n\n"
            f"The offer is {offer_str}.\n"
            f"The round is {state.step}.\n"
            f"Relative time is {state.relative_time:.1%}.\n"
        )
        if utility is not None:
            user_message += f"Your utility for this offer is {utility:.3f}.\n"

        if state.step == 0:
            user_message += "\nOpening offer."
        else:
            user_message += "\nCounter-offer."

        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages, self.negotiator, state)

    def generate_response_text(
        self,
        state: GBState,
        offer: Outcome | None,
        response: ResponseType,
        source: str | None,
    ) -> str:
        """Generate text to accompany a response.

        Override this method to customize response text generation.

        Args:
            state: The current negotiation state.
            offer: The offer being responded to.
            response: The response type (accept, reject, end).
            source: The source of the offer.

        Returns:
            Generated text to accompany the response.
        """
        response_names = {
            ResponseType.ACCEPT_OFFER: "ACCEPT",
            ResponseType.REJECT_OFFER: "REJECT",
            ResponseType.END_NEGOTIATION: "END NEGOTIATION",
        }
        response_name = response_names.get(response, "REJECT")

        offer_str = self._format_outcome(offer, self.negotiator) if offer else "None"

        user_message = (
            "Generate a brief message for this response.\n\n"
            f"The response is {response_name}.\n"
            f"The offer is {offer_str}.\n"
            f"The round is {state.step}.\n"
        )

        if response == ResponseType.ACCEPT_OFFER:
            user_message += "\nExpress agreement positively."
        elif response == ResponseType.REJECT_OFFER:
            user_message += "\nExplain the rejection constructively."
        else:
            user_message += "\nExplain why you're ending."

        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages, self.negotiator, state)

    def after_proposing(
        self, state: GBState, offer: Outcome | None, dest: str | None = None
    ) -> None:
        """Generate text after a proposal is made."""
        self._last_generated_text = self.generate_offer_text(state, offer, dest)

    def after_responding(
        self,
        state: GBState,
        offer: Outcome | None,
        response: ResponseType,
        source: str | None = None,
    ) -> None:
        """Generate text after a response is made."""
        self._last_generated_text = self.generate_response_text(
            state, offer, response, source
        )

    @property
    def last_text(self) -> str | None:
        """Get the last generated text."""
        return self._last_generated_text


# =============================================================================
# LLM Validator
# =============================================================================


@define
class LLMValidator(GBComponent, LLMComponentMixin):
    """A component that validates consistency between text and actions.

    This component uses an LLM to check if generated text matches the
    action being taken, and optionally modifies one to match the other.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        mode: How to handle mismatches:
            - "text_wins": Modify the action to match the text
            - "action_wins": Modify the text to match the action
            - "validate_only": Only report mismatches, don't modify
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        validation_prompt: Custom validation prompt.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.

    Example:
        >>> from negmas.gb.negotiators.modular import MAPNegotiator
        >>> from negmas_llm import LLMAcceptancePolicy, LLMOfferingPolicy, LLMValidator
        >>>
        >>> negotiator = MAPNegotiator(
        ...     acceptance=LLMAcceptancePolicy(provider="openai", model="gpt-4o"),
        ...     offering=LLMOfferingPolicy(provider="openai", model="gpt-4o"),
        ...     extra_components=[
        ...         LLMValidator(
        ...             provider="openai",
        ...             model="gpt-4o",
        ...             mode="action_wins",
        ...         ),
        ...     ],
        ... )
    """

    # LLM configuration fields
    provider: str = field()
    model: str = field()
    api_key: str | None = field(default=None)
    api_base: str | None = field(default=None)
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1024)
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Component-specific fields
    mode: Literal["text_wins", "action_wins", "validate_only"] = field(
        default="validate_only"
    )

    # Configurable prompt
    validation_prompt: str = field(default=DEFAULT_VALIDATOR_PROMPT)

    # Store validation results
    _last_validation_result: dict[str, Any] | None = field(default=None, init=False)

    def build_validation_prompt(self) -> str:
        """Build the system prompt for validation."""
        return self.validation_prompt

    def validate_response(
        self,
        text: str | None,
        response: ResponseType,
        offer: Outcome | None,
    ) -> dict[str, Any]:
        """Validate that text matches a response action.

        Args:
            text: The text accompanying the response.
            response: The response type.
            offer: The offer being responded to.

        Returns:
            Validation result dictionary.
        """
        if text is None:
            return {"consistent": True, "issues": []}

        response_names = {
            ResponseType.ACCEPT_OFFER: "ACCEPT",
            ResponseType.REJECT_OFFER: "REJECT",
            ResponseType.END_NEGOTIATION: "END NEGOTIATION",
        }
        response_name = response_names.get(response, "REJECT")
        offer_str = self._format_outcome(offer, self.negotiator) if offer else "None"

        user_message = (
            "Validate this negotiation response.\n\n"
            f"The action is {response_name}.\n"
            f"The offer is {offer_str}.\n"
            f'The text is: "{text}"\n\n'
            "Is the text consistent with the action?"
        )

        messages = [
            {"role": "system", "content": self.build_validation_prompt()},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages, self.negotiator)

        # Parse response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return {"consistent": True, "issues": []}

        try:
            result: dict[str, Any] = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            return {"consistent": True, "issues": []}

    def validate_offer(
        self,
        text: str | None,
        offer: Outcome | None,
    ) -> dict[str, Any]:
        """Validate that text matches an offer.

        Args:
            text: The text accompanying the offer.
            offer: The offer being made.

        Returns:
            Validation result dictionary.
        """
        if text is None or offer is None:
            return {"consistent": True, "issues": []}

        offer_str = self._format_outcome(offer, self.negotiator)

        user_message = (
            "Validate this negotiation offer.\n\n"
            f"The offer is {offer_str}.\n"
            f'The text is: "{text}"\n\n'
            "Is the text consistent with the offer?"
        )

        messages = [
            {"role": "system", "content": self.build_validation_prompt()},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages, self.negotiator)

        # Parse response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return {"consistent": True, "issues": []}

        try:
            result: dict[str, Any] = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            return {"consistent": True, "issues": []}

    def correct_text(self, text: str, action_description: str) -> str:
        """Generate corrected text that matches the action.

        Args:
            text: The original text.
            action_description: Description of the action.

        Returns:
            Corrected text.
        """
        user_message = (
            "Rewrite the text to match the action. "
            "Reply with ONLY the corrected text.\n\n"
            f'The original text is: "{text}"\n'
            f"The action is {action_description}."
        )

        messages = [
            {
                "role": "system",
                "content": "You correct negotiation text to match actions.",
            },
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages, self.negotiator)

    def after_responding(
        self,
        state: GBState,
        offer: Outcome | None,
        response: ResponseType,
        source: str | None = None,
    ) -> None:
        """Validate response after it's made."""
        # This is a hook for validation - in practice, validation would
        # need to be integrated more deeply into the negotiation flow
        # to actually modify actions/text
        pass

    @property
    def last_validation(self) -> dict[str, Any] | None:
        """Get the last validation result."""
        return self._last_validation_result


# =============================================================================
# Provider-specific convenience classes
# =============================================================================


@define
class OpenAIAcceptancePolicy(LLMAcceptancePolicy):
    """LLM Acceptance Policy using OpenAI models."""

    provider: str = field(default="openai", init=False)
    model: str = field(default="gpt-4o")


@define
class OpenAIOfferingPolicy(LLMOfferingPolicy):
    """LLM Offering Policy using OpenAI models."""

    provider: str = field(default="openai", init=False)
    model: str = field(default="gpt-4o")


@define
class OllamaAcceptancePolicy(LLMAcceptancePolicy):
    """LLM Acceptance Policy using Ollama for local inference."""

    provider: str = field(default="ollama", init=False)
    model: str = field(default="llama3.2")
    api_base: str | None = field(default="http://localhost:11434")


@define
class OllamaOfferingPolicy(LLMOfferingPolicy):
    """LLM Offering Policy using Ollama for local inference."""

    provider: str = field(default="ollama", init=False)
    model: str = field(default="llama3.2")
    api_base: str | None = field(default="http://localhost:11434")


@define
class AnthropicAcceptancePolicy(LLMAcceptancePolicy):
    """LLM Acceptance Policy using Anthropic Claude models."""

    provider: str = field(default="anthropic", init=False)
    model: str = field(default="claude-sonnet-4-20250514")


@define
class AnthropicOfferingPolicy(LLMOfferingPolicy):
    """LLM Offering Policy using Anthropic Claude models."""

    provider: str = field(default="anthropic", init=False)
    model: str = field(default="claude-sonnet-4-20250514")

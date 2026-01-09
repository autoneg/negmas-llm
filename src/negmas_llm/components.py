"""LLM-based components for negmas modular negotiators.

This module provides LLM-based acceptance policies, offering policies,
and other components that can be used with MAPNegotiator and other
modular negotiators in negmas.
"""

from __future__ import annotations

import json
import re
from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, cast

import litellm
from attrs import define, field
from litellm import ModelResponse
from negmas.gb import GBState, ResponseType
from negmas.gb.components import AcceptancePolicy, GBComponent, OfferingPolicy
from negmas.inout import serialize
from negmas.outcomes import Outcome

if TYPE_CHECKING:
    from litellm.types.utils import Choices
    from negmas.negotiators import Negotiator


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

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and get a response."""
        kwargs: dict[str, Any] = {
            "model": self.get_model_string(),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.llm_kwargs,
        }

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

            parts = ["## Outcome Space"]
            parts.append("")
            parts.append(
                "The negotiation outcome space defines the possible agreements:"
            )
            parts.append("")
            parts.append(f"```json\n{json.dumps(os_dict, indent=2, default=str)}\n```")
            parts.append("")
            parts.append(
                "Each outcome is a tuple of values, one for each issue/dimension."
            )
            return "\n".join(parts)
        except Exception:
            return f"## Outcome Space\n\n{outcome_space}\n"

    def format_own_ufun(self, negotiator: Negotiator | None) -> str:
        """Format the utility function for the LLM."""
        if negotiator is None or negotiator.ufun is None:
            return """## Your Utility Function

You do NOT have a utility function. You must negotiate based on general
principles and any instructions provided.
"""

        try:
            ufun_dict = serialize(negotiator.ufun)
            ufun_dict.pop("__python_class__", None)
            reserved = negotiator.reserved_value
            ufun_str = str(negotiator.ufun)

            parts = ["## Your Utility Function"]
            parts.append("")
            parts.append(
                "You have a utility function that evaluates outcomes. "
                "Higher utility values are better for you."
            )
            parts.append("")
            parts.append(f"**Description**: {ufun_str}")
            parts.append("")
            parts.append(f"**Reserved Value** (utility if no agreement): {reserved}")
            parts.append("")
            parts.append("**Full specification**:")
            parts.append(
                f"```json\n{json.dumps(ufun_dict, indent=2, default=str)}\n```"
            )
            return "\n".join(parts)
        except Exception:
            return f"""## Your Utility Function

You have a utility function: {negotiator.ufun}
Reserved value (utility if no agreement): {negotiator.reserved_value}
"""

    def format_state(
        self,
        state: GBState,
        offer: Outcome | None,
        negotiator: Negotiator | None,
    ) -> str:
        """Format the negotiation state for the LLM."""
        parts = ["## Current State"]
        parts.append("")
        parts.append(f"- **Step**: {state.step}")
        parts.append(f"- **Relative time**: {state.relative_time:.2%}")

        if offer is not None:
            offer_str = self._format_outcome(offer, negotiator)
            parts.append(f"- **Current offer on table**: {offer_str}")

            if negotiator is not None and negotiator.ufun is not None:
                utility = negotiator.ufun(offer)
                parts.append(f"- **Your utility for current offer**: {utility:.4f}")
        else:
            parts.append("- **Current offer on table**: None")

        parts.append(f"- **Negotiation running**: {state.running}")
        if state.broken:
            parts.append("- **Status**: BROKEN")
        if state.timedout:
            parts.append("- **Status**: TIMED OUT")

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
        system_prompt: Custom system prompt (overrides build_system_prompt).
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
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Component-specific fields
    _custom_system_prompt: str | None = field(default=None)

    def format_response_instructions(self) -> str:
        """Format the response instructions for acceptance decisions."""
        return """\
## Response Format

IMPORTANT: Your response MUST be valid JSON in the following format:
{
    "decision": "accept" | "reject" | "end",
    "reasoning": "brief explanation of your decision"
}

Where:
- "decision": Your decision:
  - "accept": Accept the current offer
  - "reject": Reject the offer (a counter-offer will be generated separately)
  - "end": End the negotiation without agreement
- "reasoning": Brief explanation of why you made this decision

Always respond with ONLY the JSON object, no additional text."""

    def build_system_prompt(self, state: GBState) -> str:
        """Build the system prompt for acceptance decisions.

        Override this method for complete control over the system prompt.

        Args:
            state: The current negotiation state.

        Returns:
            The system prompt string.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt

        outcome_space_info = self.format_outcome_space(self.negotiator)
        own_ufun_info = self.format_own_ufun(self.negotiator)
        response_instructions = self.format_response_instructions()

        return f"""\
You are an acceptance policy in an automated negotiation.
Your role is to decide whether to ACCEPT, REJECT, or END the negotiation
based on offers received.

{outcome_space_info}
{own_ufun_info}
{response_instructions}"""

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
        parts = [f"# Negotiation Round {state.step}"]
        parts.append("")
        parts.append(self.format_state(state, offer, self.negotiator))

        if offer is not None:
            parts.append(
                f"You received an offer from **{source or 'opponent'}**. "
                "Should you accept, reject, or end the negotiation?"
            )
        else:
            parts.append("No offer received. Should you continue (reject) or end?")

        parts.append("")
        parts.append("Respond with your decision in JSON format.")
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

        response_text = self._call_llm(messages)

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
        system_prompt: Custom system prompt (overrides build_system_prompt).
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
    llm_kwargs: dict[str, Any] = field(factory=dict)
    _conversation_history: list[dict[str, str]] = field(factory=list, init=False)

    # Component-specific fields
    _custom_system_prompt: str | None = field(default=None)

    def format_response_instructions(self) -> str:
        """Format the response instructions for offer generation."""
        return """\
## Response Format

IMPORTANT: Your response MUST be valid JSON in the following format:
{
    "outcome": [value1, value2, ...],
    "text": "optional message to accompany the offer",
    "reasoning": "brief explanation of why you chose this offer"
}

Where:
- "outcome": Your proposed offer as a list of values matching the issue order
- "text": Optional text message to send with the offer
- "reasoning": Brief explanation of your strategy

Always respond with ONLY the JSON object, no additional text."""

    def build_system_prompt(self, state: GBState) -> str:
        """Build the system prompt for offer generation.

        Override this method for complete control over the system prompt.

        Args:
            state: The current negotiation state.

        Returns:
            The system prompt string.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt

        outcome_space_info = self.format_outcome_space(self.negotiator)
        own_ufun_info = self.format_own_ufun(self.negotiator)
        response_instructions = self.format_response_instructions()

        return f"""\
You are an offering policy in an automated negotiation.
Your role is to generate strategic offers that advance your interests
while seeking mutually acceptable agreements.

{outcome_space_info}
{own_ufun_info}
{response_instructions}"""

    def build_user_message(self, state: GBState, dest: str | None) -> str:
        """Build the user message for offer generation.

        Override this method to customize how the state is presented.

        Args:
            state: The current negotiation state.
            dest: The destination negotiator ID.

        Returns:
            The user message string.
        """
        parts = [f"# Negotiation Round {state.step}"]
        parts.append("")
        parts.append(self.format_state(state, None, self.negotiator))

        if state.step == 0:
            parts.append("This is the opening round. Make your first offer.")
        else:
            parts.append("Generate your next offer based on the negotiation progress.")

        if dest:
            parts.append(f"Your offer will be sent to: {dest}")

        parts.append("")
        parts.append("Respond with your offer in JSON format.")
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
        if outcome_data is not None and isinstance(outcome_data, list):
            outcome = tuple(outcome_data)

        text = data.get("text")
        return outcome, text

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

        response_text = self._call_llm(messages)

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

    # Store generated text for retrieval
    _last_generated_text: str | None = field(default=None, init=False)

    def build_system_prompt(self) -> str:
        """Build the system prompt for text generation."""
        return """\
You are a skilled negotiation communicator. Your role is to generate
persuasive, professional text to accompany negotiation actions.

Your text should:
- Be concise but compelling
- Support the action being taken
- Maintain a professional tone
- Build rapport while advancing your position

Respond with ONLY the text message, no JSON or formatting."""

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

        user_message = f"""Generate a brief, persuasive message to accompany this offer:

Offer: {offer_str}
Round: {state.step}
Relative time: {state.relative_time:.1%}
"""
        if utility is not None:
            user_message += f"Your utility: {utility:.3f}\n"

        if state.step == 0:
            user_message += "\nThis is the opening offer."
        else:
            user_message += "\nThis is a counter-offer in an ongoing negotiation."

        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages)

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

        user_message = f"""Generate a brief message to accompany this response:

Response: {response_name}
Offer being responded to: {offer_str}
Round: {state.step}
"""

        if response == ResponseType.ACCEPT_OFFER:
            user_message += "\nExpress agreement and finalize positively."
        elif response == ResponseType.REJECT_OFFER:
            user_message += "\nExplain the rejection constructively."
        else:
            user_message += "\nExplain why you're ending the negotiation."

        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages)

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

    # Store validation results
    _last_validation_result: dict[str, Any] | None = field(default=None, init=False)

    def build_validation_prompt(self) -> str:
        """Build the system prompt for validation."""
        return """\
You are a negotiation consistency validator. Your role is to check if
text messages are consistent with negotiation actions.

Analyze whether the text accurately represents the action being taken.
Report any inconsistencies found.

Respond in JSON format:
{
    "consistent": true | false,
    "issues": ["list of inconsistencies if any"],
    "suggested_text": "corrected text if inconsistent (only if text needs correction)",
    "suggested_action": "accept" | "reject" | "end" (only if action needs correction)
}"""

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

        user_message = f"""Validate this negotiation response:

Action: {response_name}
Offer: {offer_str}
Text message: "{text}"

Is the text consistent with the action?"""

        messages = [
            {"role": "system", "content": self.build_validation_prompt()},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages)

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

        user_message = f"""Validate this negotiation offer:

Offer: {offer_str}
Text message: "{text}"

Is the text consistent with the offer being made?"""

        messages = [
            {"role": "system", "content": self.build_validation_prompt()},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages)

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
        user_message = f"""The following text is inconsistent with the action.
Generate corrected text that matches the action.

Original text: "{text}"
Action: {action_description}

Respond with ONLY the corrected text, no formatting."""

        messages = [
            {
                "role": "system",
                "content": "You correct negotiation text to match actions.",
            },
            {"role": "user", "content": user_message},
        ]

        return self._call_llm(messages)

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

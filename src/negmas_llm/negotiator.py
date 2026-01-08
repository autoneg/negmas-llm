"""LLM-based negotiators for the negmas framework."""

from __future__ import annotations

import json
import re
from abc import ABC
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm import ModelResponse
from negmas import Agent, Controller, Preferences
from negmas.gb import GBState
from negmas.inout import serialize
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState

if TYPE_CHECKING:
    from litellm.types.utils import Choices


class LLMNegotiator(SAONegotiator, ABC):
    """A negotiator that uses an LLM for decision-making.

    This negotiator delegates the negotiation strategy to a Large Language Model.
    It converts negotiation state to text, sends it to the LLM, and parses the
    response into negotiation actions.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        system_prompt: Custom system prompt (overrides build_system_prompt).
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        preferences: The preferences of the negotiator.
        ufun: The utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent Controller if any.
        owner: The Agent that owns the negotiator.
        id: Unique ID for the negotiator.
        type_name: Type name string.
        can_propose: Whether the negotiator can propose offers.
        **kwargs: Additional arguments passed to SAONegotiator.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        type_name: str | None = None,
        can_propose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            type_name=type_name,
            can_propose=can_propose,
            **kwargs,
        )
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._custom_system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs or {}

        # Conversation history for context
        self._conversation_history: list[dict[str, str]] = []

    def get_model_string(self) -> str:
        """Get the model string for litellm.

        Returns:
            The full model string in litellm format (provider/model).
        """
        return f"{self.provider}/{self.model}"

    def format_outcome_space(self, state: SAOState) -> str:
        """Format the outcome space for the LLM.

        Override this method to customize how the outcome space (negotiation
        issues and their possible values) is presented to the LLM.

        Args:
            state: The current negotiation state.

        Returns:
            A string describing the outcome space, or empty string if unavailable.
        """
        if self.nmi is None or self.nmi.outcome_space is None:
            return ""

        outcome_space = self.nmi.outcome_space
        try:
            # Use serialize() for structured representation
            os_dict = serialize(outcome_space)
            # Remove internal class identifier for cleaner output
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
            # Fallback to string representation
            return f"## Outcome Space\n\n{outcome_space}\n"

    def format_own_ufun(self, state: SAOState) -> str:
        """Format your own utility function for the LLM.

        Override this method to customize how your utility function is
        presented to the LLM.

        Args:
            state: The current negotiation state.

        Returns:
            A string describing your utility function, or a note if unavailable.
        """
        if self.ufun is None:
            return """## Your Utility Function

You do NOT have a utility function. You must negotiate based on general
principles and any instructions provided.
"""

        try:
            # Get structured representation
            ufun_dict = serialize(self.ufun)
            # Remove internal class identifier
            ufun_dict.pop("__python_class__", None)

            # Get reserved value (utility of no agreement)
            reserved = self.reserved_value

            # Also include string representation for readability
            ufun_str = str(self.ufun)

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
            # Fallback to string representation
            return f"""## Your Utility Function

You have a utility function: {self.ufun}
Reserved value (utility if no agreement): {self.reserved_value}
"""

    def format_partner_ufun(self, state: SAOState) -> str:
        """Format the partner's utility function for the LLM.

        Override this method to customize how the partner's utility function
        (if known) is presented to the LLM.

        Args:
            state: The current negotiation state.

        Returns:
            A string describing partner's utility function, or empty if unknown.
        """
        # Access partner's ufun from private_info (negmas convention)
        partner_ufun = (
            self.private_info.get("opponent_ufun") if self.private_info else None
        )

        if partner_ufun is None:
            return """## Partner's Utility Function

You do NOT know your partner's utility function. You must infer their
preferences from their offers and behavior during the negotiation.
"""

        try:
            # Get structured representation
            ufun_dict = serialize(partner_ufun)
            # Remove internal class identifier
            ufun_dict.pop("__python_class__", None)

            # Get reserved value if available
            reserved = getattr(partner_ufun, "reserved_value", None)

            # Also include string representation for readability
            ufun_str = str(partner_ufun)

            parts = ["## Partner's Utility Function (Known)"]
            parts.append("")
            parts.append(
                "You know your partner's utility function. Use this information "
                "strategically to find mutually beneficial outcomes."
            )
            parts.append("")
            parts.append(f"**Description**: {ufun_str}")
            if reserved is not None:
                parts.append("")
                parts.append(f"**Their Reserved Value**: {reserved}")
            parts.append("")
            parts.append("**Full specification**:")
            parts.append(
                f"```json\n{json.dumps(ufun_dict, indent=2, default=str)}\n```"
            )

            return "\n".join(parts)
        except Exception:
            # Fallback to string representation
            return f"""## Partner's Utility Function (Known)

Your partner's utility function: {partner_ufun}
"""

    def format_nmi_info(self) -> str:
        """Format the Negotiator Mechanism Interface (NMI) information.

        The NMI contains metadata about the negotiation mechanism including
        time limits, step limits, number of possible outcomes, etc.

        Override this method to customize how NMI info is presented to the LLM.

        Returns:
            A string describing the negotiation mechanism parameters.
        """
        if self.nmi is None:
            return ""

        parts = ["## Negotiation Mechanism Information"]
        parts.append("")

        # Number of negotiation steps
        n_steps = self.nmi.n_steps
        if n_steps is not None:
            parts.append(f"- **Maximum steps**: {n_steps}")
        else:
            parts.append("- **Maximum steps**: Unlimited")

        # Time limit
        time_limit = self.nmi.time_limit
        if time_limit is not None:
            parts.append(f"- **Time limit**: {time_limit:.2f} seconds")
        else:
            parts.append("- **Time limit**: Unlimited")

        # Number of possible outcomes
        n_outcomes = self.nmi.n_outcomes
        if n_outcomes is not None:
            parts.append(f"- **Total possible outcomes**: {n_outcomes}")

        # Number of negotiators
        n_negotiators = self.nmi.n_negotiators
        if n_negotiators is not None:
            parts.append(f"- **Number of negotiators**: {n_negotiators}")

        # Dynamic entry allowed
        if hasattr(self.nmi, "dynamic_entry") and self.nmi.dynamic_entry is not None:
            parts.append(f"- **Dynamic entry allowed**: {self.nmi.dynamic_entry}")

        # Annotation space info if available
        if hasattr(self.nmi, "annotation") and self.nmi.annotation:
            parts.append(f"- **Mechanism annotation**: {self.nmi.annotation}")

        parts.append("")
        return "\n".join(parts)

    def format_state(self, state: SAOState, offer: Outcome | None = None) -> str:
        """Format the complete SAOState for the LLM.

        This provides a comprehensive view of the current negotiation state
        including all relevant fields.

        Override this method to customize how state is presented to the LLM.

        Args:
            state: The current negotiation state.
            offer: The specific offer being responded to (may differ from
                current_offer).

        Returns:
            A string describing the complete state.
        """
        parts = ["## Current State"]
        parts.append("")

        # Core timing info
        parts.append(f"- **Step**: {state.step}")
        parts.append(f"- **Relative time**: {state.relative_time:.2%}")

        # Current offer info (use passed offer if provided, else state.current_offer)
        display_offer = offer if offer is not None else state.current_offer
        if display_offer is not None:
            offer_str = self._format_outcome(display_offer)
            parts.append(f"- **Current offer on table**: {offer_str}")
            if state.current_proposer:
                parts.append(f"- **Current proposer**: {state.current_proposer}")

            # Compute utilities if available
            if self.ufun is not None:
                utility = self.ufun(display_offer)
                parts.append(f"- **Your utility for current offer**: {utility:.4f}")

            partner_ufun = (
                self.private_info.get("opponent_ufun") if self.private_info else None
            )
            if partner_ufun is not None:
                try:
                    partner_utility = partner_ufun(display_offer)
                    parts.append(
                        f"- **Partner's utility for current offer**: "
                        f"{partner_utility:.4f}"
                    )
                except Exception:
                    pass
        else:
            parts.append("- **Current offer on table**: None")

        # Number of acceptances so far
        parts.append(f"- **Number of acceptances**: {state.n_acceptances}")

        # Negotiation status flags
        parts.append(f"- **Negotiation running**: {state.running}")
        if state.broken:
            parts.append("- **Status**: BROKEN (negotiation ended abnormally)")
        if state.timedout:
            parts.append("- **Status**: TIMED OUT")
        if state.agreement is not None:
            agreement_str = self._format_outcome(state.agreement)
            parts.append(f"- **Agreement reached**: {agreement_str}")

        # Offer history if available
        if hasattr(state, "new_offers") and state.new_offers:
            parts.append("")
            parts.append("**Recent offers this step**:")
            for proposer, prop_offer in state.new_offers:
                if prop_offer is not None:
                    offer_str = self._format_outcome(prop_offer)
                    parts.append(f"  - {proposer}: {offer_str}")
                else:
                    parts.append(f"  - {proposer}: None")

        parts.append("")
        return "\n".join(parts)

    def format_response_instructions(self) -> str:
        """Format the response format instructions for the LLM.

        Override this method to customize the expected response format.

        Returns:
            A string describing the expected JSON response format.
        """
        return """\
## Response Format

IMPORTANT: Your response MUST be valid JSON in the following format:
{
    "response_type": "accept" | "reject" | "end",
    "outcome": [value1, value2, ...] | null,
    "text": "optional explanation or message to other party",
    "data": {} | null
}

Where:
- "response_type": Your decision:
  - "accept": Accept the current offer
  - "reject": Reject and provide a counter-offer
  - "end": End the negotiation without agreement
- "outcome": If rejecting, provide your counter-offer as a list of values
  matching the issue order. If accepting or ending, this can be null.
- "text": Optional text message or explanation to the other party
- "data": Optional additional data as a dictionary

Always respond with ONLY the JSON object, no additional text."""

    def build_system_prompt(self, state: SAOState) -> str:
        """Build the system prompt for the LLM.

        This method combines the output of format_outcome_space(), format_nmi_info(),
        format_own_ufun(), format_partner_ufun(), and format_response_instructions()
        to create the full system prompt.

        Override this method for complete control, or override the individual
        format_* methods to customize specific sections.

        Args:
            state: The current negotiation state.

        Returns:
            The system prompt string.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt

        outcome_space_info = self.format_outcome_space(state)
        nmi_info = self.format_nmi_info()
        own_ufun_info = self.format_own_ufun(state)
        partner_ufun_info = self.format_partner_ufun(state)
        response_instructions = self.format_response_instructions()

        return f"""\
You are a skilled negotiator participating in an automated negotiation.
Your goal is to negotiate effectively to achieve good outcomes for yourself
while finding mutually acceptable agreements when possible.

{outcome_space_info}
{nmi_info}
{own_ufun_info}
{partner_ufun_info}
{response_instructions}"""

    def build_user_message(
        self,
        state: SAOState,
        offer: Outcome | None,
        source: str | None,
    ) -> str:
        """Build the user message describing the current negotiation state.

        This method uses format_state() to provide comprehensive state information
        to the LLM for each round.

        Override this method to customize how negotiation state is presented
        to the LLM.

        Args:
            state: The current negotiation state.
            offer: The received offer (if any).
            source: The ID of the negotiator who made the offer.

        Returns:
            The user message string.
        """
        parts = [f"# Negotiation Round {state.step}"]
        parts.append("")

        # Include complete state information
        state_info = self.format_state(state, offer)
        parts.append(state_info)

        # Add context about received offer or first proposal
        if offer is not None:
            parts.append(
                f"You received an offer from **{source or 'opponent'}**. "
                "Please analyze it and respond."
            )
        else:
            parts.append("No offer received yet. Please make the first proposal.")

        parts.append("")
        parts.append("Respond with your decision in JSON format.")

        return "\n".join(parts)

    def _format_outcome(self, outcome: Outcome) -> str:
        """Format an outcome for display.

        Args:
            outcome: The outcome to format.

        Returns:
            A formatted string representation.
        """
        if self.nmi is not None and self.nmi.outcome_space is not None:
            outcome_space = self.nmi.outcome_space
            # Try to get issue names from outcome_space
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

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and get a response.

        Args:
            messages: The conversation messages.

        Returns:
            The LLM response text.
        """
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
        # Cast to ModelResponse as litellm.completion returns
        # Union[ModelResponse, CustomStreamWrapper] when stream=False (default)
        model_response = cast(ModelResponse, response)
        choices = cast(list["Choices"], model_response.choices)
        return choices[0].message.content or ""

    def _parse_llm_response(
        self, response_text: str, state: SAOState
    ) -> tuple[ResponseType, Outcome | None, str | None, dict[str, Any] | None]:
        """Parse the LLM response into negotiation actions.

        Args:
            response_text: The raw LLM response.
            state: The current negotiation state.

        Returns:
            A tuple of (response_type, outcome, text, data).
        """
        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            # Default to reject with no counter if parsing fails
            return ResponseType.REJECT_OFFER, None, None, None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ResponseType.REJECT_OFFER, None, None, None

        # Parse response type
        response_type_str = data.get("response_type", "reject").lower()
        response_type_map = {
            "accept": ResponseType.ACCEPT_OFFER,
            "reject": ResponseType.REJECT_OFFER,
            "end": ResponseType.END_NEGOTIATION,
        }
        response_type = response_type_map.get(
            response_type_str, ResponseType.REJECT_OFFER
        )

        # Parse outcome
        outcome: Outcome | None = None
        outcome_data = data.get("outcome")
        if outcome_data is not None and isinstance(outcome_data, list):
            outcome = tuple(outcome_data)

        # Parse text and additional data
        text = data.get("text")
        extra_data = data.get("data")

        return response_type, outcome, text, extra_data

    def counter(
        self,
        state: SAOState,
        offer: Outcome | None,
        source: str | None = None,
        dest: str | None = None,
    ) -> SAOResponse:
        """Generate a counter-offer using the LLM.

        This method is called by the negotiation mechanism when this negotiator
        needs to respond to an offer.

        Args:
            state: The current negotiation state.
            offer: The received offer (or None if this is the first proposal).
            source: The ID of the negotiator who made the offer.
            dest: The ID of the destination negotiator.

        Returns:
            An SAOResponse containing the response type and optional counter-offer.
        """
        # Build messages
        system_prompt = self.build_system_prompt(state)
        user_message = self.build_user_message(state, offer, source)

        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
            {"role": "user", "content": user_message},
        ]

        # Call LLM
        response_text = self._call_llm(messages)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append(
            {"role": "assistant", "content": response_text}
        )

        # Parse response
        response_type, outcome, text, extra_data = self._parse_llm_response(
            response_text, state
        )

        # Build response data
        response_data: dict[str, Any] | None = None
        if text or extra_data:
            response_data = {}
            if text:
                response_data["text"] = text
            if extra_data:
                response_data.update(extra_data)

        return SAOResponse(
            response=response_type,
            outcome=outcome,
            data=response_data,
        )

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Propose an offer using the LLM.

        Args:
            state: The current negotiation state.
            dest: The ID of the destination negotiator.

        Returns:
            The proposed outcome or None.
        """
        response = self.counter(state, offer=None, source=None, dest=dest)
        if response.outcome is not None:
            if response.data:
                return ExtendedOutcome(outcome=response.outcome, data=response.data)
            return response.outcome
        return None

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer using the LLM.

        Args:
            state: The current negotiation state.
            source: The ID of the negotiator who made the offer.

        Returns:
            The response type.
        """
        response = self.counter(state, offer=state.current_offer, source=source)
        return response.response

    def on_negotiation_start(self, state: GBState) -> None:
        """Reset conversation history when negotiation starts."""
        super().on_negotiation_start(state)
        self._conversation_history = []


# =============================================================================
# Specialized subclasses for common providers
# =============================================================================


class OpenAINegotiator(LLMNegotiator):
    """LLM Negotiator using OpenAI models.

    Args:
        model: OpenAI model name (default: "gpt-4o").
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="openai",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class AnthropicNegotiator(LLMNegotiator):
    """LLM Negotiator using Anthropic Claude models.

    Args:
        model: Anthropic model name (default: "claude-sonnet-4-20250514").
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="anthropic",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class GeminiNegotiator(LLMNegotiator):
    """LLM Negotiator using Google Gemini models.

    Args:
        model: Gemini model name (default: "gemini-2.0-flash").
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="gemini",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class CohereNegotiator(LLMNegotiator):
    """LLM Negotiator using Cohere models.

    Args:
        model: Cohere model name (default: "command-r-plus").
        api_key: Cohere API key (uses COHERE_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "command-r-plus",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="cohere",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class MistralNegotiator(LLMNegotiator):
    """LLM Negotiator using Mistral AI models.

    Args:
        model: Mistral model name (default: "mistral-large-latest").
        api_key: Mistral API key (uses MISTRAL_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="mistral",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class GroqNegotiator(LLMNegotiator):
    """LLM Negotiator using Groq-hosted models.

    Args:
        model: Groq model name (default: "llama-3.3-70b-versatile").
        api_key: Groq API key (uses GROQ_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="groq",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class TogetherAINegotiator(LLMNegotiator):
    """LLM Negotiator using Together AI hosted models.

    Args:
        model: Together AI model name
            (default: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo").
        api_key: Together AI API key (uses TOGETHER_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="together_ai",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class AzureOpenAINegotiator(LLMNegotiator):
    """LLM Negotiator using Azure OpenAI Service.

    Args:
        model: Azure deployment name.
        api_key: Azure OpenAI API key.
        api_base: Azure OpenAI endpoint URL.
        api_version: Azure OpenAI API version (default: "2024-02-15-preview").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str = "2024-02-15-preview",
        **kwargs: Any,
    ) -> None:
        llm_kwargs = kwargs.pop("llm_kwargs", {}) or {}
        llm_kwargs["api_version"] = api_version
        super().__init__(
            provider="azure",
            model=model,
            api_key=api_key,
            api_base=api_base,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )


class AWSBedrockNegotiator(LLMNegotiator):
    """LLM Negotiator using AWS Bedrock.

    Args:
        model: Bedrock model ID (default: "anthropic.claude-3-sonnet-20240229-v1:0").
        aws_region: AWS region (default: "us-east-1").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        *,
        aws_region: str = "us-east-1",
        **kwargs: Any,
    ) -> None:
        llm_kwargs = kwargs.pop("llm_kwargs", {}) or {}
        llm_kwargs["aws_region_name"] = aws_region
        super().__init__(
            provider="bedrock",
            model=model,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )


# =============================================================================
# Specialized subclasses for open-source/local models
# =============================================================================


class OllamaNegotiator(LLMNegotiator):
    """LLM Negotiator using Ollama for local model inference.

    Args:
        model: Ollama model name (default: "llama3.2").
        api_base: Ollama server URL (default: "http://localhost:11434").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        api_base: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="ollama",
            model=model,
            api_base=api_base,
            **kwargs,
        )


class VLLMNegotiator(LLMNegotiator):
    """LLM Negotiator using vLLM server for local model inference.

    Args:
        model: Model name as configured in vLLM.
        api_base: vLLM server URL (default: "http://localhost:8000/v1").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str,
        *,
        api_base: str = "http://localhost:8000/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="openai",  # vLLM exposes OpenAI-compatible API
            model=model,
            api_base=api_base,
            **kwargs,
        )


class LMStudioNegotiator(LLMNegotiator):
    """LLM Negotiator using LM Studio for local model inference.

    Args:
        model: Model name (default: "local-model").
        api_base: LM Studio server URL (default: "http://localhost:1234/v1").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "local-model",
        *,
        api_base: str = "http://localhost:1234/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="openai",  # LM Studio exposes OpenAI-compatible API
            model=model,
            api_base=api_base,
            **kwargs,
        )


class TextGenWebUINegotiator(LLMNegotiator):
    """LLM Negotiator using text-generation-webui (oobabooga) server.

    Args:
        model: Model name.
        api_base: Server URL (default: "http://localhost:5000/v1").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "local-model",
        *,
        api_base: str = "http://localhost:5000/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="openai",  # oobabooga exposes OpenAI-compatible API
            model=model,
            api_base=api_base,
            **kwargs,
        )


class HuggingFaceNegotiator(LLMNegotiator):
    """LLM Negotiator using Hugging Face Inference API.

    Args:
        model: Hugging Face model ID (default: "meta-llama/Llama-3.2-3B-Instruct").
        api_key: Hugging Face API token (uses HF_TOKEN env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="huggingface",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class OpenRouterNegotiator(LLMNegotiator):
    """LLM Negotiator using OpenRouter API.

    OpenRouter provides access to many models through a unified API.

    Args:
        model: OpenRouter model name (default: "openai/gpt-4o").
        api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="openrouter",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class DeepSeekNegotiator(LLMNegotiator):
    """LLM Negotiator using DeepSeek models.

    Args:
        model: DeepSeek model name (default: "deepseek-chat").
        api_key: DeepSeek API key (uses DEEPSEEK_API_KEY env var if not provided).
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider="deepseek",
            model=model,
            api_key=api_key,
            **kwargs,
        )

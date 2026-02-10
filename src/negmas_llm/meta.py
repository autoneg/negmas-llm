"""Meta-negotiator that adds LLM-generated text to any base negotiator's offers."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm import ModelResponse
from negmas.gb.common import ExtendedResponseType
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.sao import ResponseType, SAOState

# SAOMetaNegotiator is available in negmas >= 0.16.0 (not yet released)
# We provide a helpful error message if it's not available
_SAO_META_NEGOTIATOR_AVAILABLE = False
_SAO_META_NEGOTIATOR_ERROR: str | None = None

try:
    from negmas.sao.negotiators.meta import SAOMetaNegotiator

    _SAO_META_NEGOTIATOR_AVAILABLE = True
except ImportError:
    SAOMetaNegotiator = None  # type: ignore[assignment, misc]
    _SAO_META_NEGOTIATOR_ERROR = (
        "LLMMetaNegotiator requires negmas >= 0.16.0 which includes SAOMetaNegotiator. "
        "The currently installed version of negmas does not include this class. "
        "Please upgrade negmas when a new version is released, or install from source: "
        "pip install git+https://github.com/yasserfarouk/negmas.git"
    )

if TYPE_CHECKING:
    from litellm.types.utils import Choices
    from negmas.sao import SAONegotiator

__all__ = [
    "LLMMetaNegotiator",
    "is_meta_negotiator_available",
    # LLM-wrapped native negotiators
    "LLMAspirationNegotiator",
    "LLMBoulwareTBNegotiator",
    "LLMConcederTBNegotiator",
    "LLMLinearTBNegotiator",
    "LLMTimeBasedConcedingNegotiator",
    "LLMTimeBasedNegotiator",
    "LLMNiceNegotiator",
    "LLMToughNegotiator",
    "LLMNaiveTitForTatNegotiator",
    "LLMRandomNegotiator",
    "LLMRandomAlwaysAcceptingNegotiator",
    "LLMCABNegotiator",
    "LLMCANNegotiator",
    "LLMCARNegotiator",
    "LLMMiCRONegotiator",
    "LLMFastMiCRONegotiator",
    "LLMUtilBasedNegotiator",
    "LLMWARNegotiator",
    "LLMWANNegotiator",
    "LLMWABNegotiator",
    "LLMLimitedOutcomesNegotiator",
    "LLMLimitedOutcomesAcceptor",
    "LLMHybridNegotiator",
]


def is_meta_negotiator_available() -> bool:
    """Check if LLMMetaNegotiator is available (requires negmas >= 0.16.0).

    Returns:
        True if SAOMetaNegotiator is available and LLMMetaNegotiator can be used.
    """
    return _SAO_META_NEGOTIATOR_AVAILABLE


def _check_availability() -> None:
    """Raise ImportError if SAOMetaNegotiator is not available."""
    if not _SAO_META_NEGOTIATOR_AVAILABLE:
        raise ImportError(_SAO_META_NEGOTIATOR_ERROR)


# Define the class conditionally to avoid runtime errors
# Use object as a placeholder base class when SAOMetaNegotiator is unavailable
# The __init__ will raise an informative error
_BaseClass = (  # type: ignore[assignment, misc]
    SAOMetaNegotiator if _SAO_META_NEGOTIATOR_AVAILABLE else object
)


class LLMMetaNegotiator(_BaseClass):  # type: ignore[valid-type, misc]
    """A meta-negotiator that wraps a base negotiator and adds LLM-generated text.

    This negotiator inherits from `SAOMetaNegotiator` and manages a single base
    negotiator as its child. It delegates the core negotiation strategy (what
    offers to make, when to accept) to the base negotiator, while using an LLM
    to generate persuasive or explanatory text to accompany each offer.

    The base negotiator receives all lifecycle callbacks (on_negotiation_start,
    on_round_start, etc.) through the standard `SAOMetaNegotiator` mechanism,
    ensuring it functions correctly.

    This allows combining proven negotiation strategies with natural language
    capabilities, enabling human-like communication without changing the
    underlying negotiation logic.

    Args:
        base_negotiator: The negotiator that handles the core negotiation logic.
            This negotiator's propose/respond methods determine the actual offers
            and acceptance decisions.
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM (default: 0.7).
        max_tokens: Maximum tokens in the LLM response (default: 512).
        system_prompt: Custom system prompt for text generation.
            If not provided, a default prompt focused on generating
            persuasive negotiation messages is used.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to SAOMetaNegotiator.

    Example:
        >>> from negmas.sao import BoulwareTBNegotiator
        >>> from negmas_llm import LLMMetaNegotiator
        >>> base = BoulwareTBNegotiator()
        >>> meta = LLMMetaNegotiator(
        ...     base_negotiator=base,
        ...     provider="openai",
        ...     model="gpt-4o",
        ... )
        >>> # meta will use BoulwareTBNegotiator's strategy but add
        >>> # LLM-generated text to each offer

    See Also:
        :class:`LLMNegotiator`: A negotiator where the LLM controls both
            strategy and text generation.
        :class:`negmas.sao.negotiators.meta.SAOMetaNegotiator`: The base class
            for meta-negotiators in SAO protocols.
    """

    def __init__(
        self,
        base_negotiator: SAONegotiator,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Check that SAOMetaNegotiator is available
        _check_availability()

        # Initialize with the base negotiator as our single child
        super().__init__(
            negotiators=[base_negotiator],
            negotiator_names=["base"],
            share_ufun=True,
            share_nmi=True,
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

        # Track received messages for context
        self._received_messages: list[dict[str, Any]] = []

    @property
    def base_negotiator(self) -> SAONegotiator:
        """The underlying negotiator that handles core negotiation logic."""
        return self._negotiators[0]  # type: ignore[return-value]

    # =========================================================================
    # SAOMetaNegotiator abstract method implementations
    # =========================================================================

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Get proposal from base negotiator and add LLM-generated text.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            An ExtendedOutcome with the base proposal and LLM-generated text.
        """
        # Get proposal from base negotiator
        base_proposal = self.base_negotiator.propose(state, dest=dest)

        if base_proposal is None:
            return None

        # Extract the outcome if it's already an ExtendedOutcome
        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            base_data = base_proposal.data or {}
        else:
            outcome = base_proposal
            base_data = {}

        if outcome is None:
            return None

        # Extract any text received from the other party
        received_text = self._extract_received_text(state)

        # Generate text to accompany the offer
        generated_text = self._generate_text(state, "propose", outcome, received_text)

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedOutcome(outcome=outcome, data=data)

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Get response from base negotiator and add LLM-generated text.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            An ExtendedResponseType with the base response and LLM-generated text.
        """
        # Get response from base negotiator
        base_response = self.base_negotiator.respond(state, source=source)

        # Extract any received text for context
        received_text = self._extract_received_text(state)
        if received_text:
            self._received_messages.append(
                {
                    "step": state.step,
                    "source": source,
                    "text": received_text,
                    "offer": state.current_offer,
                }
            )

        # Extract the response type if it's already an ExtendedResponseType
        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = base_response.data or {}
        else:
            response_type = base_response
            base_data = {}

        # Determine the action for text generation
        if response_type == ResponseType.ACCEPT_OFFER:
            action = "accept"
        elif response_type == ResponseType.END_NEGOTIATION:
            action = "end"
        else:
            # For rejections, text will be added with the counter-proposal
            # Only generate text here if there's received text to respond to
            if received_text:
                action = "reject"
            else:
                return base_response

        # Generate text to accompany the response
        generated_text = self._generate_text(
            state, action, state.current_offer, received_text
        )

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedResponseType(response=response_type, data=data)

    # =========================================================================
    # LLM text generation methods
    # =========================================================================

    def get_model_string(self) -> str:
        """Get the model string for litellm.

        Returns:
            The full model string in litellm format (provider/model).
        """
        return f"{self.provider}/{self.model}"

    def _build_system_prompt(self) -> str:
        """Build the system prompt for text generation.

        Returns:
            The system prompt string.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt

        return """\
You are assisting a negotiator by generating persuasive text to accompany offers.

Your role is to:
1. Generate natural, persuasive messages that explain or justify the offer
2. Consider any messages received from the other party
3. Build rapport while advancing the negotiation
4. Keep messages concise but impactful

You will receive:
- The offer being made (or acceptance/rejection decision)
- Any text received from the other party in their last offer
- Context about the negotiation state

Respond with ONLY a JSON object:
{
    "text": "Your message to accompany the offer"
}

Keep messages brief (1-3 sentences) and professional."""

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Build the user message for the LLM.

        Args:
            state: The current negotiation state.
            action: The action being taken ("propose", "accept", "reject", "end").
            outcome: The outcome being proposed (if any).
            received_text: Text received from the other party (if any).

        Returns:
            The user message string.
        """
        parts = [f"Negotiation round {state.step} (time: {state.relative_time:.1%})"]
        parts.append("")

        if received_text:
            parts.append(f'Message from other party: "{received_text}"')
            parts.append("")

        if action == "propose":
            parts.append(f"You are making an offer: {outcome}")
            parts.append("Generate a message to accompany this offer.")
        elif action == "accept":
            parts.append(f"You are ACCEPTING the offer: {state.current_offer}")
            parts.append("Generate a brief acceptance message.")
        elif action == "reject":
            if outcome:
                parts.append(
                    f"You are REJECTING the current offer and "
                    f"counter-proposing: {outcome}"
                )
                parts.append("Generate a message explaining your counter-offer.")
            else:
                parts.append("You are REJECTING the current offer.")
                parts.append("Generate a brief rejection message.")
        elif action == "end":
            parts.append("You are ENDING the negotiation without agreement.")
            parts.append("Generate a brief closing message.")

        return "\n".join(parts)

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
        model_response = cast(ModelResponse, response)
        choices = cast(list["Choices"], model_response.choices)
        return choices[0].message.content or ""

    def _parse_text_response(self, response_text: str) -> str:
        """Parse the LLM response to extract the text message.

        Args:
            response_text: The raw LLM response.

        Returns:
            The extracted text message.
        """
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "text" in data:
                    return str(data["text"])
            except json.JSONDecodeError:
                pass

        # Fallback: use the response as-is (stripped)
        return response_text.strip()

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Generate text to accompany an offer or response.

        Args:
            state: The current negotiation state.
            action: The action being taken.
            outcome: The outcome being proposed (if any).
            received_text: Text received from the other party (if any).

        Returns:
            The generated text message.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(state, action, outcome, received_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages)
        return self._parse_text_response(response_text)

    def _extract_received_text(self, state: SAOState) -> str | None:
        """Extract text from the most recent offer received.

        Args:
            state: The current negotiation state.

        Returns:
            The text from the received offer, or None if not available.
        """
        # Check current_data for text
        if state.current_data and isinstance(state.current_data, dict):
            text = state.current_data.get("text")
            if text:
                return str(text)

        # Check new_data for the most recent text
        if state.new_data:
            for data in reversed(state.new_data):
                if data and isinstance(data, dict):
                    text = data.get("text")
                    if text:
                        return str(text)

        return None

    def on_negotiation_start(self, state) -> None:
        """Reset received messages when negotiation starts.

        Args:
            state: The initial negotiation state.
        """
        super().on_negotiation_start(state)
        self._received_messages = []


# =============================================================================
# LLM-wrapped versions of native negmas negotiators
# =============================================================================
#
# These classes provide convenient wrappers around native negmas negotiators,
# adding LLM-generated text to their offers while preserving the original
# negotiation strategy.
#
# Each class inherits from LLMMetaNegotiator and pre-configures it with the
# appropriate base negotiator class.


def _create_llm_negotiator_class(
    base_class_name: str,
    base_module: str,
    class_name: str,
    docstring: str,
) -> type:
    """Factory function to create LLM-wrapped negotiator classes.

    Args:
        base_class_name: Name of the base negotiator class.
        base_module: Module containing the base class.
        class_name: Name for the new LLM-wrapped class.
        docstring: Docstring for the new class.

    Returns:
        A new class that wraps the base negotiator with LLM text generation.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        base_negotiator_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        _check_availability()

        # Import the base class dynamically
        import importlib

        module = importlib.import_module(base_module)
        base_cls = getattr(module, base_class_name)

        # Create the base negotiator
        base_kwargs = base_negotiator_kwargs or {}
        base_negotiator = base_cls(**base_kwargs)

        # Initialize the LLMMetaNegotiator
        super(self.__class__, self).__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )

    # Create the class dynamically
    new_class = type(
        class_name,
        (LLMMetaNegotiator,),
        {
            "__init__": __init__,
            "__doc__": docstring,
            "__module__": __name__,
            "_base_class_name": base_class_name,
            "_base_module": base_module,
        },
    )

    return new_class


# Time-based negotiators
LLMAspirationNegotiator = _create_llm_negotiator_class(
    base_class_name="AspirationNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMAspirationNegotiator",
    docstring="""\
LLM-wrapped AspirationNegotiator.

This negotiator uses the aspiration-based time-dependent strategy for making
offers and acceptance decisions, while adding LLM-generated persuasive text
to accompany each offer.

The aspiration negotiator starts with high aspirations (demanding offers)
and gradually lowers them over time.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMBoulwareTBNegotiator = _create_llm_negotiator_class(
    base_class_name="BoulwareTBNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMBoulwareTBNegotiator",
    docstring="""\
LLM-wrapped BoulwareTBNegotiator.

This negotiator uses the Boulware time-based concession strategy, which
concedes slowly at first and more rapidly near the deadline. LLM-generated
persuasive text accompanies each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMConcederTBNegotiator = _create_llm_negotiator_class(
    base_class_name="ConcederTBNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMConcederTBNegotiator",
    docstring="""\
LLM-wrapped ConcederTBNegotiator.

This negotiator uses the Conceder time-based strategy, which concedes
rapidly at first and slows down near the deadline. LLM-generated text
accompanies each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMLinearTBNegotiator = _create_llm_negotiator_class(
    base_class_name="LinearTBNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMLinearTBNegotiator",
    docstring="""\
LLM-wrapped LinearTBNegotiator.

This negotiator uses a linear time-based concession strategy, conceding
at a constant rate over time. LLM-generated text accompanies each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMTimeBasedConcedingNegotiator = _create_llm_negotiator_class(
    base_class_name="TimeBasedConcedingNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMTimeBasedConcedingNegotiator",
    docstring="""\
LLM-wrapped TimeBasedConcedingNegotiator.

A configurable time-based conceding negotiator with LLM-generated text
accompanying each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMTimeBasedNegotiator = _create_llm_negotiator_class(
    base_class_name="TimeBasedNegotiator",
    base_module="negmas.gb.negotiators.timebased",
    class_name="LLMTimeBasedNegotiator",
    docstring="""\
LLM-wrapped TimeBasedNegotiator.

A general time-based negotiator with LLM-generated text accompanying
each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Nice and Tough negotiators
LLMNiceNegotiator = _create_llm_negotiator_class(
    base_class_name="NiceNegotiator",
    base_module="negmas.gb.negotiators.nice",
    class_name="LLMNiceNegotiator",
    docstring="""\
LLM-wrapped NiceNegotiator.

A cooperative negotiator that makes nice offers with LLM-generated
persuasive text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMToughNegotiator = _create_llm_negotiator_class(
    base_class_name="ToughNegotiator",
    base_module="negmas.gb.negotiators.tough",
    class_name="LLMToughNegotiator",
    docstring="""\
LLM-wrapped ToughNegotiator.

An aggressive negotiator that makes tough offers with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Tit-for-tat negotiators
LLMNaiveTitForTatNegotiator = _create_llm_negotiator_class(
    base_class_name="NaiveTitForTatNegotiator",
    base_module="negmas.gb.negotiators.titfortat",
    class_name="LLMNaiveTitForTatNegotiator",
    docstring="""\
LLM-wrapped NaiveTitForTatNegotiator.

A tit-for-tat negotiator that mirrors the opponent's behavior with
LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Random negotiators
LLMRandomNegotiator = _create_llm_negotiator_class(
    base_class_name="RandomNegotiator",
    base_module="negmas.gb.negotiators.randneg",
    class_name="LLMRandomNegotiator",
    docstring="""\
LLM-wrapped RandomNegotiator.

A negotiator that makes random offers with LLM-generated text. Useful
for testing and baselines.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMRandomAlwaysAcceptingNegotiator = _create_llm_negotiator_class(
    base_class_name="RandomAlwaysAcceptingNegotiator",
    base_module="negmas.gb.negotiators.randneg",
    class_name="LLMRandomAlwaysAcceptingNegotiator",
    docstring="""\
LLM-wrapped RandomAlwaysAcceptingNegotiator.

A negotiator that makes random offers but always accepts with LLM-generated
text. Useful for testing.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# CAB/CAN/CAR negotiators (Curve-based)
LLMCABNegotiator = _create_llm_negotiator_class(
    base_class_name="CABNegotiator",
    base_module="negmas.gb.negotiators.cab",
    class_name="LLMCABNegotiator",
    docstring="""\
LLM-wrapped CABNegotiator.

Curve-based Aspiration Boulware negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMCANNegotiator = _create_llm_negotiator_class(
    base_class_name="CANNegotiator",
    base_module="negmas.gb.negotiators.cab",
    class_name="LLMCANNegotiator",
    docstring="""\
LLM-wrapped CANNegotiator.

Curve-based Aspiration Nice negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMCARNegotiator = _create_llm_negotiator_class(
    base_class_name="CARNegotiator",
    base_module="negmas.gb.negotiators.cab",
    class_name="LLMCARNegotiator",
    docstring="""\
LLM-wrapped CARNegotiator.

Curve-based Aspiration Random negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# MiCRO negotiators
LLMMiCRONegotiator = _create_llm_negotiator_class(
    base_class_name="MiCRONegotiator",
    base_module="negmas.gb.negotiators.micro",
    class_name="LLMMiCRONegotiator",
    docstring="""\
LLM-wrapped MiCRONegotiator.

MiCRO (Mixed strategy with CRoss Offers) negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMFastMiCRONegotiator = _create_llm_negotiator_class(
    base_class_name="FastMiCRONegotiator",
    base_module="negmas.gb.negotiators.micro",
    class_name="LLMFastMiCRONegotiator",
    docstring="""\
LLM-wrapped FastMiCRONegotiator.

Fast version of MiCRO negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Utility-based negotiators
LLMUtilBasedNegotiator = _create_llm_negotiator_class(
    base_class_name="UtilBasedNegotiator",
    base_module="negmas.gb.negotiators.utilbased",
    class_name="LLMUtilBasedNegotiator",
    docstring="""\
LLM-wrapped UtilBasedNegotiator.

Utility-based negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# WAR/WAN/WAB negotiators
LLMWARNegotiator = _create_llm_negotiator_class(
    base_class_name="WARNegotiator",
    base_module="negmas.gb.negotiators.war",
    class_name="LLMWARNegotiator",
    docstring="""\
LLM-wrapped WARNegotiator.

War negotiator (aggressive) with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMWANNegotiator = _create_llm_negotiator_class(
    base_class_name="WANNegotiator",
    base_module="negmas.gb.negotiators.war",
    class_name="LLMWANNegotiator",
    docstring="""\
LLM-wrapped WANNegotiator.

War-Aspiration-Nice negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMWABNegotiator = _create_llm_negotiator_class(
    base_class_name="WABNegotiator",
    base_module="negmas.gb.negotiators.war",
    class_name="LLMWABNegotiator",
    docstring="""\
LLM-wrapped WABNegotiator.

War-Aspiration-Boulware negotiator with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Limited outcomes negotiators
LLMLimitedOutcomesNegotiator = _create_llm_negotiator_class(
    base_class_name="LimitedOutcomesNegotiator",
    base_module="negmas.gb.negotiators.limited",
    class_name="LLMLimitedOutcomesNegotiator",
    docstring="""\
LLM-wrapped LimitedOutcomesNegotiator.

A negotiator that works with a limited set of outcomes, with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


LLMLimitedOutcomesAcceptor = _create_llm_negotiator_class(
    base_class_name="LimitedOutcomesAcceptor",
    base_module="negmas.gb.negotiators.limited",
    class_name="LLMLimitedOutcomesAcceptor",
    docstring="""\
LLM-wrapped LimitedOutcomesAcceptor.

An acceptor that works with a limited set of outcomes, with LLM-generated text.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)


# Hybrid negotiator
LLMHybridNegotiator = _create_llm_negotiator_class(
    base_class_name="HybridNegotiator",
    base_module="negmas.gb.negotiators.hybrid",
    class_name="LLMHybridNegotiator",
    docstring="""\
LLM-wrapped HybridNegotiator.

A hybrid negotiator that combines multiple strategies using HybridOfferingPolicy
and ACNext acceptance policy, with LLM-generated text accompanying each offer.

Args:
    provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
    model: The model name (e.g., "gpt-4", "claude-3-opus").
    api_key: API key for the provider (if required).
    api_base: Base URL for the API.
    temperature: Sampling temperature for the LLM (default: 0.7).
    max_tokens: Maximum tokens in the LLM response (default: 512).
    system_prompt: Custom system prompt for text generation.
    llm_kwargs: Additional keyword arguments passed to litellm.completion.
    base_negotiator_kwargs: Keyword arguments passed to the base negotiator.
        Supports `alpha` and `beta` parameters for the acceptance policy.
    **kwargs: Additional arguments passed to SAOMetaNegotiator.
""",
)

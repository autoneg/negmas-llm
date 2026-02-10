"""LLM-based negotiators for the negmas framework."""

from __future__ import annotations

import json
import re
import textwrap
from abc import ABC
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm import ModelResponse
from negmas import Agent, Controller, Preferences
from negmas.common import PreferencesChange
from negmas.gb import GBState
from negmas.inout import serialize
from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.base import SAOCallNegotiator

from negmas_llm.tags import process_prompt as _process_prompt

if TYPE_CHECKING:
    from litellm.types.utils import Choices

__all__ = [
    "LLMNegotiator",
    # Cloud providers
    "OpenAINegotiator",
    "AnthropicNegotiator",
    "GeminiNegotiator",
    "CohereNegotiator",
    "MistralNegotiator",
    "GroqNegotiator",
    "TogetherAINegotiator",
    "AzureOpenAINegotiator",
    "AWSBedrockNegotiator",
    "GitHubCopilotNegotiator",
    # Local/open-source
    "OllamaNegotiator",
    "VLLMNegotiator",
    "LMStudioNegotiator",
    "TextGenWebUINegotiator",
    "HuggingFaceNegotiator",
    "OpenRouterNegotiator",
    "DeepSeekNegotiator",
]


def _dedent(text: str) -> str:
    """Dedent a multi-line string, stripping the first line if empty.

    This allows triple-quoted strings to be written with proper indentation:

        message = _dedent('''
            First line here
            Second line here
        ''')

    Results in:
        "First line here\\nSecond line here\\n"

    Args:
        text: The text to dedent.

    Returns:
        The dedented text with consistent indentation removed.
    """
    # Remove leading newline if present (from ''' starting on its own line)
    if text.startswith("\n"):
        text = text[1:]
    return textwrap.dedent(text)


# =============================================================================
# Docstrings from negmas for LLM context
# =============================================================================

_SAONMI_DOCSTRING = _dedent("""
    The Negotiator Mechanism Interface (NMI) provides information about the negotiation:
    - n_steps: Maximum number of negotiation steps (None = unlimited)
    - time_limit: Maximum time in seconds (None = unlimited)
    - n_outcomes: Total number of possible outcomes in the outcome space
    - n_negotiators: Number of participants in this negotiation
    - end_on_no_response: If true, negotiation ends when any negotiator returns None
    - one_offer_per_step: If true, only one negotiator acts per step
    - offering_is_accepting: If true, making an offer implies accepting it if echoed
    """)

_SAOSTATE_DOCSTRING = _dedent("""
    The negotiation state contains:
    - step: Current negotiation step number (0-indexed)
    - relative_time: Progress through negotiation (0.0 to 1.0)
    - running: Whether negotiation is still active
    - current_offer: The offer currently on the table (or None)
    - current_proposer: ID of who made the current offer
    - n_acceptances: Number of acceptances for current offer
    - broken: True if negotiation ended abnormally
    - timedout: True if negotiation exceeded time limit
    - agreement: The final agreed outcome (if any)
    """)

_UFUN_DOCSTRING = _dedent("""
    A utility function maps outcomes to real numbers representing preference.
    - Higher values = more preferred outcomes
    - reserved_value: The utility of no agreement (your walk-away point)
    - You should aim to get outcomes with utility > reserved_value
    """)

# =============================================================================
# Structured Output Schema for LLM Responses
# =============================================================================

# JSON Schema for negotiation response (used with structured outputs)
_NEGOTIATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "negotiation_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "response_type": {
                    "type": "string",
                    "enum": ["accept", "reject", "end", "wait"],
                    "description": "The negotiation action to take",
                },
                "outcome": {
                    "type": ["array", "null"],
                    "items": {},
                    "description": "Counter-offer as list of values, or null",
                },
                "text": {
                    "type": ["string", "null"],
                    "description": "Optional persuasive message for opponent",
                },
                "reasoning": {
                    "type": ["string", "null"],
                    "description": "Brief explanation of your decision",
                },
            },
            "required": ["response_type", "outcome", "text", "reasoning"],
            "additionalProperties": False,
        },
    },
}

# Providers that support structured outputs (JSON mode with schema)
# These providers guarantee valid JSON matching the schema
_STRUCTURED_OUTPUT_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "azure",
        "gemini",
        "anthropic",
    }
)

# Headers required for GitHub Copilot (simulates IDE client)
_GITHUB_COPILOT_HEADERS: dict[str, str] = {
    "editor-version": "vscode/1.85.1",
    "editor-plugin-version": "copilot/1.155.0",
    "user-agent": "GithubCopilot/1.155.0",
}

# Providers that support basic JSON mode (valid JSON, but no schema guarantee)
_JSON_MODE_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "azure",
        "gemini",
        "anthropic",
        "mistral",
        "groq",
        "ollama",
        "deepseek",
        "together_ai",
        "cohere",
    }
)


def _supports_structured_output(provider: str) -> bool:
    """Check if a provider supports structured output with JSON schema."""
    return provider.lower() in _STRUCTURED_OUTPUT_PROVIDERS


def _supports_json_mode(provider: str) -> bool:
    """Check if a provider supports basic JSON mode."""
    return provider.lower() in _JSON_MODE_PROVIDERS


class LLMNegotiator(SAOCallNegotiator, ABC):
    """A negotiator that uses an LLM for decision-making.

    This negotiator delegates the negotiation strategy to a Large Language Model.
    Each negotiation is treated as a conversation with the LLM:

    1. on_preferences_changed: Informs LLM of outcome space, utility functions
    2. on_negotiation_start: Signals negotiation is starting
    3. __call__: For each round, sends state and asks for decision + counter-offer

    The LLM returns structured responses with:
    - response_type: accept/reject/end/wait
    - outcome: Counter-offer if rejecting
    - text: Optional persuasion text for the opponent

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
        model: The model name (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens in the LLM response.
        use_structured_output: If True (default), use structured output/JSON mode
            when the provider supports it. This guarantees valid JSON responses.
            Set to False to disable and rely on prompt-based JSON extraction.
        system_prompt: Custom system prompt for the LLM conversation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        preferences: The preferences of the negotiator.
        ufun: The utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent Controller if any.
        owner: The Agent that owns the negotiator.
        id: Unique ID for the negotiator.
        type_name: Type name string.
        can_propose: Whether the negotiator can propose offers.
        **kwargs: Additional arguments passed to SAOCallNegotiator.
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
        use_structured_output: bool = True,
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
        self.use_structured_output = use_structured_output
        self._custom_system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs or {}

        # Conversation history for the LLM (persistent across rounds)
        self._conversation_history: list[dict[str, str]] = []
        # Track if preferences have been sent
        self._preferences_sent: bool = False

    def get_model_string(self) -> str:
        """Get the model string for litellm.

        Returns:
            The full model string in litellm format (provider/model).
        """
        return f"{self.provider}/{self.model}"

    def process_prompt(self, prompt: str, state: SAOState | None = None) -> str:
        """Process a prompt, replacing all tags with their values.

        This method processes tags in the format {{tag_name:format(params)}}.
        Tags are automatically replaced with dynamic content based on the
        current negotiation context.

        Supported tags include:
        - {{outcome-space}} or {{outcome-space:json}}: The negotiation outcome space
        - {{utility-function}}: Your utility function
        - {{opponent-utility-function}}: Opponent's utility function (if known)
        - {{nmi}}: Negotiation mechanism interface information
        - {{current-state}}: Current negotiation state
        - {{reserved-value}}: Your reserved value (walk-away point)
        - {{opponent-reserved-value}}: Opponent's reserved value (if known)
        - {{my-last-offer}}, {{my-first-offer}}: Your offers
        - {{opponent-last-offer}}, {{opponent-first-offer}}: Opponent's offers
        - {{history:text(k=5)}}: Last k items from negotiation history
        - {{trace}}, {{extended-trace}}, {{full-trace}}: Negotiation traces
        - {{utility:text(outcome={{opponent-last-offer}})}}: Compute utility

        Args:
            prompt: The prompt string containing tags.
            state: The current SAO state (optional).

        Returns:
            The prompt with all tags replaced.
        """
        return _process_prompt(prompt, self, state)  # type: ignore

    # =========================================================================
    # LLM Communication
    # =========================================================================

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        require_json: bool = False,
    ) -> str:
        """Call the LLM and get a response.

        Args:
            messages: The conversation messages.
            require_json: If True and provider supports it, enforce JSON output.

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

        # Add structured output / JSON mode if requested and supported
        if require_json and self.use_structured_output:
            if _supports_structured_output(self.provider):
                # Use full JSON schema for providers that support it
                kwargs["response_format"] = _NEGOTIATION_RESPONSE_SCHEMA
            elif _supports_json_mode(self.provider):
                # Fall back to basic JSON mode
                kwargs["response_format"] = {"type": "json_object"}

        # Add required headers for GitHub Copilot (simulates IDE client)
        if self.provider == "github_copilot":
            extra_headers = kwargs.get("extra_headers", {})
            extra_headers.update(_GITHUB_COPILOT_HEADERS)
            kwargs["extra_headers"] = extra_headers

        response = litellm.completion(**kwargs)
        model_response = cast(ModelResponse, response)
        choices = cast(list["Choices"], model_response.choices)
        return choices[0].message.content or ""

    def _send_to_llm(
        self,
        message: str,
        role: str = "user",
        require_json: bool = False,
    ) -> str:
        """Send a message to the LLM and get a response.

        This adds the message to conversation history and returns the response.

        Args:
            message: The message content.
            role: The role (user or system).
            require_json: If True, enforce JSON output mode if supported.

        Returns:
            The LLM response text.
        """
        # Build messages: system prompt + conversation history + new message
        messages = []

        if self._custom_system_prompt:
            messages.append({"role": "system", "content": self._custom_system_prompt})
        else:
            messages.append(
                {"role": "system", "content": self._build_base_system_prompt()}
            )

        messages.extend(self._conversation_history)
        messages.append({"role": role, "content": message})

        response = self._call_llm(messages, require_json=require_json)

        # Update conversation history
        self._conversation_history.append({"role": role, "content": message})
        self._conversation_history.append({"role": "assistant", "content": response})

        return response

    def _build_base_system_prompt(self) -> str:
        """Build the base system prompt that sets up the LLM as a negotiator."""
        return _dedent("""
            You are an expert negotiator participating in an automated negotiation.
            Your goal is to negotiate effectively to achieve good outcomes for yourself
            while finding mutually acceptable agreements when possible.

            You will receive information about the negotiation setup (outcome space,
            utility functions) at the start, and then be asked to make decisions for
            each negotiation round.

            Always respond in the exact JSON format requested. Be strategic and
            rational, aiming to maximize your utility while reaching agreements.
            """)

    # =========================================================================
    # Negotiation Lifecycle Callbacks
    # =========================================================================

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Called when preferences (utility function) change.

        This sends the negotiation context to the LLM including:
        - NMI (mechanism information)
        - Outcome space
        - Your utility function
        - Opponent's utility function (if known)
        - Whether this is initial setup or a change

        Args:
            changes: List of preference changes that occurred.
        """
        super().on_preferences_changed(changes)

        # Determine if this is first time or a change
        is_first = not self._preferences_sent
        self._preferences_sent = True

        # Build the preferences message
        parts = []

        if is_first:
            parts.append("# Negotiation Setup\n")
            parts.append(
                "You are about to participate in a negotiation. Here is the setup:\n"
            )
        else:
            parts.append("# Preferences Changed\n")
            change_types = [c.type.name for c in changes]
            change_str = ", ".join(change_types)
            parts.append(f"Your preferences have changed. Change types: {change_str}\n")

        # Add NMI info
        parts.append(f"\n## Mechanism Information\n{_SAONMI_DOCSTRING}\n")
        parts.append("{{nmi:text}}\n")

        # Add outcome space
        parts.append("\n## Outcome Space\n")
        parts.append("The negotiation outcome space defines possible agreements:\n")
        parts.append("{{outcome-space:json}}\n")

        # Add utility function
        parts.append(f"\n## Your Utility Function\n{_UFUN_DOCSTRING}\n")
        parts.append("{{utility-function:text}}\n")
        parts.append(
            "Your reserved value (utility of no agreement): {{reserved-value}}\n"
        )

        # Add opponent utility if known
        parts.append("\n## Opponent's Utility Function\n")
        parts.append("{{opponent-utility-function:text}}\n")

        message = self.process_prompt("".join(parts), state=None)

        # Send to LLM (don't need to use the response, just inform it)
        response = self._send_to_llm(message)
        _ = response  # LLM acknowledgment

    def on_negotiation_start(self, state: GBState) -> None:
        """Called when negotiation starts.

        Informs the LLM that negotiation is beginning.

        Args:
            state: The initial negotiation state.
        """
        super().on_negotiation_start(state)

        message = _dedent("""
            # Negotiation Started

            The negotiation has now started. For each round, you will be asked to:
            1. Analyze the current state and any offer received
            2. Decide whether to ACCEPT, REJECT (with counter-offer), or END
            3. Optionally provide persuasive text for the other party

            Respond in this JSON format for each decision:
            ```json
            {
                "response_type": "accept" | "reject" | "end" | "wait",
                "outcome": [value1, value2, ...] | null,
                "text": "optional persuasive message to opponent",
                "reasoning": "brief explanation of your decision (for your records)"
            }
            ```

            Where:
            - "accept": Accept the current offer on the table
            - "reject": Reject and provide a counter-offer in "outcome"
            - "end": End the negotiation without agreement
            - "wait": Wait without making an offer (only if allowed by mechanism)
            - "outcome": Your counter-offer as a list matching issue order, or null

            You may occasionally send ONLY text (null outcome) to persuade the
            opponent, but this should be rare and strategic. Include an outcome usually.

            Ready to begin!
            """)
        response = self._send_to_llm(message)
        _ = response

    # =========================================================================
    # Main Decision Method (__call__)
    # =========================================================================

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Make a negotiation decision.

        This is the main method called each round. It:
        1. Formats the current state for the LLM
        2. Asks the LLM for a decision
        3. Parses the response into an SAOResponse

        Args:
            state: The current negotiation state.
            dest: Target negotiator ID (None broadcasts to all).

        Returns:
            SAOResponse with response type, outcome, and optional data.
        """
        # Build the round message
        parts = [f"# Round {state.step + 1}\n"]

        # Current state (concise)
        parts.append(f"**Step**: {state.step} | ")
        parts.append(f"**Time**: {state.relative_time:.1%} | ")
        parts.append(f"**Running**: {state.running}\n\n")

        # Current offer
        if state.current_offer is not None:
            offer_str = self._format_outcome(state.current_offer)
            parts.append(f"**Offer on table**: {offer_str}")
            if state.current_proposer:
                is_mine = state.current_proposer == self.id
                proposer_label = "You" if is_mine else state.current_proposer
                parts.append(f" (from {proposer_label})")
            parts.append("\n")

            # Utility of current offer
            if self.ufun is not None:
                utility = self.ufun(state.current_offer)
                reserved = self.reserved_value
                parts.append(f"**Your utility for this offer**: {utility:.4f}")
                if reserved is not None:
                    parts.append(f" (reserved value: {reserved:.4f})")
                parts.append("\n")
        else:
            parts.append("**No offer on table** - you may make a proposal.\n")

        # Ask for decision
        parts.append("\nWhat is your decision? Respond with JSON.")

        message = "".join(parts)
        response_text = self._send_to_llm(message, require_json=True)

        # Parse the response
        response_type, outcome, text, reasoning = self._parse_llm_response(
            response_text, state
        )

        # Build response data
        response_data: dict[str, Any] | None = None
        if text or reasoning:
            response_data = {}
            if text:
                response_data["text"] = text
            if reasoning:
                response_data["reasoning"] = reasoning

        return SAOResponse(
            response=response_type,
            outcome=outcome,
            data=response_data,
        )

    def _parse_llm_response(
        self, response_text: str, state: SAOState
    ) -> tuple[ResponseType, Outcome | None, str | None, str | None]:
        """Parse the LLM response into negotiation actions.

        Args:
            response_text: The raw LLM response.
            state: The current negotiation state.

        Returns:
            A tuple of (response_type, outcome, text, reasoning).
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
            "wait": ResponseType.WAIT,
            "no_response": ResponseType.NO_RESPONSE,
        }
        response_type = response_type_map.get(
            response_type_str, ResponseType.REJECT_OFFER
        )

        # Parse outcome
        outcome: Outcome | None = None
        outcome_data = data.get("outcome")
        if outcome_data is not None and isinstance(outcome_data, list):
            outcome = tuple(outcome_data)

        # Parse text and reasoning
        text = data.get("text")
        reasoning = data.get("reasoning")

        return response_type, outcome, text, reasoning

    def _format_outcome(self, outcome: Outcome) -> str:
        """Format an outcome for display.

        Args:
            outcome: The outcome to format.

        Returns:
            A formatted string representation.
        """
        if self.nmi is not None and self.nmi.outcome_space is not None:
            outcome_space = self.nmi.outcome_space
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

    # =========================================================================
    # Legacy format methods (kept for compatibility with components/meta)
    # =========================================================================

    def format_outcome_space(self, state: SAOState) -> str:
        """Format the outcome space for the LLM."""
        if self.nmi is None or self.nmi.outcome_space is None:
            return ""

        outcome_space = self.nmi.outcome_space
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

    def format_own_ufun(self, state: SAOState) -> str:
        """Format your own utility function for the LLM."""
        if self.ufun is None:
            return _dedent("""
                ## Your Utility Function

                You do NOT have a utility function. You must negotiate based on general
                principles and any instructions provided.
                """)

        try:
            ufun_dict = serialize(self.ufun)
            ufun_dict.pop("__python_class__", None)
            reserved = self.reserved_value
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
            return _dedent(f"""
                ## Your Utility Function

                You have a utility function: {self.ufun}
                Reserved value (utility if no agreement): {self.reserved_value}
                """)

    def format_partner_ufun(self, state: SAOState) -> str:
        """Format the partner's utility function for the LLM."""
        partner_ufun = (
            self.private_info.get("opponent_ufun") if self.private_info else None
        )

        if partner_ufun is None:
            return _dedent("""
                ## Partner's Utility Function

                You do NOT know your partner's utility function. You must infer their
                preferences from their offers and behavior during the negotiation.
                """)

        try:
            ufun_dict = serialize(partner_ufun)
            ufun_dict.pop("__python_class__", None)
            reserved = getattr(partner_ufun, "reserved_value", None)
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
            return _dedent(f"""
                ## Partner's Utility Function (Known)

                Your partner's utility function: {partner_ufun}
                """)

    def format_nmi_info(self) -> str:
        """Format the Negotiator Mechanism Interface (NMI) information."""
        if self.nmi is None:
            return ""

        parts = ["## Negotiation Mechanism Information"]
        parts.append("")

        n_steps = self.nmi.n_steps
        if n_steps is not None:
            parts.append(f"- **Maximum steps**: {n_steps}")
        else:
            parts.append("- **Maximum steps**: Unlimited")

        time_limit = self.nmi.time_limit
        if time_limit is not None:
            parts.append(f"- **Time limit**: {time_limit:.2f} seconds")
        else:
            parts.append("- **Time limit**: Unlimited")

        n_outcomes = self.nmi.n_outcomes
        if n_outcomes is not None:
            parts.append(f"- **Total possible outcomes**: {n_outcomes}")

        n_negotiators = self.nmi.n_negotiators
        if n_negotiators is not None:
            parts.append(f"- **Number of negotiators**: {n_negotiators}")

        if hasattr(self.nmi, "dynamic_entry") and self.nmi.dynamic_entry is not None:
            parts.append(f"- **Dynamic entry allowed**: {self.nmi.dynamic_entry}")

        if hasattr(self.nmi, "annotation") and self.nmi.annotation:
            parts.append(f"- **Mechanism annotation**: {self.nmi.annotation}")

        parts.append("")
        return "\n".join(parts)

    def format_state(self, state: SAOState, offer: Outcome | None = None) -> str:
        """Format the complete SAOState for the LLM."""
        parts = ["## Current State"]
        parts.append("")

        parts.append(f"- **Step**: {state.step}")
        parts.append(f"- **Relative time**: {state.relative_time:.2%}")

        display_offer = offer if offer is not None else state.current_offer
        if display_offer is not None:
            offer_str = self._format_outcome(display_offer)
            parts.append(f"- **Current offer on table**: {offer_str}")
            if state.current_proposer:
                parts.append(f"- **Current proposer**: {state.current_proposer}")

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

        parts.append(f"- **Number of acceptances**: {state.n_acceptances}")
        parts.append(f"- **Negotiation running**: {state.running}")

        if state.broken:
            parts.append("- **Status**: BROKEN (negotiation ended abnormally)")
        if state.timedout:
            parts.append("- **Status**: TIMED OUT")
        if state.agreement is not None:
            agreement_str = self._format_outcome(state.agreement)
            parts.append(f"- **Agreement reached**: {agreement_str}")

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


class GitHubCopilotNegotiator(LLMNegotiator):
    """LLM Negotiator using GitHub Copilot.

    GitHub Copilot uses OAuth device flow for authentication. On first use,
    you'll be prompted to visit a URL and enter a code to authenticate.
    After authentication, tokens are cached locally.

    Pre-authentication: Set GITHUB_COPILOT_TOKEN_DIR environment variable
    to point to a directory containing cached tokens (access-token file).

    Args:
        model: Model name (default: "gpt-4o").
        **kwargs: Additional arguments passed to LLMNegotiator.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        # Ensure extra_headers are included via llm_kwargs
        llm_kwargs = kwargs.pop("llm_kwargs", {}) or {}
        extra_headers = llm_kwargs.get("extra_headers", {})
        extra_headers.update(_GITHUB_COPILOT_HEADERS)
        llm_kwargs["extra_headers"] = extra_headers
        super().__init__(
            provider="github_copilot",
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

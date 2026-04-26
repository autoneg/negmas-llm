# Customization

negmas-llm is designed to be highly customizable. You can override various methods
to control how information is presented to the LLM and how responses are parsed.

## Customization Points

The `LLMNegotiator` class provides several methods you can override:

| Method | Purpose |
|--------|---------|
| `format_outcome_space()` | Format the negotiation outcome space |
| `format_own_ufun()` | Format your utility function |
| `format_partner_ufun()` | Format the partner's utility function |
| `format_nmi_info()` | Format negotiation mechanism metadata |
| `format_state()` | Format the complete negotiation state |
| `format_response_instructions()` | Define the expected JSON response format |
| `build_system_prompt()` | Build the complete system prompt |
| `build_user_message()` | Build the user message for each round |
| `_parse_llm_response()` | Parse LLM responses into actions |

## Example: Custom System Prompt

The simplest customization is providing a custom system prompt:

```python
from negmas_llm import OpenAINegotiator

negotiator = OpenAINegotiator(
    model="gpt-4o",
    system_prompt="""You are a tough negotiator who never accepts the first offer.
    Always counter with a better deal for yourself.
    Respond in JSON format with response_type, outcome, and text fields.""",
    ufun=my_utility_function,
)
```

## Example: Custom Formatting Methods

For more control, create a subclass and override specific methods:

```python
from negmas_llm import OpenAINegotiator
from negmas.sao import SAOState


class VerboseNegotiator(OpenAINegotiator):
    """A negotiator that provides more detailed state information."""

    def format_state(self, state: SAOState, offer=None) -> str:
        """Add extra context to state formatting."""
        base_state = super().format_state(state, offer)

        # Add custom analysis
        extra_info = []
        extra_info.append("## Strategic Analysis")
        extra_info.append("")

        # Time pressure indicator
        if state.relative_time > 0.8:
            extra_info.append("**WARNING**: Running low on time!")
        elif state.relative_time > 0.5:
            extra_info.append("**Note**: Past the halfway point.")

        # Utility analysis if we have an offer
        if offer is not None and self.ufun is not None:
            utility = self.ufun(offer)
            reserved = self.reserved_value
            if utility < reserved:
                extra_info.append(
                    f"**Alert**: This offer ({utility:.3f}) is BELOW "
                    f"your reservation value ({reserved:.3f})!"
                )
            elif utility > reserved * 1.5:
                extra_info.append(
                    f"**Opportunity**: This offer provides good value "
                    f"({utility:.3f} vs reserved {reserved:.3f})."
                )

        extra_info.append("")
        return base_state + "\n".join(extra_info)
```

## Example: Custom Response Format

Change the expected response format for your LLM:

```python
import json
import re
from negmas.sao import ResponseType, SAOState
from negmas.outcomes import Outcome
from typing import Any

from negmas_llm import AnthropicNegotiator


class XMLResponseNegotiator(AnthropicNegotiator):
    """A negotiator that expects XML-formatted responses."""

    def format_response_instructions(self) -> str:
        return """\
## Response Format

Respond in XML format:
<response>
    <action>accept|reject|end</action>
    <offer>value1, value2, ...</offer>
    <reasoning>your explanation</reasoning>
</response>
"""

    def _parse_llm_response(
        self, response_text: str, state: SAOState
    ) -> tuple[ResponseType, Outcome | None, str | None, dict[str, Any] | None]:
        """Parse XML response into negotiation actions."""
        # Extract action
        action_match = re.search(r"<action>(\w+)</action>", response_text)
        action = action_match.group(1).lower() if action_match else "reject"

        response_type_map = {
            "accept": ResponseType.ACCEPT_OFFER,
            "reject": ResponseType.REJECT_OFFER,
            "end": ResponseType.END_NEGOTIATION,
        }
        response_type = response_type_map.get(action, ResponseType.REJECT_OFFER)

        # Extract offer
        outcome = None
        offer_match = re.search(r"<offer>([^<]+)</offer>", response_text)
        if offer_match:
            values = [v.strip() for v in offer_match.group(1).split(",")]
            # Convert to appropriate types based on outcome space
            outcome = tuple(self._convert_values(values))

        # Extract reasoning
        reasoning = None
        reasoning_match = re.search(r"<reasoning>([^<]+)</reasoning>", response_text)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return response_type, outcome, reasoning, None

    def _convert_values(self, values: list[str]) -> list[Any]:
        """Convert string values to appropriate types."""
        converted = []
        for v in values:
            try:
                # Try integer first
                converted.append(int(v))
            except ValueError:
                try:
                    # Then float
                    converted.append(float(v))
                except ValueError:
                    # Keep as string
                    converted.append(v)
        return converted
```

## Example: Domain-Specific Negotiator

Create a negotiator specialized for a specific domain:

```python
from negmas_llm import GeminiNegotiator
from negmas.sao import SAOState


class RealEstateNegotiator(GeminiNegotiator):
    """A negotiator specialized for real estate deals."""

    def __init__(self, property_type: str = "residential", **kwargs):
        super().__init__(**kwargs)
        self.property_type = property_type

    def build_system_prompt(self, state: SAOState) -> str:
        base_prompt = super().build_system_prompt(state)

        domain_context = f"""
## Domain Context: Real Estate ({self.property_type.title()})

You are negotiating a real estate transaction. Keep in mind:
- Market conditions and comparable sales
- Inspection contingencies and closing timelines
- Financing considerations
- Typical negotiation patterns in real estate

Be professional and build rapport while advocating for your interests.
"""
        return base_prompt + domain_context

    def format_own_ufun(self, state: SAOState) -> str:
        base_ufun = super().format_own_ufun(state)

        # Add domain-specific interpretation
        interpretation = """
### Interpretation for Real Estate

- **Price**: Lower is better for buyers, higher for sellers
- **Closing Date**: Flexibility may be valuable to either party
- **Contingencies**: More contingencies favor buyers, fewer favor sellers
"""
        return base_ufun + interpretation
```

## Example: Negotiator with Memory

Create a negotiator that maintains strategic memory across rounds:

```python
from negmas_llm import OpenAINegotiator
from negmas.sao import SAOState
from negmas.outcomes import Outcome


class StrategicMemoryNegotiator(OpenAINegotiator):
    """A negotiator that tracks patterns and adjusts strategy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.offer_history: list[tuple[str, Outcome | None, float | None]] = []
        self.concession_pattern: list[float] = []

    def build_user_message(
        self, state: SAOState, offer: Outcome | None, source: str | None
    ) -> str:
        base_message = super().build_user_message(state, offer, source)

        # Add strategic memory context
        memory_context = ["## Strategic Memory"]
        memory_context.append("")

        if self.offer_history:
            memory_context.append("### Offer History")
            for i, (who, hist_offer, utility) in enumerate(self.offer_history[-5:]):
                util_str = f" (utility: {utility:.3f})" if utility else ""
                memory_context.append(f"{i+1}. {who}: {hist_offer}{util_str}")
            memory_context.append("")

        if len(self.concession_pattern) >= 2:
            trend = self.concession_pattern[-1] - self.concession_pattern[-2]
            if trend > 0:
                memory_context.append(
                    "**Trend**: Opponent is making concessions (offers improving)"
                )
            elif trend < 0:
                memory_context.append(
                    "**Trend**: Opponent is hardening (offers getting worse)"
                )
            else:
                memory_context.append("**Trend**: Opponent is holding steady")
            memory_context.append("")

        # Track current offer
        if offer is not None:
            utility = self.ufun(offer) if self.ufun else None
            self.offer_history.append((source or "opponent", offer, utility))
            if utility is not None:
                self.concession_pattern.append(utility)

        return base_message + "\n".join(memory_context)

    def on_negotiation_start(self, state) -> None:
        super().on_negotiation_start(state)
        self.offer_history = []
        self.concession_pattern = []
```

## Tips for Customization

1. **Start Simple**: Begin with a custom system prompt before creating subclasses.

2. **Test Incrementally**: Override one method at a time and verify behavior.

3. **Preserve Base Behavior**: Call `super()` methods when you want to extend
   rather than replace functionality.

4. **Consider Token Limits**: Be mindful of context length when adding information
   to prompts.

5. **Handle Errors Gracefully**: The `_parse_llm_response` method should always
   return a valid response, even if parsing fails.

6. **Use Type Hints**: Maintain type hints in your subclasses for better IDE
   support and error catching.

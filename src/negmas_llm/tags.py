"""Tag processing system for LLM prompts.

This module provides a tag replacement system for dynamically inserting
negotiation context into prompts. Tags use the syntax:
    {{tag_name:format(param1=value1, param2=value2)}}

Where:
- tag_name: The tag identifier (e.g., "outcome-space", "utility")
- format: Optional output format ("text" or "json", defaults to "text")
- params: Optional comma-separated key=value pairs

Tags can be nested - parameter values can themselves be tags.

Example:
    "The outcome space is: {{outcome-space:json}}"
    "Your utility for their offer: {{utility:text(outcome={{opponent-last-offer}})}}"
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from negmas.inout import serialize

if TYPE_CHECKING:
    from negmas.negotiators import Negotiator
    from negmas.outcomes import Outcome
    from negmas.preferences import BaseUtilityFunction
    from negmas.sao import SAOState

__all__ = [
    "Tag",
    "TagFormat",
    "TagContext",
    "TagHandler",
    "process_prompt",
    "register_tag_handler",
    "get_tag_handler",
    "get_available_tags",
    "get_tag_documentation",
    "print_tag_help",
]


class Tag(str, Enum):
    """Supported tag identifiers.

    Tags are used to inject dynamic content into prompts based on
    the current negotiation context.
    """

    # Context tags (no parameters)
    OUTCOME_SPACE = "outcome-space"
    UTILITY_FUNCTION = "utility-function"
    OPPONENT_UTILITY_FUNCTION = "opponent-utility-function"
    NMI = "nmi"
    CURRENT_STATE = "current-state"

    # Reserved value tags
    RESERVED_VALUE = "reserved-value"
    OPPONENT_RESERVED_VALUE = "opponent-reserved-value"

    # Offer reference tags (no parameters)
    MY_LAST_OFFER = "my-last-offer"
    MY_FIRST_OFFER = "my-first-offer"
    OPPONENT_LAST_OFFER = "opponent-last-offer"
    OPPONENT_FIRST_OFFER = "opponent-first-offer"

    # History tags (optional k parameter)
    PARTNER_OFFERS = "partner-offers"
    HISTORY = "history"
    TRACE = "trace"
    EXTENDED_TRACE = "extended-trace"
    FULL_TRACE = "full-trace"

    # Utility computation tag (requires outcome parameter)
    UTILITY = "utility"

    @classmethod
    def available_tags(cls) -> dict[str, str]:
        """Get documentation for all available tags.

        Returns a dictionary mapping each tag name to its markdown-formatted
        documentation, including description, parameters, and usage examples.

        Returns:
            Dict mapping tag names to markdown documentation strings.

        Example:
            >>> docs = Tag.available_tags()
            >>> print(docs["outcome-space"])
        """
        return {
            "outcome-space": """\
### `{{outcome-space}}`

Returns the negotiation outcome space definition.

**Parameters:** None

**Formats:**
- `{{outcome-space}}` or `{{outcome-space:text}}` - Human-readable format
- `{{outcome-space:json}}` - JSON format with full structure

**Description:**
The outcome space defines all possible agreements in the negotiation,
including the issues/dimensions and their possible values.

**Example output (text):**
```
Outcome Space:
  - price: [10, 20, 30, 40, 50]
  - quantity: [1, 2, 3, 4, 5]
  - delivery: ['fast', 'standard', 'slow']
```
""",
            "utility-function": """\
### `{{utility-function}}`

Returns your utility function definition.

**Parameters:** None

**Formats:**
- `{{utility-function}}` or `{{utility-function:text}}` - Human-readable format
- `{{utility-function:json}}` - JSON format with full specification

**Description:**
Your utility function maps outcomes to real numbers representing your preferences.
Higher values indicate more preferred outcomes. The reserved value is included,
which represents the utility of no agreement (your walk-away point).

**Example output (text):**
```
Utility function: LinearAdditiveUtilityFunction(...)
Reserved value: 0.3
```
""",
            "opponent-utility-function": """\
### `{{opponent-utility-function}}`

Returns the opponent's utility function (if known).

**Parameters:** None

**Formats:**
- `{{opponent-utility-function}}` or `{{opponent-utility-function:text}}`
  Human-readable format
- `{{opponent-utility-function:json}}` - JSON format

**Description:**
If the opponent's utility function is available (from private_info["opponent_ufun"]),
returns its specification. Otherwise returns "Opponent utility function unknown".

**Example output (text):**
```
Opponent utility function: LinearAdditiveUtilityFunction(...)
Opponent reserved value: 0.25
```
""",
            "nmi": """\
### `{{nmi}}`

Returns the Negotiator Mechanism Interface information.

**Parameters:** None

**Formats:**
- `{{nmi}}` or `{{nmi:text}}` - Human-readable format
- `{{nmi:json}}` - JSON format

**Description:**
The NMI provides metadata about the negotiation mechanism:
- `n_steps`: Maximum negotiation steps (None = unlimited)
- `time_limit`: Maximum time in seconds (None = unlimited)
- `n_outcomes`: Total possible outcomes
- `n_negotiators`: Number of participants
- `end_on_no_response`: Whether negotiation ends if any party returns None
- `one_offer_per_step`: Whether only one negotiator acts per step
- `offering_is_accepting`: Whether making an offer implies accepting it if echoed

**Example output (text):**
```
Negotiation Mechanism:
  N Steps: 100
  Time Limit: 60.0
  N Outcomes: 125
  N Negotiators: 2
```
""",
            "current-state": """\
### `{{current-state}}`

Returns the current negotiation state.

**Parameters:** None

**Formats:**
- `{{current-state}}` or `{{current-state:text}}` - Human-readable format
- `{{current-state:json}}` - JSON format

**Description:**
Returns the current state of the negotiation including:
- `step`: Current step number
- `relative_time`: Progress (0.0 to 1.0)
- `running`: Whether negotiation is active
- `current_offer`: The offer on the table
- `current_proposer`: Who made the current offer
- `n_acceptances`: Acceptances for current offer
- Status flags (broken, timedout, agreement)

**Example output (text):**
```
Current State:
  Step: 5
  Relative Time: 25.00%
  Running: True
  Current Offer: {price=30, quantity=3}
  Proposer: opponent_1
  Acceptances: 0
```
""",
            "reserved-value": """\
### `{{reserved-value}}`

Returns your reserved value (walk-away point).

**Parameters:** None

**Formats:**
- `{{reserved-value}}` or `{{reserved-value:text}}` - Numeric value
- `{{reserved-value:json}}` - JSON object with reserved_value key

**Description:**
The reserved value is the utility you receive if negotiation fails
(no agreement reached). You should aim for outcomes with utility
above this value.

**Example output:** `0.35`
""",
            "opponent-reserved-value": """\
### `{{opponent-reserved-value}}`

Returns the opponent's reserved value (if known).

**Parameters:** None

**Formats:**
- `{{opponent-reserved-value}}` or `{{opponent-reserved-value:text}}` - Numeric value
- `{{opponent-reserved-value:json}}` - JSON object

**Description:**
If the opponent's utility function is known and has a reserved value,
returns it. Otherwise returns "Opponent reserved value unknown".

**Example output:** `0.25`
""",
            "my-last-offer": """\
### `{{my-last-offer}}`

Returns the last offer you made.

**Parameters:** None

**Formats:**
- `{{my-last-offer}}` or `{{my-last-offer:text}}` - Formatted outcome
- `{{my-last-offer:json}}` - JSON array

**Description:**
Returns your most recent offer from the negotiation trace.
Returns "None" if you haven't made any offers yet.

**Example output (text):** `{price=35, quantity=2, delivery=fast}`
**Example output (json):** `[35, 2, "fast"]`
""",
            "my-first-offer": """\
### `{{my-first-offer}}`

Returns the first offer you made.

**Parameters:** None

**Formats:**
- `{{my-first-offer}}` or `{{my-first-offer:text}}` - Formatted outcome
- `{{my-first-offer:json}}` - JSON array

**Description:**
Returns your first offer from the negotiation trace.
Returns "None" if you haven't made any offers yet.

**Example output (text):** `{price=50, quantity=1, delivery=fast}`
""",
            "opponent-last-offer": """\
### `{{opponent-last-offer}}`

Returns the opponent's last offer.

**Parameters:** None

**Formats:**
- `{{opponent-last-offer}}` or `{{opponent-last-offer:text}}` - Formatted outcome
- `{{opponent-last-offer:json}}` - JSON array

**Description:**
Returns the opponent's most recent offer from the negotiation trace.
Returns "None" if opponent hasn't made any offers yet.

**Example output (text):** `{price=20, quantity=4, delivery=slow}`
""",
            "opponent-first-offer": """\
### `{{opponent-first-offer}}`

Returns the opponent's first offer.

**Parameters:** None

**Formats:**
- `{{opponent-first-offer}}` or `{{opponent-first-offer:text}}` - Formatted outcome
- `{{opponent-first-offer:json}}` - JSON array

**Description:**
Returns the opponent's first offer from the negotiation trace.
Returns "None" if opponent hasn't made any offers yet.
""",
            "partner-offers": """\
### `{{partner-offers}}`

Returns a list of offers made by the opponent/partner.

**Parameters:**
- `k` (optional, int): Limit to the last k offers. Default: 0 (all offers)

**Formats:**
- `{{partner-offers}}` or `{{partner-offers:text}}` - Numbered list
- `{{partner-offers:json}}` - JSON array of arrays
- `{{partner-offers:text(k=5)}}` - Last 5 offers

**Description:**
Returns all offers made by opponents during the negotiation.
Use the `k` parameter to limit to recent offers.

**Example output (text):**
```
Partner offers:
  1. {price=15, quantity=5}
  2. {price=20, quantity=4}
  3. {price=22, quantity=3}
```
""",
            "history": """\
### `{{history}}`

Returns the negotiation history.

**Parameters:**
- `k` (optional, int): Limit to the last k entries. Default: 0 (all)

**Formats:**
- `{{history}}` or `{{history:text}}` - Human-readable list
- `{{history:json}}` - JSON array of objects
- `{{history:text(k=10)}}` - Last 10 entries

**Description:**
Returns the sequence of offers made during negotiation,
labeled by who made each offer (You or Opponent).

**Example output (text):**
```
Negotiation history:
  1. You: [50, 1, 'fast']
  2. Opponent: [20, 5, 'slow']
  3. You: [40, 2, 'fast']
```
""",
            "trace": """\
### `{{trace}}`

Returns the basic negotiation trace.

**Parameters:**
- `k` (optional, int): Limit to the last k entries. Default: 0 (all)

**Formats:**
- `{{trace}}` or `{{trace:text}}` - List with negotiator IDs
- `{{trace:json}}` - JSON array of [id, outcome] pairs

**Description:**
Returns the raw negotiation trace as (negotiator_id, outcome) pairs.
Lower-level than {{history}} which labels entries as You/Opponent.

**Example output (text):**
```
Trace:
  1. negotiator_0: (50, 1, 'fast')
  2. negotiator_1: (20, 5, 'slow')
```
""",
            "extended-trace": """\
### `{{extended-trace}}`

Returns the extended negotiation trace with response types.

**Parameters:**
- `k` (optional, int): Limit to the last k entries. Default: 0 (all)

**Formats:**
- `{{extended-trace}}` or `{{extended-trace:text}}` - Detailed list
- `{{extended-trace:json}}` - JSON array with step, response_type, outcome

**Description:**
Returns detailed trace including step numbers and response types
(ACCEPT, REJECT, etc.) in addition to outcomes.

**Example output (text):**
```
Extended trace:
  1. Step 0: REJECT_OFFER - (50, 1, 'fast')
  2. Step 1: REJECT_OFFER - (20, 5, 'slow')
```
""",
            "full-trace": """\
### `{{full-trace}}`

Returns the complete negotiation trace with all details.

**Parameters:**
- `k` (optional, int): Limit to the last k entries. Default: 0 (all)

**Formats:**
- `{{full-trace}}` or `{{full-trace:text}}` - Complete details
- `{{full-trace:json}}` - JSON array with all fields

**Description:**
Combines information from trace and extended trace, labeling
entries as You/Opponent and including response types.

**Example output (text):**
```
Full negotiation trace:
  Step 0: You - REJECT_OFFER: [50, 1, 'fast']
  Step 1: Opponent - REJECT_OFFER: [20, 5, 'slow']
```
""",
            "utility": """\
### `{{utility}}`

Computes the utility of a specific outcome.

**Parameters:**
- `outcome` (required): The outcome to evaluate. Can be:
  - A tuple/list: `(30, 2, 'fast')` or `[30, 2, "fast"]`
  - A nested tag: `{{opponent-last-offer}}`
  - A formatted string: `{price=30, quantity=2}`

**Formats:**
- `{{utility:text(outcome=...)}}` - Just the utility value
- `{{utility:json(outcome=...)}}` - JSON with outcome and utility

**Description:**
Evaluates the given outcome using your utility function.
Useful for computing utilities of specific offers or nested tags.

**Examples:**
- `{{utility:text(outcome=(30, 2, 'fast'))}}` → `0.72`
- `{{utility:text(outcome={{opponent-last-offer}})}}` → `0.45`
- `{{utility:json(outcome=[25, 3, "standard"])}}`
  → `{"outcome": [25, 3, "standard"], "utility": 0.58}`

**Note:** Returns an error message if outcome cannot be parsed.
""",
        }


class TagFormat(str, Enum):
    """Output format for tag values."""

    TEXT = "text"
    JSON = "json"


class TagContext:
    """Context object passed to tag handlers.

    This provides access to the negotiator, current state, and helper
    methods for formatting output.

    Attributes:
        negotiator: The negotiator instance.
        state: The current SAO state (may be None if not in negotiation).
        format: The requested output format.
        params: Parameters passed to the tag.
    """

    def __init__(
        self,
        negotiator: Negotiator,
        state: SAOState | None,
        format: TagFormat,
        params: dict[str, str],
    ) -> None:
        self.negotiator = negotiator
        self.state = state
        self.format = format
        self.params = params

    def format_outcome(self, outcome: Outcome | None) -> str:
        """Format an outcome according to the requested format.

        Args:
            outcome: The outcome to format.

        Returns:
            Formatted string representation.
        """
        if outcome is None:
            return "None" if self.format == TagFormat.TEXT else "null"

        if self.format == TagFormat.JSON:
            return json.dumps(list(outcome))

        # Text format - try to include issue names
        if self.negotiator.nmi is not None and self.negotiator.nmi.outcome_space:
            try:
                issues = self.negotiator.nmi.outcome_space.issues  # type: ignore
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

    def format_dict(self, data: dict[str, Any]) -> str:
        """Format a dictionary according to the requested format.

        Args:
            data: Dictionary to format.

        Returns:
            Formatted string.
        """
        if self.format == TagFormat.JSON:
            return json.dumps(data, indent=2, default=str)
        else:
            # Text format - simple key: value pairs
            lines = []
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)

    def get_k_param(self, default: int = 0) -> int:
        """Get the 'k' parameter as an integer.

        Args:
            default: Default value if not specified or invalid.

        Returns:
            The k parameter value.
        """
        k_str = self.params.get("k", str(default))
        try:
            return int(k_str)
        except ValueError:
            return default


# Type alias for tag handler functions
TagHandler = Callable[[TagContext], str]

# Global registry of tag handlers
_tag_handlers: dict[str, TagHandler] = {}


def register_tag_handler(tag: Tag | str, handler: TagHandler) -> None:
    """Register a handler for a tag.

    Args:
        tag: The tag to handle (Tag enum or string).
        handler: Callable that takes TagContext and returns a string.
    """
    tag_name = tag.value if isinstance(tag, Tag) else tag
    _tag_handlers[tag_name] = handler


def get_tag_handler(tag_name: str) -> TagHandler | None:
    """Get the handler for a tag.

    Args:
        tag_name: The tag name to look up.

    Returns:
        The handler function or None if not found.
    """
    return _tag_handlers.get(tag_name)


# =============================================================================
# Tag parsing and processing
# =============================================================================

# Regex pattern for tags: {{tag_name:format(params)}}
# - tag_name: alphanumeric with hyphens/underscores
# - :format is optional
# - (params) is optional
# We use a non-greedy pattern and handle nested parens separately
_PARAMS_PATTERN = r"(\([^)]*(?:\([^)]*\)[^)]*)*\))?"  # (params) with one nesting
_TAG_PATTERN = re.compile(
    r"\{\{"  # Opening {{
    r"([a-zA-Z][a-zA-Z0-9_-]*)"  # Tag name (group 1)
    r"(?::([a-zA-Z]+))?"  # Optional :format (group 2)
    + _PARAMS_PATTERN  # Optional (params) (group 3)
    + r"\}\}"  # Closing }}
)

# More robust pattern for complex nested parentheses - used as fallback
_TAG_START_PATTERN = re.compile(r"\{\{([a-zA-Z][a-zA-Z0-9_-]*)(?::([a-zA-Z]+))?")


def _find_matching_paren(s: str, start: int) -> int:
    """Find the index of the closing paren that matches the one at start.

    Uses a simple counter approach with string literal handling.

    Args:
        s: The string to search in.
        start: Index of the opening paren.

    Returns:
        Index of the matching closing paren, or -1 if not found.
    """
    if start >= len(s) or s[start] != "(":
        return -1

    depth = 1
    i = start + 1
    in_string = False
    string_char = None

    while i < len(s) and depth > 0:
        char = s[i]

        # Handle string literals
        if char in ('"', "'") and (i == 0 or s[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

        i += 1

    return i - 1 if depth == 0 else -1


def _find_tags(text: str) -> list[tuple[int, int, str, str | None, str | None]]:
    """Find all tags in the text with their positions.

    Uses regex for fast initial matching, with fallback to manual parsing
    for complex nested parentheses in parameters.

    Returns:
        List of (start, end, tag_name, format, params) tuples.
    """
    tags: list[tuple[int, int, str, str | None, str | None]] = []

    # Try fast regex matching first for simple cases
    for match in _TAG_PATTERN.finditer(text):
        tag_name = match.group(1)
        tag_format = match.group(2)
        params_with_parens = match.group(3)

        # Extract params without outer parentheses
        params = None
        if params_with_parens:
            params = params_with_parens[1:-1]  # Strip ( and )

        tags.append((match.start(), match.end(), tag_name, tag_format, params))

    # If we found tags, return them
    if tags:
        return tags

    # Fallback: manual parsing for complex cases (deeply nested parens)
    pos = 0
    while pos < len(text) - 4:  # Need at least {{x}}
        start_match = _TAG_START_PATTERN.search(text, pos)
        if not start_match:
            break

        tag_start = start_match.start()
        tag_name = start_match.group(1)
        tag_format = start_match.group(2)
        end_pos = start_match.end()

        # Check for params
        params = None
        if end_pos < len(text) and text[end_pos] == "(":
            paren_end = _find_matching_paren(text, end_pos)
            if paren_end > 0:
                params = text[end_pos + 1 : paren_end]
                end_pos = paren_end + 1

        # Check for closing }}
        if end_pos < len(text) - 1 and text[end_pos : end_pos + 2] == "}}":
            tags.append((tag_start, end_pos + 2, tag_name, tag_format, params))
            pos = end_pos + 2
        else:
            pos = tag_start + 1

    return tags


def _parse_params(param_str: str | None) -> dict[str, str]:
    """Parse parameter string into a dictionary.

    Args:
        param_str: Comma-separated key=value pairs.

    Returns:
        Dictionary of parameter names to values.
    """
    if not param_str:
        return {}

    params: dict[str, str] = {}
    # Split by comma, but be careful of nested tags and parentheses
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    in_string = False
    string_char = None
    current = ""

    for i, char in enumerate(param_str):
        # Handle string literals
        if char in ('"', "'") and (i == 0 or param_str[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            current += char
        elif in_string:
            current += char
        elif char == "{":
            brace_depth += 1
            current += char
        elif char == "}":
            brace_depth -= 1
            current += char
        elif char == "(":
            paren_depth += 1
            current += char
        elif char == ")":
            paren_depth -= 1
            current += char
        elif char == "[":
            bracket_depth += 1
            current += char
        elif char == "]":
            bracket_depth -= 1
            current += char
        elif (
            char == "," and brace_depth == 0 and paren_depth == 0 and bracket_depth == 0
        ):
            if "=" in current:
                key, _, value = current.partition("=")
                params[key.strip()] = value.strip()
            current = ""
        else:
            current += char

    # Handle last parameter
    if current and "=" in current:
        key, _, value = current.partition("=")
        params[key.strip()] = value.strip()

    return params


def _process_single_tag(
    tag_name: str,
    format_str: str | None,
    param_str: str | None,
    original_text: str,
    negotiator: Negotiator,
    state: SAOState | None,
) -> str:
    """Process a single tag.

    Args:
        tag_name: The tag name.
        format_str: The format string (e.g., "json", "text").
        param_str: The parameter string.
        original_text: The original tag text (for returning unchanged).
        negotiator: The negotiator instance.
        state: The current state.

    Returns:
        The replacement string.
    """
    # Parse format
    try:
        tag_format = TagFormat((format_str or "text").lower())
    except ValueError:
        tag_format = TagFormat.TEXT

    # Parse parameters
    params = _parse_params(param_str)

    # Recursively process any tags in parameter values
    for key, value in params.items():
        if "{{" in value and "}}" in value:
            params[key] = process_prompt(value, negotiator, state)

    # Get handler
    handler = get_tag_handler(tag_name)
    if handler is None:
        # Unknown tag - return as-is
        return original_text

    # Create context and call handler
    context = TagContext(negotiator, state, tag_format, params)
    try:
        return handler(context)
    except Exception as e:
        # On error, return a descriptive message
        return f"[Error processing {tag_name}: {e}]"


def process_prompt(
    prompt: str,
    negotiator: Negotiator,
    state: SAOState | None = None,
) -> str:
    """Process a prompt, replacing all tags with their values.

    This function finds all tags in the prompt and replaces them with
    dynamically generated content based on the current negotiation context.

    Tags are processed from innermost to outermost to support nesting.

    Escape sequences:
    - Use \\{{ to output a literal {{
    - Use \\}} to output a literal }}

    Args:
        prompt: The prompt string containing tags.
        negotiator: The negotiator instance.
        state: The current SAO state (optional).

    Returns:
        The prompt with all tags replaced.

    Example:
        >>> prompt = "Outcome space: {{outcome-space:json}}"
        >>> processed = process_prompt(prompt, negotiator, state)
        >>> # Use \\{{ and \\}} for literal braces:
        >>> prompt = "JSON example: \\{{\\"key\\": \\"value\\"\\}}"
        >>> # Results in: JSON example: {{"key": "value"}}
    """
    # Temporarily replace escape sequences with placeholders
    _ESCAPE_OPEN = "\x00ESCAPE_OPEN\x00"
    _ESCAPE_CLOSE = "\x00ESCAPE_CLOSE\x00"
    prompt = prompt.replace("\\{{", _ESCAPE_OPEN)
    prompt = prompt.replace("\\}}", _ESCAPE_CLOSE)

    # Process iteratively until no more tags are found
    # This handles nested tags by processing innermost first
    max_iterations = 100  # Prevent infinite loops
    iteration = 0

    while "{{" in prompt and "}}" in prompt and iteration < max_iterations:
        tags = _find_tags(prompt)
        if not tags:
            break

        # Process tags in reverse order to preserve positions
        new_prompt = prompt
        for start, end, tag_name, format_str, param_str in reversed(tags):
            original_text = prompt[start:end]
            replacement = _process_single_tag(
                tag_name,
                format_str,
                param_str,
                original_text,
                negotiator,
                state,
            )
            new_prompt = new_prompt[:start] + replacement + new_prompt[end:]

        if new_prompt == prompt:
            # No changes made, stop processing
            break
        prompt = new_prompt
        iteration += 1

    # Restore escaped sequences to literal braces
    prompt = prompt.replace(_ESCAPE_OPEN, "{{")
    prompt = prompt.replace(_ESCAPE_CLOSE, "}}")

    return prompt


# =============================================================================
# Built-in tag handlers
# =============================================================================


def _handle_outcome_space(ctx: TagContext) -> str:
    """Handle the outcome-space tag."""
    if ctx.negotiator.nmi is None or ctx.negotiator.nmi.outcome_space is None:
        return "No outcome space available"

    outcome_space = ctx.negotiator.nmi.outcome_space

    if ctx.format == TagFormat.JSON:
        try:
            os_dict = serialize(outcome_space)
            os_dict.pop("__python_class__", None)
            return json.dumps(os_dict, indent=2, default=str)
        except Exception:
            return json.dumps({"error": "Could not serialize outcome space"})
    else:
        # Text format
        try:
            os_dict = serialize(outcome_space)
            os_dict.pop("__python_class__", None)
            lines = ["Outcome Space:"]
            if "issues" in os_dict:
                for issue in os_dict["issues"]:
                    name = issue.get("name", "unnamed")
                    values = issue.get("values", [])
                    lines.append(f"  - {name}: {values}")
            else:
                lines.append(f"  {outcome_space}")
            return "\n".join(lines)
        except Exception:
            return str(outcome_space)


def _handle_utility_function(ctx: TagContext) -> str:
    """Handle the utility-function tag."""
    if ctx.negotiator.ufun is None:
        return "No utility function available"

    ufun = ctx.negotiator.ufun

    if ctx.format == TagFormat.JSON:
        try:
            ufun_dict = serialize(ufun)
            ufun_dict.pop("__python_class__", None)
            return json.dumps(ufun_dict, indent=2, default=str)
        except Exception:
            return json.dumps({"description": str(ufun)})
    else:
        # Text format
        reserved = ctx.negotiator.reserved_value
        return f"Utility function: {ufun}\nReserved value: {reserved}"


def _handle_opponent_utility_function(ctx: TagContext) -> str:
    """Handle the opponent-utility-function tag."""
    opponent_ufun: BaseUtilityFunction | None = None
    if ctx.negotiator.private_info:
        opponent_ufun = ctx.negotiator.private_info.get("opponent_ufun")

    if opponent_ufun is None:
        return "Opponent utility function unknown"

    if ctx.format == TagFormat.JSON:
        try:
            ufun_dict = serialize(opponent_ufun)
            ufun_dict.pop("__python_class__", None)
            return json.dumps(ufun_dict, indent=2, default=str)
        except Exception:
            return json.dumps({"description": str(opponent_ufun)})
    else:
        reserved = getattr(opponent_ufun, "reserved_value", None)
        text = f"Opponent utility function: {opponent_ufun}"
        if reserved is not None:
            text += f"\nOpponent reserved value: {reserved}"
        return text


def _get_my_offers(ctx: TagContext) -> list[Outcome]:
    """Get list of offers made by this negotiator."""
    if ctx.state is None:
        return []

    my_id = ctx.negotiator.id
    offers: list[Outcome] = []

    # Get from extended trace if available
    if hasattr(ctx.state, "extended_trace") and ctx.state.extended_trace:
        for item in ctx.state.extended_trace:
            if hasattr(item, "response") and item.response:
                resp = item.response
                if hasattr(resp, "outcome") and resp.outcome is not None:
                    source = getattr(resp, "source", None) or getattr(
                        item, "source", None
                    )
                    if source == my_id:
                        offers.append(resp.outcome)

    # Fallback to trace
    if not offers and hasattr(ctx.state, "trace") and ctx.state.trace:
        # Trace is a list of (negotiator_id, outcome) tuples
        for proposer_id, outcome in ctx.state.trace:
            if proposer_id == my_id and outcome is not None:
                offers.append(outcome)

    return offers


def _get_opponent_offers(ctx: TagContext) -> list[Outcome]:
    """Get list of offers made by opponents."""
    if ctx.state is None:
        return []

    my_id = ctx.negotiator.id
    offers: list[Outcome] = []

    # Get from extended trace if available
    if hasattr(ctx.state, "extended_trace") and ctx.state.extended_trace:
        for item in ctx.state.extended_trace:
            if hasattr(item, "response") and item.response:
                resp = item.response
                if hasattr(resp, "outcome") and resp.outcome is not None:
                    source = getattr(resp, "source", None) or getattr(
                        item, "source", None
                    )
                    if source and source != my_id:
                        offers.append(resp.outcome)

    # Fallback to trace
    if not offers and hasattr(ctx.state, "trace") and ctx.state.trace:
        for proposer_id, outcome in ctx.state.trace:
            if proposer_id != my_id and outcome is not None:
                offers.append(outcome)

    return offers


def _handle_my_last_offer(ctx: TagContext) -> str:
    """Handle the my-last-offer tag."""
    offers = _get_my_offers(ctx)
    if offers:
        return ctx.format_outcome(offers[-1])
    return ctx.format_outcome(None)


def _handle_my_first_offer(ctx: TagContext) -> str:
    """Handle the my-first-offer tag."""
    offers = _get_my_offers(ctx)
    if offers:
        return ctx.format_outcome(offers[0])
    return ctx.format_outcome(None)


def _handle_opponent_last_offer(ctx: TagContext) -> str:
    """Handle the opponent-last-offer tag."""
    offers = _get_opponent_offers(ctx)
    if offers:
        return ctx.format_outcome(offers[-1])
    return ctx.format_outcome(None)


def _handle_opponent_first_offer(ctx: TagContext) -> str:
    """Handle the opponent-first-offer tag."""
    offers = _get_opponent_offers(ctx)
    if offers:
        return ctx.format_outcome(offers[0])
    return ctx.format_outcome(None)


def _handle_partner_offers(ctx: TagContext) -> str:
    """Handle the partner-offers tag (alias for opponent offers)."""
    offers = _get_opponent_offers(ctx)
    k = ctx.get_k_param(0)

    if k > 0:
        offers = offers[-k:]

    if ctx.format == TagFormat.JSON:
        return json.dumps([list(o) for o in offers])
    else:
        if not offers:
            return "No partner offers yet"
        lines = ["Partner offers:"]
        for i, offer in enumerate(offers, 1):
            lines.append(f"  {i}. {ctx.format_outcome(offer)}")
        return "\n".join(lines)


def _handle_history(ctx: TagContext) -> str:
    """Handle the history tag - shows negotiation history."""
    if ctx.state is None:
        return "No history available"

    k = ctx.get_k_param(0)
    my_id = ctx.negotiator.id

    history: list[dict[str, Any]] = []

    # Try trace first
    if hasattr(ctx.state, "trace") and ctx.state.trace:
        for proposer_id, outcome in ctx.state.trace:
            who = "You" if proposer_id == my_id else "Opponent"
            history.append(
                {
                    "proposer": who,
                    "offer": list(outcome) if outcome else None,
                }
            )

    if k > 0:
        history = history[-k:]

    if ctx.format == TagFormat.JSON:
        return json.dumps(history, indent=2)
    else:
        if not history:
            return "No negotiation history yet"
        lines = ["Negotiation history:"]
        for i, item in enumerate(history, 1):
            proposer = item["proposer"]
            offer = item["offer"]
            offer_str = str(offer) if offer else "None"
            lines.append(f"  {i}. {proposer}: {offer_str}")
        return "\n".join(lines)


def _handle_trace(ctx: TagContext) -> str:
    """Handle the trace tag - shows basic trace."""
    if ctx.state is None or not hasattr(ctx.state, "trace"):
        return "No trace available"

    k = ctx.get_k_param(0)
    trace = list(ctx.state.trace) if ctx.state.trace else []

    if k > 0:
        trace = trace[-k:]

    if ctx.format == TagFormat.JSON:
        return json.dumps([(pid, list(o) if o else None) for pid, o in trace])
    else:
        if not trace:
            return "Empty trace"
        lines = ["Trace:"]
        for i, (pid, outcome) in enumerate(trace, 1):
            outcome_str = str(outcome) if outcome else "None"
            lines.append(f"  {i}. {pid}: {outcome_str}")
        return "\n".join(lines)


def _handle_extended_trace(ctx: TagContext) -> str:
    """Handle the extended-trace tag."""
    if ctx.state is None or not hasattr(ctx.state, "extended_trace"):
        return "No extended trace available"

    k = ctx.get_k_param(0)
    trace = list(ctx.state.extended_trace) if ctx.state.extended_trace else []

    if k > 0:
        trace = trace[-k:]

    if ctx.format == TagFormat.JSON:
        # Serialize extended trace items
        items = []
        for item in trace:
            item_dict: dict[str, Any] = {}
            if hasattr(item, "step"):
                item_dict["step"] = item.step
            if hasattr(item, "response"):
                resp = item.response
                if resp:
                    item_dict["response_type"] = (
                        str(resp.response) if hasattr(resp, "response") else None
                    )
                    item_dict["outcome"] = (
                        list(resp.outcome)
                        if hasattr(resp, "outcome") and resp.outcome
                        else None
                    )
            items.append(item_dict)
        return json.dumps(items, indent=2)
    else:
        if not trace:
            return "Empty extended trace"
        lines = ["Extended trace:"]
        for i, item in enumerate(trace, 1):
            step = getattr(item, "step", "?")
            resp = getattr(item, "response", None)
            if resp:
                resp_type = getattr(resp, "response", "?")
                outcome = getattr(resp, "outcome", None)
                lines.append(f"  {i}. Step {step}: {resp_type} - {outcome}")
            else:
                lines.append(f"  {i}. Step {step}: no response")
        return "\n".join(lines)


def _handle_full_trace(ctx: TagContext) -> str:
    """Handle the full-trace tag - combines all trace information."""
    if ctx.state is None:
        return "No trace available"

    k = ctx.get_k_param(0)
    my_id = ctx.negotiator.id

    entries: list[dict[str, Any]] = []

    # Try extended trace first for most detail
    if hasattr(ctx.state, "extended_trace") and ctx.state.extended_trace:
        for item in ctx.state.extended_trace:
            entry: dict[str, Any] = {"step": getattr(item, "step", None)}
            if hasattr(item, "response") and item.response:
                resp = item.response
                entry["response_type"] = str(getattr(resp, "response", None))
                entry["outcome"] = (
                    list(resp.outcome) if getattr(resp, "outcome", None) else None
                )
                source = getattr(resp, "source", None) or getattr(item, "source", None)
                entry["who"] = "You" if source == my_id else "Opponent"
            entries.append(entry)
    elif hasattr(ctx.state, "trace") and ctx.state.trace:
        for i, (pid, outcome) in enumerate(ctx.state.trace):
            entries.append(
                {
                    "step": i,
                    "who": "You" if pid == my_id else "Opponent",
                    "outcome": list(outcome) if outcome else None,
                }
            )

    if k > 0:
        entries = entries[-k:]

    if ctx.format == TagFormat.JSON:
        return json.dumps(entries, indent=2)
    else:
        if not entries:
            return "No trace entries"
        lines = ["Full negotiation trace:"]
        for entry in entries:
            step = entry.get("step", "?")
            who = entry.get("who", "?")
            outcome = entry.get("outcome")
            resp_type = entry.get("response_type", "offer")
            lines.append(f"  Step {step}: {who} - {resp_type}: {outcome}")
        return "\n".join(lines)


def _handle_utility(ctx: TagContext) -> str:
    """Handle the utility tag - compute utility for an outcome."""
    if ctx.negotiator.ufun is None:
        return "No utility function available"

    # Get the outcome parameter
    outcome_str = ctx.params.get("outcome")
    if not outcome_str:
        return "Error: 'outcome' parameter required"

    # Try to parse the outcome
    outcome: Outcome | None = None

    # Handle outcome tag results (already processed by recursive tag processing)
    if outcome_str in ("None", "null"):
        return "Cannot compute utility for None outcome"

    # Try to eval as a tuple/list
    try:
        parsed = eval(outcome_str)  # noqa: S307
        if isinstance(parsed, (list, tuple)):
            outcome = tuple(parsed)
    except Exception:
        # If eval fails, the string might be a formatted outcome like "{a=1, b=2}"
        # Try to extract values
        if outcome_str.startswith("{") and outcome_str.endswith("}"):
            # Parse "{key=value, key=value}" format
            inner = outcome_str[1:-1]
            values = []
            for part in inner.split(","):
                if "=" in part:
                    _, _, val = part.partition("=")
                    val = val.strip()
                    try:
                        values.append(eval(val))  # noqa: S307
                    except Exception:
                        values.append(val)
            if values:
                outcome = tuple(values)

    if outcome is None:
        return f"Could not parse outcome: {outcome_str}"

    try:
        utility = ctx.negotiator.ufun(outcome)
        if ctx.format == TagFormat.JSON:
            return json.dumps({"outcome": list(outcome), "utility": utility})
        else:
            return str(utility)
    except Exception as e:
        return f"Error computing utility: {e}"


def _handle_reserved_value(ctx: TagContext) -> str:
    """Handle the reserved-value tag."""
    reserved = ctx.negotiator.reserved_value

    if ctx.format == TagFormat.JSON:
        return json.dumps({"reserved_value": reserved})
    else:
        if reserved is None:
            return "No reserved value set"
        return str(reserved)


def _handle_opponent_reserved_value(ctx: TagContext) -> str:
    """Handle the opponent-reserved-value tag."""
    opponent_ufun: BaseUtilityFunction | None = None
    if ctx.negotiator.private_info:
        opponent_ufun = ctx.negotiator.private_info.get("opponent_ufun")

    if opponent_ufun is None:
        return "Opponent reserved value unknown"

    reserved = getattr(opponent_ufun, "reserved_value", None)

    if ctx.format == TagFormat.JSON:
        return json.dumps({"opponent_reserved_value": reserved})
    else:
        if reserved is None:
            return "Opponent reserved value unknown"
        return str(reserved)


def _handle_nmi(ctx: TagContext) -> str:
    """Handle the nmi tag - returns negotiator mechanism interface info."""
    nmi = ctx.negotiator.nmi
    if nmi is None:
        return "No NMI available (not in a negotiation)"

    info: dict[str, Any] = {}

    # Core mechanism info
    info["n_steps"] = nmi.n_steps
    info["time_limit"] = nmi.time_limit
    info["n_outcomes"] = nmi.n_outcomes
    info["n_negotiators"] = nmi.n_negotiators

    # SAO-specific attributes
    if hasattr(nmi, "end_on_no_response"):
        info["end_on_no_response"] = nmi.end_on_no_response
    if hasattr(nmi, "one_offer_per_step"):
        info["one_offer_per_step"] = nmi.one_offer_per_step
    if hasattr(nmi, "offering_is_accepting"):
        info["offering_is_accepting"] = nmi.offering_is_accepting

    # Dynamic entry
    if hasattr(nmi, "dynamic_entry"):
        info["dynamic_entry"] = nmi.dynamic_entry

    if ctx.format == TagFormat.JSON:
        return json.dumps(info, indent=2, default=str)
    else:
        lines = ["Negotiation Mechanism:"]
        for key, value in info.items():
            if value is not None:
                # Format key as human-readable
                formatted_key = key.replace("_", " ").title()
                lines.append(f"  {formatted_key}: {value}")
        return "\n".join(lines)


def _handle_current_state(ctx: TagContext) -> str:
    """Handle the current-state tag - returns current negotiation state."""
    state = ctx.state
    if state is None:
        return "No state available"

    info: dict[str, Any] = {}

    # Core state info
    info["step"] = state.step
    info["relative_time"] = state.relative_time
    info["running"] = state.running

    # Current offer
    if state.current_offer is not None:
        info["current_offer"] = list(state.current_offer)
    else:
        info["current_offer"] = None

    info["current_proposer"] = state.current_proposer
    info["n_acceptances"] = state.n_acceptances

    # Status flags
    if hasattr(state, "broken") and state.broken:
        info["broken"] = True
    if hasattr(state, "timedout") and state.timedout:
        info["timedout"] = True
    if hasattr(state, "agreement") and state.agreement is not None:
        info["agreement"] = list(state.agreement)

    # New offers this step
    if hasattr(state, "new_offers") and state.new_offers:
        info["new_offers"] = [
            {"proposer": pid, "offer": list(o) if o else None}
            for pid, o in state.new_offers
        ]

    if ctx.format == TagFormat.JSON:
        return json.dumps(info, indent=2, default=str)
    else:
        lines = ["Current State:"]
        lines.append(f"  Step: {info['step']}")
        lines.append(f"  Relative Time: {info['relative_time']:.2%}")
        lines.append(f"  Running: {info['running']}")

        if info["current_offer"]:
            offer_str = ctx.format_outcome(tuple(info["current_offer"]))
            lines.append(f"  Current Offer: {offer_str}")
            if info["current_proposer"]:
                lines.append(f"  Proposer: {info['current_proposer']}")
        else:
            lines.append("  Current Offer: None")

        lines.append(f"  Acceptances: {info['n_acceptances']}")

        if info.get("broken"):
            lines.append("  Status: BROKEN")
        if info.get("timedout"):
            lines.append("  Status: TIMED OUT")
        if info.get("agreement"):
            lines.append(f"  Agreement: {info['agreement']}")

        return "\n".join(lines)


# =============================================================================
# Tag discovery and documentation functions
# =============================================================================


def get_available_tags() -> list[str]:
    """Get a list of all available tag names.

    Returns a list of tag names that can be used in prompts.
    Tags are used with the syntax: {{tag_name}} or {{tag_name:format(params)}}

    Returns:
        List of tag name strings.

    Example:
        >>> tags = get_available_tags()
        >>> print(tags[:3])
        ['outcome-space', 'utility-function', 'opponent-utility-function']
    """
    return [tag.value for tag in Tag]


def get_tag_documentation(tag_name: str) -> str | None:
    """Get documentation for a specific tag.

    Args:
        tag_name: The name of the tag (e.g., "outcome-space", "utility").

    Returns:
        Markdown-formatted documentation string, or None if tag not found.

    Example:
        >>> docs = get_tag_documentation("utility")
        >>> print(docs)  # Prints markdown documentation for the utility tag
    """
    docs = Tag.available_tags()
    return docs.get(tag_name)


def print_tag_help(tag_name: str | None = None) -> None:
    """Print formatted help for tags.

    If tag_name is provided, prints documentation for that specific tag.
    Otherwise, prints a summary of all available tags.

    Args:
        tag_name: Optional tag name to get help for. If None, prints all tags.

    Example:
        >>> print_tag_help()  # Prints summary of all tags
        >>> print_tag_help("utility")  # Prints detailed help for utility tag
    """
    if tag_name is not None:
        # Print specific tag documentation
        docs = get_tag_documentation(tag_name)
        if docs is None:
            print(f"Unknown tag: {tag_name}")
            print(f"Available tags: {', '.join(get_available_tags())}")
        else:
            print(docs)
    else:
        # Print summary of all tags
        print("Available Tags for LLM Prompts")
        print("=" * 40)
        print()
        print(
            "Tags are used with the syntax: {{tag_name}} or {{tag_name:format(params)}}"
        )
        print()
        print("Formats: 'text' (default), 'json'")
        print()

        # Group tags by category
        context_tags = [
            Tag.OUTCOME_SPACE,
            Tag.UTILITY_FUNCTION,
            Tag.OPPONENT_UTILITY_FUNCTION,
            Tag.NMI,
            Tag.CURRENT_STATE,
        ]
        reserved_tags = [Tag.RESERVED_VALUE, Tag.OPPONENT_RESERVED_VALUE]
        offer_tags = [
            Tag.MY_LAST_OFFER,
            Tag.MY_FIRST_OFFER,
            Tag.OPPONENT_LAST_OFFER,
            Tag.OPPONENT_FIRST_OFFER,
        ]
        history_tags = [
            Tag.PARTNER_OFFERS,
            Tag.HISTORY,
            Tag.TRACE,
            Tag.EXTENDED_TRACE,
            Tag.FULL_TRACE,
        ]
        utility_tags = [Tag.UTILITY]

        print("Context Tags (no parameters):")
        for tag in context_tags:
            print(f"  {{{{{tag.value}}}}}  - {_get_short_description(tag.value)}")

        print()
        print("Reserved Value Tags:")
        for tag in reserved_tags:
            print(f"  {{{{{tag.value}}}}}  - {_get_short_description(tag.value)}")

        print()
        print("Offer Reference Tags:")
        for tag in offer_tags:
            print(f"  {{{{{tag.value}}}}}  - {_get_short_description(tag.value)}")

        print()
        print("History Tags (optional k parameter):")
        for tag in history_tags:
            print(f"  {{{{{tag.value}}}}}  - {_get_short_description(tag.value)}")

        print()
        print("Utility Computation:")
        for tag in utility_tags:
            print(f"  {{{{{tag.value}}}}}  - {_get_short_description(tag.value)}")

        print()
        print("Use get_tag_documentation(tag_name) or print_tag_help(tag_name)")
        print("for detailed documentation on a specific tag.")


def _get_short_description(tag_name: str) -> str:
    """Get a short one-line description for a tag."""
    descriptions = {
        "outcome-space": "The negotiation outcome space definition",
        "utility-function": "Your utility function",
        "opponent-utility-function": "Opponent's utility function (if known)",
        "nmi": "Negotiation mechanism interface info",
        "current-state": "Current negotiation state",
        "reserved-value": "Your walk-away point",
        "opponent-reserved-value": "Opponent's walk-away point (if known)",
        "my-last-offer": "Your most recent offer",
        "my-first-offer": "Your first offer",
        "opponent-last-offer": "Opponent's most recent offer",
        "opponent-first-offer": "Opponent's first offer",
        "partner-offers": "List of opponent offers",
        "history": "Negotiation history (who offered what)",
        "trace": "Basic negotiation trace",
        "extended-trace": "Trace with response types",
        "full-trace": "Complete trace with all details",
        "utility": "Compute utility for an outcome",
    }
    return descriptions.get(tag_name, "No description available")


# =============================================================================
# Register all built-in handlers
# =============================================================================


def _register_builtin_handlers() -> None:
    """Register all built-in tag handlers."""
    register_tag_handler(Tag.OUTCOME_SPACE, _handle_outcome_space)
    register_tag_handler(Tag.UTILITY_FUNCTION, _handle_utility_function)
    register_tag_handler(
        Tag.OPPONENT_UTILITY_FUNCTION, _handle_opponent_utility_function
    )
    register_tag_handler(Tag.NMI, _handle_nmi)
    register_tag_handler(Tag.CURRENT_STATE, _handle_current_state)
    register_tag_handler(Tag.RESERVED_VALUE, _handle_reserved_value)
    register_tag_handler(Tag.OPPONENT_RESERVED_VALUE, _handle_opponent_reserved_value)
    register_tag_handler(Tag.MY_LAST_OFFER, _handle_my_last_offer)
    register_tag_handler(Tag.MY_FIRST_OFFER, _handle_my_first_offer)
    register_tag_handler(Tag.OPPONENT_LAST_OFFER, _handle_opponent_last_offer)
    register_tag_handler(Tag.OPPONENT_FIRST_OFFER, _handle_opponent_first_offer)
    register_tag_handler(Tag.PARTNER_OFFERS, _handle_partner_offers)
    register_tag_handler(Tag.HISTORY, _handle_history)
    register_tag_handler(Tag.TRACE, _handle_trace)
    register_tag_handler(Tag.EXTENDED_TRACE, _handle_extended_trace)
    register_tag_handler(Tag.FULL_TRACE, _handle_full_trace)
    register_tag_handler(Tag.UTILITY, _handle_utility)


# Register handlers on module load
_register_builtin_handlers()

"""Tests for the tag processing system."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

from negmas_llm import OllamaNegotiator
from negmas_llm.tags import (
    Tag,
    TagContext,
    TagFormat,
    _parse_params,
    get_tag_handler,
    process_prompt,
    register_tag_handler,
)


@pytest.fixture
def issues():
    """Create sample negotiation issues."""
    return [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=5),
        make_issue(name="delivery", values=["fast", "standard", "slow"]),
    ]


@pytest.fixture
def outcome_space(issues):
    """Create a sample outcome space."""
    return make_os(issues)


@pytest.fixture
def ufun(issues):
    """Create a sample utility function."""
    return LUFun.random(issues=issues, reserved_value=0.0)


@pytest.fixture
def negotiator(ufun):
    """Create a sample negotiator for testing."""
    return OllamaNegotiator(
        model="test-model",
        ufun=ufun,
        name="test-negotiator",
    )


@pytest.fixture
def mock_nmi(outcome_space):
    """Create a mock NMI."""
    nmi = MagicMock()
    nmi.outcome_space = outcome_space
    nmi.n_steps = 100
    nmi.time_limit = None
    return nmi


class TestTagEnums:
    """Test Tag and TagFormat enums."""

    def test_tag_values(self):
        """Test that all expected tags are defined."""
        assert Tag.OUTCOME_SPACE.value == "outcome-space"
        assert Tag.UTILITY_FUNCTION.value == "utility-function"
        assert Tag.OPPONENT_UTILITY_FUNCTION.value == "opponent-utility-function"
        assert Tag.MY_LAST_OFFER.value == "my-last-offer"
        assert Tag.MY_FIRST_OFFER.value == "my-first-offer"
        assert Tag.OPPONENT_LAST_OFFER.value == "opponent-last-offer"
        assert Tag.OPPONENT_FIRST_OFFER.value == "opponent-first-offer"
        assert Tag.PARTNER_OFFERS.value == "partner-offers"
        assert Tag.HISTORY.value == "history"
        assert Tag.TRACE.value == "trace"
        assert Tag.EXTENDED_TRACE.value == "extended-trace"
        assert Tag.FULL_TRACE.value == "full-trace"
        assert Tag.UTILITY.value == "utility"

    def test_tag_format_values(self):
        """Test TagFormat enum."""
        assert TagFormat.TEXT.value == "text"
        assert TagFormat.JSON.value == "json"


class TestTagContext:
    """Test TagContext class."""

    def test_format_outcome_text(self, negotiator, mock_nmi):
        """Test formatting outcome in text format."""
        negotiator._nmi = mock_nmi
        ctx = TagContext(negotiator, None, TagFormat.TEXT, {})

        outcome = (5, 2, "fast")
        formatted = ctx.format_outcome(outcome)

        # Should include issue names
        assert "price=5" in formatted
        assert "quantity=2" in formatted
        assert "delivery=fast" in formatted

    def test_format_outcome_json(self, negotiator):
        """Test formatting outcome in JSON format."""
        ctx = TagContext(negotiator, None, TagFormat.JSON, {})

        outcome = (5, 2, "fast")
        formatted = ctx.format_outcome(outcome)

        # Should be valid JSON list
        parsed = json.loads(formatted)
        assert parsed == [5, 2, "fast"]

    def test_format_outcome_none(self, negotiator):
        """Test formatting None outcome."""
        ctx_text = TagContext(negotiator, None, TagFormat.TEXT, {})
        ctx_json = TagContext(negotiator, None, TagFormat.JSON, {})

        assert ctx_text.format_outcome(None) == "None"
        assert ctx_json.format_outcome(None) == "null"

    def test_get_k_param(self, negotiator):
        """Test getting k parameter."""
        ctx = TagContext(negotiator, None, TagFormat.TEXT, {"k": "5"})
        assert ctx.get_k_param() == 5

        ctx_default = TagContext(negotiator, None, TagFormat.TEXT, {})
        assert ctx_default.get_k_param(10) == 10

        ctx_invalid = TagContext(negotiator, None, TagFormat.TEXT, {"k": "invalid"})
        assert ctx_invalid.get_k_param(3) == 3


class TestParseParams:
    """Test parameter parsing."""

    def test_empty_params(self):
        """Test parsing empty params."""
        assert _parse_params(None) == {}
        assert _parse_params("") == {}

    def test_single_param(self):
        """Test parsing single parameter."""
        result = _parse_params("k=5")
        assert result == {"k": "5"}

    def test_multiple_params(self):
        """Test parsing multiple parameters."""
        result = _parse_params("k=5, format=json")
        assert result == {"k": "5", "format": "json"}

    def test_nested_tag_param(self):
        """Test parsing parameter with nested tag."""
        result = _parse_params("outcome={{my-last-offer}}")
        assert result == {"outcome": "{{my-last-offer}}"}

    def test_complex_nested_params(self):
        """Test parsing with complex nested tags."""
        result = _parse_params("outcome={{utility:json(outcome={{my-last-offer}})}}")
        assert "outcome" in result
        assert "{{utility:json(outcome={{my-last-offer}})}}" in result["outcome"]


class TestTagHandlerRegistry:
    """Test tag handler registration and retrieval."""

    def test_get_builtin_handlers(self):
        """Test that all builtin handlers are registered."""
        for tag in Tag:
            handler = get_tag_handler(tag.value)
            assert handler is not None, f"Handler for {tag.value} not found"

    def test_register_custom_handler(self):
        """Test registering a custom handler."""

        def custom_handler(ctx: TagContext) -> str:
            return "custom result"

        register_tag_handler("custom-tag", custom_handler)

        handler = get_tag_handler("custom-tag")
        assert handler is not None
        assert handler.__name__ == "custom_handler"

    def test_get_unknown_handler(self):
        """Test getting handler for unknown tag."""
        handler = get_tag_handler("nonexistent-tag")
        assert handler is None


class TestProcessPrompt:
    """Test the process_prompt function."""

    def test_no_tags(self, negotiator):
        """Test prompt with no tags."""
        prompt = "This is a simple prompt with no tags."
        result = process_prompt(prompt, negotiator)
        assert result == prompt

    def test_unknown_tag_preserved(self, negotiator):
        """Test that unknown tags are preserved."""
        prompt = "This has an {{unknown-tag}} in it."
        result = process_prompt(prompt, negotiator)
        assert "{{unknown-tag}}" in result

    def test_outcome_space_tag(self, negotiator, mock_nmi):
        """Test outcome-space tag replacement."""
        negotiator._nmi = mock_nmi
        prompt = "The outcome space is: {{outcome-space:text}}"
        result = process_prompt(prompt, negotiator)

        assert "{{outcome-space" not in result
        assert "Outcome Space" in result or "price" in result

    def test_outcome_space_json(self, negotiator, mock_nmi):
        """Test outcome-space tag with JSON format."""
        negotiator._nmi = mock_nmi
        prompt = "{{outcome-space:json}}"
        result = process_prompt(prompt, negotiator)

        # Result should be valid JSON or contain JSON-like structure
        assert "{{" not in result

    def test_utility_function_tag(self, negotiator):
        """Test utility-function tag replacement."""
        prompt = "My utility: {{utility-function:text}}"
        result = process_prompt(prompt, negotiator)

        assert "{{utility-function" not in result
        assert "utility" in result.lower() or "Utility" in result

    def test_utility_function_no_ufun(self, negotiator):
        """Test utility-function tag when no ufun available."""
        # Use patch to properly mock the ufun property
        with patch.object(
            type(negotiator), "ufun", new_callable=lambda: property(lambda self: None)
        ):
            prompt = "{{utility-function}}"
            result = process_prompt(prompt, negotiator)

        assert "No utility function" in result or "not available" in result.lower()

    def test_opponent_ufun_unknown(self, negotiator):
        """Test opponent-utility-function when unknown."""
        negotiator._private_info = {}
        prompt = "{{opponent-utility-function}}"
        result = process_prompt(prompt, negotiator)

        assert "unknown" in result.lower() or "Unknown" in result

    def test_opponent_ufun_known(self, negotiator, ufun):
        """Test opponent-utility-function when known."""
        negotiator._private_info = {"opponent_ufun": ufun}
        prompt = "{{opponent-utility-function:text}}"
        result = process_prompt(prompt, negotiator)

        assert "{{" not in result
        assert "opponent" in result.lower() or "Opponent" in result

    def test_format_default_is_text(self, negotiator, mock_nmi):
        """Test that default format is text."""
        negotiator._nmi = mock_nmi
        prompt1 = "{{outcome-space}}"
        prompt2 = "{{outcome-space:text}}"

        result1 = process_prompt(prompt1, negotiator)
        result2 = process_prompt(prompt2, negotiator)

        # Both should produce similar text output (not JSON)
        assert "{{" not in result1
        assert "{{" not in result2


class TestOfferTags:
    """Test offer-related tags."""

    def test_my_last_offer_no_offers(self, negotiator):
        """Test my-last-offer when no offers made."""
        state = MagicMock()
        state.trace = []
        state.extended_trace = []

        prompt = "{{my-last-offer}}"
        result = process_prompt(prompt, negotiator, state)

        assert "None" in result or "null" in result

    def test_opponent_last_offer_no_offers(self, negotiator):
        """Test opponent-last-offer when no offers received."""
        state = MagicMock()
        state.trace = []
        state.extended_trace = []

        prompt = "{{opponent-last-offer}}"
        result = process_prompt(prompt, negotiator, state)

        assert "None" in result or "null" in result

    def test_offers_with_trace(self, negotiator):
        """Test offer tags with trace data."""
        state = MagicMock()
        state.trace = [
            (negotiator.id, (1, 2, "fast")),
            ("opponent_id", (3, 4, "slow")),
            (negotiator.id, (5, 6, "standard")),
        ]
        state.extended_trace = []

        my_last = process_prompt("{{my-last-offer}}", negotiator, state)
        my_first = process_prompt("{{my-first-offer}}", negotiator, state)
        opp_last = process_prompt("{{opponent-last-offer}}", negotiator, state)
        opp_first = process_prompt("{{opponent-first-offer}}", negotiator, state)

        # My offers: (1,2,fast), (5,6,standard)
        assert "5" in my_last or "standard" in my_last
        assert "1" in my_first or "fast" in my_first

        # Opponent offers: (3,4,slow)
        assert "3" in opp_last or "slow" in opp_last
        assert "3" in opp_first or "slow" in opp_first


class TestHistoryTags:
    """Test history-related tags."""

    def test_history_no_state(self, negotiator):
        """Test history tag without state."""
        prompt = "{{history}}"
        result = process_prompt(prompt, negotiator, None)

        assert "No history" in result or "no history" in result.lower()

    def test_history_with_trace(self, negotiator):
        """Test history tag with trace data."""
        state = MagicMock()
        state.trace = [
            (negotiator.id, (1, 2, "fast")),
            ("opponent_id", (3, 4, "slow")),
        ]

        prompt = "{{history:text}}"
        result = process_prompt(prompt, negotiator, state)

        assert "{{" not in result
        # Should contain some history info
        assert "history" in result.lower() or "You" in result or "Opponent" in result

    def test_history_with_k_param(self, negotiator):
        """Test history tag with k parameter."""
        state = MagicMock()
        state.trace = [
            (negotiator.id, (1, 1, "fast")),
            ("opponent_id", (2, 2, "slow")),
            (negotiator.id, (3, 3, "standard")),
            ("opponent_id", (4, 4, "fast")),
        ]

        prompt = "{{history:text(k=2)}}"
        result = process_prompt(prompt, negotiator, state)

        assert "{{" not in result

    def test_trace_json_format(self, negotiator):
        """Test trace tag with JSON format."""
        state = MagicMock()
        state.trace = [
            (negotiator.id, (1, 2, "fast")),
        ]

        prompt = "{{trace:json}}"
        result = process_prompt(prompt, negotiator, state)

        # Should be valid JSON
        try:
            parsed = json.loads(result)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail("trace:json should return valid JSON")


class TestUtilityTag:
    """Test the utility computation tag."""

    def test_utility_with_outcome(self, negotiator):
        """Test utility tag with explicit outcome."""
        prompt = "{{utility:text(outcome=(5, 2, 'fast'))}}"
        result = process_prompt(prompt, negotiator)

        assert "{{" not in result
        # Should be a number or error message
        try:
            float(result)
        except ValueError:
            # Might be an error message if outcome is invalid for ufun
            assert "error" in result.lower() or "Error" in result

    def test_utility_no_outcome_param(self, negotiator):
        """Test utility tag without outcome parameter."""
        prompt = "{{utility}}"
        result = process_prompt(prompt, negotiator)

        assert "outcome" in result.lower() or "required" in result.lower()

    def test_utility_no_ufun(self, negotiator):
        """Test utility tag when no ufun available."""
        with patch.object(
            type(negotiator), "ufun", new_callable=lambda: property(lambda self: None)
        ):
            prompt = "{{utility:text(outcome=(1, 1, 'fast'))}}"
            result = process_prompt(prompt, negotiator)

        assert "No utility function" in result or "not available" in result.lower()


class TestNestedTags:
    """Test nested tag processing."""

    def test_nested_utility_with_offer_tag(self, negotiator):
        """Test utility tag with nested offer tag."""
        state = MagicMock()
        state.trace = [
            ("opponent_id", (5, 3, "fast")),
        ]
        state.extended_trace = []

        # Nested: utility of opponent's last offer
        prompt = "{{utility:text(outcome={{opponent-last-offer}})}}"
        result = process_prompt(prompt, negotiator, state)

        # Should process both tags
        assert "{{" not in result

    def test_multiple_nested_tags(self, negotiator, mock_nmi):
        """Test multiple nested tags in same prompt."""
        negotiator._nmi = mock_nmi
        state = MagicMock()
        state.trace = [("opponent_id", (2, 1, "slow"))]
        state.extended_trace = []

        prompt = "Space: {{outcome-space:text}} | Offer: {{opponent-last-offer:text}}"
        result = process_prompt(prompt, negotiator, state)

        assert "{{" not in result
        assert "|" in result  # Separator should still be there


class TestCustomTagHandler:
    """Test custom tag handler registration."""

    def test_custom_handler_works(self, negotiator):
        """Test that custom handlers work correctly."""

        def my_custom_handler(ctx: TagContext) -> str:
            k = ctx.get_k_param(1)
            if ctx.format == TagFormat.JSON:
                return json.dumps({"custom": True, "k": k})
            return f"Custom result with k={k}"

        register_tag_handler("my-custom", my_custom_handler)

        prompt = "Result: {{my-custom:text(k=42)}}"
        result = process_prompt(prompt, negotiator)

        assert "Custom result with k=42" in result

    def test_custom_handler_json_format(self, negotiator):
        """Test custom handler with JSON format."""

        def json_handler(ctx: TagContext) -> str:
            return json.dumps({"status": "ok", "format": ctx.format.value})

        register_tag_handler("json-test", json_handler)

        prompt = "{{json-test:json}}"
        result = process_prompt(prompt, negotiator)

        parsed = json.loads(result)
        assert parsed["format"] == "json"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_tag_preserved(self, negotiator):
        """Test that malformed tags are preserved."""
        prompts = [
            "{{incomplete",
            "not a tag}}",
            "{{ spaces }}",  # Spaces inside
        ]
        for prompt in prompts:
            result = process_prompt(prompt, negotiator)
            # Malformed tags should be preserved or handled gracefully
            assert result is not None

    def test_empty_prompt(self, negotiator):
        """Test empty prompt."""
        result = process_prompt("", negotiator)
        assert result == ""

    def test_escaped_braces(self, negotiator):
        """Test that escaped braces are preserved as literals."""
        # Single escape
        prompt = "JSON example: \\{{key\\}}"
        result = process_prompt(prompt, negotiator)
        assert result == "JSON example: {{key}}"

        # Double braces escaped
        prompt = "Use \\{{tag-name\\}} for tags"
        result = process_prompt(prompt, negotiator)
        assert result == "Use {{tag-name}} for tags"

        # Mixed escaped and real tags
        prompt = "Real: {{reserved-value}}, Escaped: \\{{not-a-tag\\}}"
        result = process_prompt(prompt, negotiator)
        assert "\\{{" not in result
        assert "{{not-a-tag}}" in result
        # reserved-value should be replaced with actual value
        assert "{{reserved-value}}" not in result

    def test_escaped_braces_in_json_example(self, negotiator):
        """Test escaped braces for showing JSON format in prompts."""
        prompt = """Respond in JSON format:
\\{{"decision": "accept", "reasoning": "explanation"\\}}"""
        result = process_prompt(prompt, negotiator)
        assert (
            result
            == """Respond in JSON format:
{{"decision": "accept", "reasoning": "explanation"}}"""
        )

    def test_max_iterations_protection(self, negotiator):
        """Test that infinite loops are prevented."""

        # Create a handler that produces another tag (potential infinite loop)
        def recursive_handler(ctx: TagContext) -> str:
            return "{{recursive-tag}}"

        register_tag_handler("recursive-tag", recursive_handler)

        prompt = "{{recursive-tag}}"
        # Should not hang, max iterations should stop it
        result = process_prompt(prompt, negotiator)
        # Result will contain the tag since it keeps regenerating
        assert result is not None

    def test_special_characters_in_values(self, negotiator):
        """Test handling of special characters."""
        state = MagicMock()
        state.trace = [
            (negotiator.id, (1, 2, "a=b,c")),  # Contains = and ,
        ]
        state.extended_trace = []

        prompt = "{{my-last-offer}}"
        result = process_prompt(prompt, negotiator, state)

        assert "{{" not in result


class TestNewTags:
    """Test new tags: nmi, current-state, reserved-value, opponent-reserved-value."""

    def test_nmi_tag_text(self, negotiator_with_nmi):
        """Test nmi tag in text format."""
        prompt = "{{nmi}}"
        result = process_prompt(prompt, negotiator_with_nmi)

        assert "{{" not in result
        assert "Negotiation Mechanism" in result or "N Steps" in result

    def test_nmi_tag_json(self, negotiator_with_nmi):
        """Test nmi tag in JSON format."""
        prompt = "{{nmi:json}}"
        result = process_prompt(prompt, negotiator_with_nmi)

        parsed = json.loads(result)
        assert "n_steps" in parsed or "n_outcomes" in parsed

    def test_current_state_tag_text(self, negotiator_with_nmi):
        """Test current-state tag in text format."""
        state = MagicMock()
        state.step = 5
        state.relative_time = 0.25
        state.running = True
        state.current_offer = (3, 2, "fast")
        state.current_proposer = "opponent_1"
        state.n_acceptances = 0
        state.broken = False
        state.timedout = False
        state.agreement = None
        state.new_offers = []

        prompt = "{{current-state}}"
        result = process_prompt(prompt, negotiator_with_nmi, state)

        assert "{{" not in result
        assert "Step" in result
        assert "5" in result

    def test_current_state_tag_json(self, negotiator_with_nmi):
        """Test current-state tag in JSON format."""
        state = MagicMock()
        state.step = 3
        state.relative_time = 0.15
        state.running = True
        state.current_offer = (2, 1, "slow")
        state.current_proposer = None
        state.n_acceptances = 0
        state.broken = False
        state.timedout = False
        state.agreement = None
        state.new_offers = []

        prompt = "{{current-state:json}}"
        result = process_prompt(prompt, negotiator_with_nmi, state)

        parsed = json.loads(result)
        assert parsed["step"] == 3
        assert parsed["running"] is True
        assert parsed["current_offer"] == [2, 1, "slow"]

    def test_reserved_value_tag(self, negotiator):
        """Test reserved-value tag."""
        prompt = "{{reserved-value}}"
        result = process_prompt(prompt, negotiator)

        # Should be a number or "No reserved value set"
        assert "{{" not in result

    def test_reserved_value_tag_json(self, negotiator):
        """Test reserved-value tag in JSON format."""
        prompt = "{{reserved-value:json}}"
        result = process_prompt(prompt, negotiator)

        parsed = json.loads(result)
        assert "reserved_value" in parsed

    def test_opponent_reserved_value_unknown(self, negotiator):
        """Test opponent-reserved-value tag when unknown."""
        prompt = "{{opponent-reserved-value}}"
        result = process_prompt(prompt, negotiator)

        assert "{{" not in result
        assert "unknown" in result.lower()

    def test_opponent_reserved_value_known(self, negotiator_with_opponent_ufun):
        """Test opponent-reserved-value tag when known."""
        prompt = "{{opponent-reserved-value}}"
        result = process_prompt(prompt, negotiator_with_opponent_ufun)

        assert "{{" not in result
        # Should be a number (could be 0.0 or similar)

    def test_nmi_no_nmi(self, negotiator):
        """Test nmi tag when no NMI is available."""
        prompt = "{{nmi}}"
        result = process_prompt(prompt, negotiator)

        assert "{{" not in result
        assert "No NMI" in result or "not in a negotiation" in result.lower()

    def test_current_state_no_state(self, negotiator):
        """Test current-state tag when no state is available."""
        prompt = "{{current-state}}"
        result = process_prompt(prompt, negotiator)

        assert "{{" not in result
        assert "No state" in result


class TestAvailableTags:
    """Test the Tag.available_tags() class method."""

    def test_available_tags_returns_dict(self):
        """Test that available_tags returns a dictionary."""
        docs = Tag.available_tags()
        assert isinstance(docs, dict)

    def test_available_tags_contains_all_tags(self):
        """Test that available_tags contains all Tag enum values."""
        docs = Tag.available_tags()

        for tag in Tag:
            assert tag.value in docs, f"Tag {tag.value} not in available_tags()"

    def test_available_tags_has_markdown_format(self):
        """Test that documentation is in markdown format."""
        docs = Tag.available_tags()

        for tag_name, doc in docs.items():
            # Should have a header
            assert "###" in doc, f"Tag {tag_name} missing markdown header"
            # Should have description
            assert "**" in doc, f"Tag {tag_name} missing bold formatting"
            # Should mention parameters or formats
            assert "Parameter" in doc or "Format" in doc, (
                f"Tag {tag_name} missing parameters/formats section"
            )

    def test_available_tags_utility_has_outcome_param(self):
        """Test that utility tag documents the outcome parameter."""
        docs = Tag.available_tags()
        utility_doc = docs["utility"]

        assert "outcome" in utility_doc.lower()
        assert "required" in utility_doc.lower()

    def test_available_tags_history_has_k_param(self):
        """Test that history-related tags document the k parameter."""
        docs = Tag.available_tags()

        for tag_name in ["history", "trace", "extended-trace", "full-trace"]:
            doc = docs[tag_name]
            assert "k" in doc, f"Tag {tag_name} should document k parameter"


@pytest.fixture
def negotiator_with_nmi(ufun, outcome_space):
    """Create a negotiator with a mocked NMI."""
    neg = OllamaNegotiator(
        model="llama3.2",
        ufun=ufun,
    )
    # Mock NMI
    nmi = MagicMock()
    nmi.outcome_space = outcome_space
    nmi.n_steps = 100
    nmi.time_limit = 60.0
    nmi.n_outcomes = 150
    nmi.n_negotiators = 2
    nmi.end_on_no_response = True
    nmi.one_offer_per_step = False
    nmi.offering_is_accepting = True
    nmi.dynamic_entry = False
    neg._nmi = nmi
    return neg


@pytest.fixture
def negotiator_with_opponent_ufun(ufun, issues):
    """Create a negotiator with opponent ufun in private_info."""
    opponent_ufun = LUFun.random(issues=issues, reserved_value=0.25)
    neg = OllamaNegotiator(
        model="llama3.2",
        ufun=ufun,
        private_info={"opponent_ufun": opponent_ufun},
    )
    return neg

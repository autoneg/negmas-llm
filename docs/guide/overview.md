# Overview

`negmas-llm` bridges the gap between Large Language Models and automated negotiation by providing LLM-based negotiators that work seamlessly with the negmas framework.

## Architecture

### Core Components

1. **`LLMNegotiator`** - The abstract base class that handles:
   - LLM communication via litellm
   - Prompt construction and formatting
   - Response parsing
   - Conversation history management

2. **Provider-Specific Classes** - Pre-configured negotiators for each provider:
   - Cloud: `OpenAINegotiator`, `AnthropicNegotiator`, `GeminiNegotiator`, etc.
   - Local: `OllamaNegotiator`, `VLLMNegotiator`, `LMStudioNegotiator`, etc.

### Information Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Negotiation   │     │  LLMNegotiator  │     │       LLM       │
│   Mechanism     │────▶│                 │────▶│    Provider     │
│   (negmas)      │     │  - Format info  │     │                 │
│                 │◀────│  - Parse resp   │◀────│                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Information Provided to the LLM

### System Prompt (once per negotiation)

The system prompt includes:

1. **Outcome Space**: The issues being negotiated and their possible values
2. **NMI Info**: Mechanism parameters (time limits, step limits, etc.)
3. **Your Utility Function**: How outcomes are valued
4. **Partner's Utility Function**: If known, shared via `private_info`
5. **Response Format**: JSON format instructions

### Per-Round Messages

Each negotiation round, the LLM receives:

1. **Current State**: Step number, timing, running status
2. **Current Offer**: The offer being considered
3. **Utilities**: Calculated utilities for both parties
4. **History**: Previous conversation turns

## LLM Response Format

The LLM must respond with JSON:

```json
{
    "response_type": "accept" | "reject" | "end",
    "outcome": [value1, value2, ...] | null,
    "text": "optional explanation",
    "data": {} | null
}
```

- `accept`: Accept the current offer
- `reject`: Reject and counter with `outcome`
- `end`: End negotiation without agreement

## Key Concepts

### Utility Functions

Negotiators use utility functions to evaluate outcomes:

```python
from negmas.preferences import LinearAdditiveUtilityFunction

ufun = LinearAdditiveUtilityFunction(
    weights={"price": 0.6, "quality": 0.4},
    issues=issues,
)
```

The utility is communicated to the LLM so it can make informed decisions.

### Private Information

You can share information about the opponent:

```python
negotiator = OpenAINegotiator(
    model="gpt-4o",
    ufun=my_ufun,
    private_info={"opponent_ufun": their_ufun},
)
```

### Conversation History

The negotiator maintains conversation history across rounds, giving the LLM context about how the negotiation has progressed.

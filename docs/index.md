# negmas-llm

**LLM-based negotiators for the negmas negotiation framework**

`negmas-llm` integrates Large Language Models with the [negmas](https://negmas.readthedocs.io/) negotiation framework, enabling you to create negotiators that leverage LLMs for intelligent decision-making in automated negotiations.

## Key Features

- **Multiple LLM Providers**: Support for 16+ LLM providers through [litellm](https://github.com/BerriAI/litellm), including OpenAI, Anthropic, Google Gemini, Cohere, Mistral, and local models via Ollama, vLLM, and more.

- **Seamless negmas Integration**: `LLMNegotiator` extends `SAONegotiator`, making it fully compatible with existing negmas mechanisms and negotiation protocols.

- **Customizable Prompts**: Override formatting methods to control exactly how negotiation information is presented to the LLM.

- **Rich Context**: Automatically provides the LLM with outcome space, utility functions, negotiation mechanism parameters, and complete state information.

## Quick Example

```python
from negmas import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.outcomes import make_issue
from negmas_llm import OpenAINegotiator

# Define the negotiation space
issues = [
    make_issue("price", (0, 100)),
    make_issue("quantity", (1, 10)),
]

# Create negotiators with utility functions
seller = OpenAINegotiator(
    model="gpt-4o",
    name="seller",
    ufun=LinearAdditiveUtilityFunction(weights=[0.6, 0.4], issues=issues),
)
buyer = OpenAINegotiator(
    model="gpt-4o", 
    name="buyer",
    ufun=LinearAdditiveUtilityFunction(weights=[-0.6, 0.4], issues=issues),
)

# Run the negotiation
mechanism = SAOMechanism(outcome_space=issues, n_steps=20)
mechanism.add(seller)
mechanism.add(buyer)
result = mechanism.run()

print(f"Agreement: {result.agreement}")
```

## How It Works

1. **System Prompt**: At the start of negotiation, the LLM receives comprehensive context including:
   - The outcome space (issues and their possible values)
   - Mechanism parameters (time limits, step limits, etc.)
   - Your utility function (what outcomes you prefer)
   - Partner's utility function (if known)
   - Response format instructions

2. **Per-Round Messages**: Each negotiation round, the LLM receives:
   - Complete negotiation state (step, timing, offers, acceptances)
   - The current offer and its utilities for both parties
   - Request for a JSON-formatted decision

3. **LLM Response**: The LLM responds with a JSON object containing:
   - `response_type`: "accept", "reject", or "end"
   - `outcome`: Counter-offer values (if rejecting)
   - `text`: Optional explanation
   - `data`: Optional additional data

## Installation

```bash
pip install negmas-llm
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running in minutes
- [User Guide](guide/overview.md) - In-depth documentation
- [API Reference](api/negotiators.md) - Complete API documentation

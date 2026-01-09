# negmas-llm

LLM-based negotiators for the [negmas](https://github.com/yasserfarouk/negmas) negotiation framework.

[![PyPI](https://img.shields.io/pypi/v/negmas-llm.svg)](https://pypi.org/project/negmas-llm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/negmas-llm.svg)](https://pypi.org/project/negmas-llm/)
[![PyPI - Status](https://img.shields.io/pypi/status/negmas-llm.svg)](https://pypi.org/project/negmas-llm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/negmas-llm.svg)](https://pypi.org/project/negmas-llm/)
[![PyPI - License](https://img.shields.io/pypi/l/negmas-llm.svg)](https://pypi.org/project/negmas-llm/)
[![Documentation Status](https://readthedocs.org/projects/negmas-llm/badge/?version=latest)](https://negmas-llm.readthedocs.io/en/latest/)
[![CI](https://github.com/autoneg/negmas-llm/workflows/CI/badge.svg)](https://github.com/autoneg/negmas-llm/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

`negmas-llm` integrates Large Language Models with the negmas negotiation framework, allowing you to create negotiators that use LLMs for decision-making. The library supports multiple LLM providers through [litellm](https://github.com/BerriAI/litellm), including:

- **Cloud providers**: OpenAI, Anthropic, Google Gemini, Cohere, Mistral, Groq, Together AI, Azure OpenAI, AWS Bedrock, OpenRouter, DeepSeek, Hugging Face
- **Local models**: Ollama, vLLM, LM Studio, text-generation-webui

## Installation

```bash
pip install negmas-llm
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add negmas-llm
```

## Quick Start

```python
from negmas import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.outcomes import make_issue
from negmas_llm import OllamaNegotiator, OpenAINegotiator

# Define negotiation issues
issues = [
    make_issue("price", (0, 100)),
    make_issue("quantity", (1, 10)),
]

# Create utility functions for each party
seller_ufun = LinearAdditiveUtilityFunction(
    weights=[0.6, 0.4],
    issues=issues,
)
buyer_ufun = LinearAdditiveUtilityFunction(
    weights=[-0.6, 0.4],  # Buyer prefers lower prices
    issues=issues,
)

# Create LLM-based negotiators
seller = OpenAINegotiator(
    model="gpt-4o",
    name="seller",
    ufun=seller_ufun,
)
buyer = OllamaNegotiator(
    model="llama3.2",
    name="buyer",
    ufun=buyer_ufun,
)

# Run the negotiation
mechanism = SAOMechanism(outcome_space=issues, n_steps=20)
mechanism.add(seller)
mechanism.add(buyer)
result = mechanism.run()

print(f"Agreement: {result.agreement}")
```

## Supported Negotiators

| Class | Provider | Default Model |
|-------|----------|---------------|
| `OpenAINegotiator` | OpenAI | gpt-4o |
| `AnthropicNegotiator` | Anthropic | claude-sonnet-4-20250514 |
| `GeminiNegotiator` | Google | gemini-2.0-flash |
| `CohereNegotiator` | Cohere | command-r-plus |
| `MistralNegotiator` | Mistral AI | mistral-large-latest |
| `GroqNegotiator` | Groq | llama-3.3-70b-versatile |
| `TogetherAINegotiator` | Together AI | Meta-Llama-3.1-70B-Instruct-Turbo |
| `AzureOpenAINegotiator` | Azure OpenAI | (deployment name) |
| `AWSBedrockNegotiator` | AWS Bedrock | claude-3-sonnet |
| `OpenRouterNegotiator` | OpenRouter | openai/gpt-4o |
| `DeepSeekNegotiator` | DeepSeek | deepseek-chat |
| `HuggingFaceNegotiator` | Hugging Face | Llama-3.2-3B-Instruct |
| `OllamaNegotiator` | Ollama (local) | llama3.2 |
| `VLLMNegotiator` | vLLM (local) | (model name) |
| `LMStudioNegotiator` | LM Studio (local) | local-model |
| `TextGenWebUINegotiator` | text-generation-webui | local-model |

## Customization

You can customize how information is presented to the LLM by subclassing `LLMNegotiator` and overriding the formatting methods:

```python
from negmas_llm import LLMNegotiator

class CustomNegotiator(LLMNegotiator):
    def format_outcome_space(self, state):
        # Customize how the outcome space is described
        return "Custom outcome space description..."

    def format_own_ufun(self, state):
        # Customize how your utility function is described
        return "Custom utility function description..."

    def format_state(self, state, offer=None):
        # Customize how the negotiation state is presented
        return "Custom state description..."
```

For complete control over the prompts, override `build_system_prompt` or `build_user_message`:

```python
from negmas_llm import LLMNegotiator

class CustomPromptNegotiator(LLMNegotiator):
    def build_system_prompt(self, state):
        # Complete control over the system prompt
        # By default, this combines: format_outcome_space(), format_nmi_info(),
        # format_own_ufun(), format_partner_ufun(), format_response_instructions()
        return f"""You are an aggressive negotiator who never accepts the first offer.

{self.format_outcome_space(state)}
{self.format_own_ufun(state)}
{self.format_response_instructions()}

Always aim for at least 80% of your maximum utility."""

    def build_user_message(self, state, offer, source):
        # Complete control over the per-round user message
        if offer is not None:
            return f"Round {state.step}: Opponent offered {offer}. Counter aggressively."
        return f"Round {state.step}: Make an ambitious opening offer."
```

## Documentation

Full documentation is available at [negmas-llm.readthedocs.io](https://negmas-llm.readthedocs.io/en/latest/).

## License

GNU Affero General Public License v3.0 or later - see [LICENSE](LICENSE) for details.

## Related Projects

- [negmas](https://github.com/yasserfarouk/negmas) - NEGotiation MAnagement System
- [litellm](https://github.com/BerriAI/litellm) - Call all LLM APIs using the OpenAI format

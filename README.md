# negmas-llm

LLM-based negotiators for the [negmas](https://github.com/yasserfarouk/negmas) negotiation framework.

[![Documentation Status](https://readthedocs.org/projects/negmas-llm/badge/?version=latest)](https://negmas-llm.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/negmas-llm.svg)](https://badge.fury.io/py/negmas-llm)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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

## Documentation

Full documentation is available at [negmas-llm.readthedocs.io](https://negmas-llm.readthedocs.io/).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [negmas](https://github.com/yasserfarouk/negmas) - NEGotiation MAnagement System
- [litellm](https://github.com/BerriAI/litellm) - Call all LLM APIs using the OpenAI format

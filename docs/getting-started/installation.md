# Installation

## Requirements

- Python 3.11 or higher
- [negmas](https://negmas.readthedocs.io/) >= 0.11.3
- [litellm](https://github.com/BerriAI/litellm) >= 1.80.11

## Install from PyPI

```bash
pip install negmas-llm
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
uv add negmas-llm
```

## Install from Source

```bash
git clone https://github.com/yasserfarouk/negmas-llm.git
cd negmas-llm
pip install -e .
```

## Provider-Specific Setup

### Cloud Providers

Most cloud providers require API keys. Set them as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google Gemini
export GOOGLE_API_KEY="your-key"

# Cohere
export COHERE_API_KEY="your-key"

# Mistral
export MISTRAL_API_KEY="your-key"

# Groq
export GROQ_API_KEY="your-key"

# Together AI
export TOGETHER_API_KEY="your-key"

# OpenRouter
export OPENROUTER_API_KEY="your-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-key"

# Hugging Face
export HF_TOKEN="your-token"
```

### Local Models with Ollama

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3.2`
3. Start the Ollama server (runs by default on `http://localhost:11434`)

```python
from negmas_llm import OllamaNegotiator

negotiator = OllamaNegotiator(model="llama3.2")
```

### Local Models with vLLM

1. Install vLLM: `pip install vllm`
2. Start the server: `python -m vllm.entrypoints.openai.api_server --model your-model`

```python
from negmas_llm import VLLMNegotiator

negotiator = VLLMNegotiator(
    model="your-model",
    api_base="http://localhost:8000/v1"
)
```

### Local Models with LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Load a model and start the local server

```python
from negmas_llm import LMStudioNegotiator

negotiator = LMStudioNegotiator()  # Uses default localhost:1234
```

## Verifying Installation

```python
import negmas_llm

# List available negotiator classes
print(negmas_llm.__all__)
```

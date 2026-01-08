# Providers

`negmas-llm` supports a wide range of LLM providers through [litellm](https://github.com/BerriAI/litellm).

## Cloud Providers

### OpenAI

```python
from negmas_llm import OpenAINegotiator

negotiator = OpenAINegotiator(
    model="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo", etc.
    api_key="sk-...",  # or set OPENAI_API_KEY env var
)
```

### Anthropic

```python
from negmas_llm import AnthropicNegotiator

negotiator = AnthropicNegotiator(
    model="claude-sonnet-4-20250514",  # or "claude-3-opus", etc.
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
)
```

### Google Gemini

```python
from negmas_llm import GeminiNegotiator

negotiator = GeminiNegotiator(
    model="gemini-2.0-flash",  # or "gemini-pro", etc.
    api_key="...",  # or set GOOGLE_API_KEY env var
)
```

### Cohere

```python
from negmas_llm import CohereNegotiator

negotiator = CohereNegotiator(
    model="command-r-plus",
    api_key="...",  # or set COHERE_API_KEY env var
)
```

### Mistral

```python
from negmas_llm import MistralNegotiator

negotiator = MistralNegotiator(
    model="mistral-large-latest",
    api_key="...",  # or set MISTRAL_API_KEY env var
)
```

### Groq

```python
from negmas_llm import GroqNegotiator

negotiator = GroqNegotiator(
    model="llama-3.3-70b-versatile",
    api_key="...",  # or set GROQ_API_KEY env var
)
```

### Together AI

```python
from negmas_llm import TogetherAINegotiator

negotiator = TogetherAINegotiator(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key="...",  # or set TOGETHER_API_KEY env var
)
```

### Azure OpenAI

```python
from negmas_llm import AzureOpenAINegotiator

negotiator = AzureOpenAINegotiator(
    model="your-deployment-name",
    api_key="...",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview",
)
```

### AWS Bedrock

```python
from negmas_llm import AWSBedrockNegotiator

negotiator = AWSBedrockNegotiator(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    aws_region="us-east-1",
)
```

### OpenRouter

```python
from negmas_llm import OpenRouterNegotiator

negotiator = OpenRouterNegotiator(
    model="openai/gpt-4o",
    api_key="...",  # or set OPENROUTER_API_KEY env var
)
```

### DeepSeek

```python
from negmas_llm import DeepSeekNegotiator

negotiator = DeepSeekNegotiator(
    model="deepseek-chat",
    api_key="...",  # or set DEEPSEEK_API_KEY env var
)
```

### Hugging Face

```python
from negmas_llm import HuggingFaceNegotiator

negotiator = HuggingFaceNegotiator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    api_key="...",  # or set HF_TOKEN env var
)
```

## Local Providers

### Ollama

```python
from negmas_llm import OllamaNegotiator

negotiator = OllamaNegotiator(
    model="llama3.2",
    api_base="http://localhost:11434",  # default
)
```

### vLLM

```python
from negmas_llm import VLLMNegotiator

negotiator = VLLMNegotiator(
    model="your-model-name",
    api_base="http://localhost:8000/v1",  # default
)
```

### LM Studio

```python
from negmas_llm import LMStudioNegotiator

negotiator = LMStudioNegotiator(
    model="local-model",  # default
    api_base="http://localhost:1234/v1",  # default
)
```

### text-generation-webui

```python
from negmas_llm import TextGenWebUINegotiator

negotiator = TextGenWebUINegotiator(
    model="local-model",
    api_base="http://localhost:5000/v1",  # default
)
```

## Using the Base Class

For providers not listed above, use `LLMNegotiator` directly:

```python
from negmas_llm import LLMNegotiator

negotiator = LLMNegotiator(
    provider="your-provider",
    model="your-model",
    api_key="...",
    api_base="...",
)
```

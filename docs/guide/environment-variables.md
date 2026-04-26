# Environment Variables

`negmas-llm` supports several environment variables for configuration.

## Default Model Configuration

You can set default models for each provider using environment variables. These are used when you don't explicitly specify a model in the negotiator constructor.

| Environment Variable | Provider | Default Value |
|---------------------|----------|---------------|
| `NEGMAS_LLM_OLLAMA_DEFAULT_MODEL` | Ollama | `qwen3:4b-instruct` |
| `NEGMAS_LLM_OPENAI_DEFAULT_MODEL` | OpenAI | `gpt-4o-mini` |
| `NEGMAS_LLM_ANTHROPIC_DEFAULT_MODEL` | Anthropic | `claude-sonnet-4-20250514` |
| `NEGMAS_LLM_GEMINI_DEFAULT_MODEL` | Google Gemini | `gemini-2.0-flash` |
| `NEGMAS_LLM_GITHUB_COPILOT_DEFAULT_MODEL` | GitHub Copilot | `gpt-4o` |
| `NEGMAS_LLM_GITHUB_DEFAULT_MODEL` | GitHub Models | `gpt-4o` |
| `NEGMAS_LLM_GROQ_DEFAULT_MODEL` | Groq | `llama-3.3-70b-versatile` |
| `NEGMAS_LLM_MISTRAL_DEFAULT_MODEL` | Mistral | `mistral-large-latest` |
| `NEGMAS_LLM_DEEPSEEK_DEFAULT_MODEL` | DeepSeek | `deepseek-chat` |
| `NEGMAS_LLM_OPENROUTER_DEFAULT_MODEL` | OpenRouter | `openai/gpt-4o-mini` |
| `NEGMAS_LLM_TOGETHER_AI_DEFAULT_MODEL` | Together AI | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| `NEGMAS_LLM_COHERE_DEFAULT_MODEL` | Cohere | `command-r-plus` |
| `NEGMAS_LLM_HUGGINGFACE_DEFAULT_MODEL` | Hugging Face | `meta-llama/Llama-3.2-3B-Instruct` |

### Example Usage

```bash
# Set a different default model for Ollama
export NEGMAS_LLM_OLLAMA_DEFAULT_MODEL=llama3.2:latest

# Set a different default model for OpenAI
export NEGMAS_LLM_OPENAI_DEFAULT_MODEL=gpt-4-turbo
```

Then in your Python code:

```python
from negmas_llm import OllamaNegotiator, OpenAINegotiator

# These will use the models from environment variables
ollama_negotiator = OllamaNegotiator()  # Uses llama3.2:latest
openai_negotiator = OpenAINegotiator()  # Uses gpt-4-turbo
```

### Legacy Support

For backwards compatibility, `OLLAMA_MODEL` is also supported for the Ollama provider. The new `NEGMAS_LLM_OLLAMA_DEFAULT_MODEL` takes precedence if both are set.

## API Keys

Each provider has its own environment variable for API keys. These are handled by [litellm](https://github.com/BerriAI/litellm) and follow its conventions.

| Environment Variable | Provider |
|---------------------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google Gemini |
| `COHERE_API_KEY` | Cohere |
| `MISTRAL_API_KEY` | Mistral |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `HF_TOKEN` | Hugging Face |
| `GITHUB_TOKEN` | GitHub Models |

### Example Usage

```bash
# Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Then in your Python code:

```python
from negmas_llm import OpenAINegotiator

# API key is automatically read from environment
negotiator = OpenAINegotiator(model="gpt-4o")
```

## Azure OpenAI

For Azure OpenAI, you can use these environment variables:

| Environment Variable | Description |
|---------------------|-------------|
| `AZURE_API_KEY` | Azure OpenAI API key |
| `AZURE_API_BASE` | Azure OpenAI endpoint URL |
| `AZURE_API_VERSION` | Azure OpenAI API version |

## AWS Bedrock

For AWS Bedrock, standard AWS credentials are used:

| Environment Variable | Description |
|---------------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION_NAME` | AWS region (e.g., `us-east-1`) |

## LLM Call Configuration

You can configure timeout and retry behavior for all LLM calls:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NEGMAS_LLM_TIMEOUT` | Timeout in seconds for LLM calls | None (no timeout) |
| `NEGMAS_LLM_NUM_RETRIES` | Number of retries for failed LLM calls | None (no retries) |

### Example Usage

```bash
# Set a 30-second timeout for all LLM calls
export NEGMAS_LLM_TIMEOUT=30

# Retry failed calls up to 3 times
export NEGMAS_LLM_NUM_RETRIES=3
```

These can also be set programmatically:

```python
from negmas_llm import OpenAINegotiator

# Override environment settings for this negotiator
negotiator = OpenAINegotiator(
    model="gpt-4o",
    timeout=60,  # 60 second timeout
    num_retries=2,  # 2 retries
)
```

See the [litellm documentation](https://docs.litellm.ai/) for more details on provider-specific environment variables.

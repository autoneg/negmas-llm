# Environment Variables

`negmas-llm` supports several environment variables for configuration. The most
important is the **provider/model resolution** system, which lets you decide
which LLM every negotiator uses — globally, per provider, or per negotiator type
— without touching code.

## Provider & Model Resolution (single source of truth)

Every LLM negotiator, meta-negotiator, and component resolves its effective
configuration through **one** function, `resolve_llm_config` in
`negmas_llm/config.py`. There is exactly one resolution algorithm, described
here, so behavior is predictable and easy to drive from the environment when
running experiments.

Each negotiator stores its resolved `provider` and `model` on the instance, and
the built-in fallbacks live as the class attributes `DEFAULT_PROVIDER` /
`DEFAULT_MODEL` next to each negotiator class.

### The three configuration axes

| Axis | Environment variable | Scope |
|------|----------------------|-------|
| **Global** | `NEGMAS_LLM_<VAR>` | Every negotiator that does not set the value explicitly. *"Run the whole experiment on model X."* |
| **Per negotiator type** | `NEGMAS_LLM_<ClassName>_<VAR>` | Only the negotiator whose concrete class name is `<ClassName>`. *"Everyone on X, but this one on Y."* |
| **Per provider (model only)** | `NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL` | The default model used *whenever that provider is selected*, regardless of negotiator type. |

`<VAR>` is one of: `PROVIDER`, `MODEL`, `EFFORT`, `TEMPERATURE`, `MAX_TOKENS`,
`TIMEOUT`, `NUM_RETRIES`, `API_KEY`, `API_BASE`. (`EFFORT` is the reasoning
effort — e.g. `low`/`medium`/`high` — sent verbatim as `reasoning_effort`.)

`<ClassName>` is the **exact** concrete class name of the negotiator, e.g.
`LLMBoulwareTBNegotiator`, `OpenAINegotiator`, or the component class
`OpenAIOfferingPolicy`. It is case-sensitive.

!!! tip "Discover the exact variable name"
    ```python
    from negmas_llm import per_type_env_var
    per_type_env_var("LLMBoulwareTBNegotiator", "model")
    # -> 'NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL'
    ```

### Precedence

For every variable, the first source that is set wins:

```
explicit constructor argument
  >  NEGMAS_LLM_<ClassName>_<VAR>        (per negotiator type)
  >  NEGMAS_LLM_<VAR>                    (global)
  >  built-in default
```

For `MODEL` specifically, the built-in default expands to:

```
NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL      (per-provider env override)
  >  OLLAMA_MODEL                        (legacy, ollama only)
  >  the class's own DEFAULT_MODEL       (e.g. OpenAINegotiator -> "gpt-4o")
  >  the hardcoded per-provider fallback table
```

### Coherence rules (so a model never lands on the wrong provider)

A model name is meaningful only for a specific provider, so two guardrails keep
provider and model consistent:

1. **Provider-locked classes ignore the global `NEGMAS_LLM_PROVIDER`.** The
   provider-named convenience classes (`OpenAINegotiator`,
   `AnthropicNegotiator`, `OllamaNegotiator`, `OpenAIOfferingPolicy`, …) have
   their provider baked into their identity. Setting `NEGMAS_LLM_PROVIDER=groq`
   will **not** turn an `OpenAINegotiator` into a Groq negotiator. To change one
   of these, use a per-type override such as
   `NEGMAS_LLM_OpenAINegotiator_PROVIDER=azure`.

2. **The global `NEGMAS_LLM_MODEL` applies only when the provider matches.** A
   global model is applied to a negotiator only if that negotiator's effective
   provider equals the global provider (`NEGMAS_LLM_PROVIDER`, or the default
   `ollama`). If a per-type/explicit override put the negotiator on a *different*
   provider, its model falls back to *that provider's* default instead of a
   global model string meant for another provider. A per-type
   `NEGMAS_LLM_<ClassName>_MODEL` is always honored (you named the class).

The default provider, when nothing selects one, is `ollama`.

### Worked example: everyone on model A, one negotiator on model B

Run an experiment where **every** negotiator uses OpenAI `gpt-4o`, except
`LLMBoulwareTBNegotiator`, which should use Anthropic `claude-3-opus`:

```bash
# Global default: everyone on openai/gpt-4o
export NEGMAS_LLM_PROVIDER=openai
export NEGMAS_LLM_MODEL=gpt-4o

# Override just one negotiator type (different model AND provider)
export NEGMAS_LLM_LLMBoulwareTBNegotiator_PROVIDER=anthropic
export NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL=claude-3-opus
```

```python
from negmas_llm import LLMBoulwareTBNegotiator, LLMConcederTBNegotiator

LLMBoulwareTBNegotiator(name="a")   # -> anthropic / claude-3-opus
LLMConcederTBNegotiator(name="b")   # -> openai / gpt-4o
```

Because `LLMBoulwareTBNegotiator` was moved to `anthropic`, it does **not** pick
up the global `gpt-4o`; its per-type `MODEL` (`claude-3-opus`) is used instead.
If you omit `NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL`, it would use Anthropic's
default model rather than `gpt-4o`.

### Overriding other parameters per type

The same scheme works for the other knobs. For example, give one negotiator a
larger token budget and a longer timeout while leaving the rest alone:

```bash
export NEGMAS_LLM_LLMBoulwareTBNegotiator_MAX_TOKENS=8192
export NEGMAS_LLM_LLMBoulwareTBNegotiator_TEMPERATURE=0.2
export NEGMAS_LLM_LLMBoulwareTBNegotiator_TIMEOUT=120
export NEGMAS_LLM_LLMBoulwareTBNegotiator_NUM_RETRIES=3
```

Explicit constructor arguments always win over every environment variable, so
code that passes `model=...` deliberately is never overridden.

## Model types (tiers)

A *model type* is a named `(provider, model, effort)` bundle — for example
`fast`, `accurate`, or `friendly` — so that **different parts of the same
negotiator can use different models** (a cheap model for routine text, a strong
model for the critical decision, and so on).

When code requests a model type, the (uppercased) type name is **appended** as a
suffix to every environment variable involved:

```bash
# Define two tiers globally
export NEGMAS_LLM_PROVIDER_FAST=groq
export NEGMAS_LLM_MODEL_FAST=llama-3.1-8b-instant
export NEGMAS_LLM_EFFORT_FAST=low

export NEGMAS_LLM_PROVIDER_ACCURATE=openai
export NEGMAS_LLM_MODEL_ACCURATE=o3
export NEGMAS_LLM_EFFORT_ACCURATE=high
```

A part of the code asks for a tier by passing `model_type=` — either to
`resolve_llm_config`, or (per call) to a negotiator's/component's `_call_llm`:

```python
from negmas_llm import resolve_llm_config

resolve_llm_config("LLMBoulwareTBNegotiator", model_type="fast")
# -> provider="groq", model="llama-3.1-8b-instant", effort="low"

resolve_llm_config("LLMBoulwareTBNegotiator", model_type="accurate")
# -> provider="openai", model="o3", effort="high"
```

Type names must be **identifier-safe** (letters, digits, underscores) since they
become part of an environment variable name. Every tier variable exists in both
the global and per-negotiator-type forms:

| Scope | Base variable | Tier variable (`fast`) |
|-------|---------------|------------------------|
| Global | `NEGMAS_LLM_MODEL` | `NEGMAS_LLM_MODEL_FAST` |
| Global | `NEGMAS_LLM_PROVIDER` | `NEGMAS_LLM_PROVIDER_FAST` |
| Global | `NEGMAS_LLM_EFFORT` | `NEGMAS_LLM_EFFORT_FAST` |
| Per type | `NEGMAS_LLM_<ClassName>_MODEL` | `NEGMAS_LLM_<ClassName>_MODEL_FAST` |

### Tier precedence

With a requested tier `<T>`, each variable resolves (highest first):

```
explicit argument
  >  NEGMAS_LLM_<ClassName>_<VAR>_<T>       (per-type, tier)
  >  NEGMAS_LLM_<ClassName>_<VAR>           (per-type, base)
  >  NEGMAS_LLM_<VAR>_<T>                   (global, tier)
  >  NEGMAS_LLM_<VAR>                       (global, base)
  >  built-in default
```

The ordering is **scope-dominant**: the class scope outranks the tier qualifier,
so a per-type base (`NEGMAS_LLM_<ClassName>_MODEL`) beats a global tier
(`NEGMAS_LLM_MODEL_FAST`). Set the fully-qualified
`NEGMAS_LLM_<ClassName>_MODEL_<T>` for unambiguous per-type-per-tier control. A
tier with nothing configured falls back to the negotiator's base configuration —
its own construction-time settings and the un-suffixed variables — so asking for
`fast` when no `fast` variables are set behaves exactly like that negotiator's
default.

The same coherence rules apply per tier: `NEGMAS_LLM_MODEL_FAST` lands only on
negotiators whose (tier-resolved) provider matches `NEGMAS_LLM_PROVIDER_FAST` (→
`NEGMAS_LLM_PROVIDER` → `ollama`), and provider-locked classes ignore the global
tier provider too. `EFFORT` has no coherence guard — it is not provider-coupled.

## Per-provider default models

You can set the default model for each provider. These are used whenever that
provider is selected and no more specific value (explicit argument, per-type, or
matching global) applies.

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

!!! note "Per-provider vs. per-type vs. global"
    These three axes are independent. `NEGMAS_LLM_OPENAI_DEFAULT_MODEL` changes
    the OpenAI model *for any negotiator that ends up on OpenAI*.
    `NEGMAS_LLM_MODEL` changes the model *for negotiators on the global
    provider*. `NEGMAS_LLM_<ClassName>_MODEL` changes the model *for one
    negotiator type only*.

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

For backwards compatibility, `OLLAMA_MODEL` is also supported for the Ollama
provider. The new `NEGMAS_LLM_OLLAMA_DEFAULT_MODEL` takes precedence if both are
set.

## API Keys

Each provider has its own environment variable for API keys. These are handled
by [litellm](https://github.com/BerriAI/litellm) and follow its conventions.

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

In addition, `negmas-llm` accepts `NEGMAS_LLM_API_KEY` (global) and
`NEGMAS_LLM_<ClassName>_API_KEY` (per type) for cases where you want to route a
key through the resolver rather than litellm's provider-specific variables.

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

You can configure timeout, retries, and the model-dependent token/temperature
defaults for all LLM calls. Each of these also supports the per-type form
(`NEGMAS_LLM_<ClassName>_<VAR>`).

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `NEGMAS_LLM_TIMEOUT` | Timeout in seconds for LLM calls | None (no timeout) |
| `NEGMAS_LLM_NUM_RETRIES` | Number of retries for failed LLM calls | None (no retries) |
| `NEGMAS_LLM_MAX_TOKENS` | Output-token budget for every model (alias: `NEGMAS_LLM_DEFAULT_MAX_TOKENS`) | model-dependent |
| `NEGMAS_LLM_TEMPERATURE` | Sampling temperature for every model (alias: `NEGMAS_LLM_DEFAULT_TEMPERATURE`) | `0.7` (omitted for models that reject it) |
| `NEGMAS_LLM_EFFORT` | Reasoning effort sent verbatim as `reasoning_effort` (e.g. `low`/`medium`/`high`) | None (omitted) |

!!! note "Model-dependent token/temperature defaults"
    When `max_tokens`/`temperature` are left unset, `negmas-llm` picks a
    model-appropriate value at call time: reasoning/thinking models get a larger
    token budget so hidden deliberation cannot starve the visible response, and
    OpenAI reasoning models (o-series, gpt-5) omit `temperature` entirely because
    they reject non-default values. `NEGMAS_LLM_MAX_TOKENS` /
    `NEGMAS_LLM_TEMPERATURE` override this for every model.

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

## Ollama server location

`OLLAMA_HOST` (the same variable `ollama serve` reads) selects where the Ollama
client connects. Accepts `host:port`, a bare `host`, or a full URL. When an
Ollama API key is available (`OLLAMA_API_KEY`), the Cloud/Turbo endpoint
(`https://ollama.com`) is used instead.

See the [litellm documentation](https://docs.litellm.ai/) for more details on
provider-specific environment variables.

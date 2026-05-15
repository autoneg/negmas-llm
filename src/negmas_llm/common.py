import os
from typing import Any


def _get_default_model(provider: str, fallback: str) -> str:
    """Get default model for a provider from environment variable or fallback.

    Checks NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL environment variable.
    For ollama, also checks legacy OLLAMA_MODEL for backwards compatibility.

    Args:
        provider: Provider name (e.g., "ollama", "openai")
        fallback: Fallback model name if env var is not set

    Returns:
        The model name from environment or fallback value
    """
    env_var = f"NEGMAS_LLM_{provider.upper()}_DEFAULT_MODEL"
    model = os.environ.get(env_var)
    if model:
        return model
    # Legacy support for OLLAMA_MODEL
    if provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL")
        if model:
            return model
    return fallback


def resolve_ollama_api_base(default: str = "http://localhost:11434") -> str:
    """Resolve the Ollama API base URL from the ``OLLAMA_HOST`` env var.

    ``ollama serve`` itself reads ``OLLAMA_HOST`` to choose where to listen
    (accepts ``host:port``, bare ``host``, or a full URL), so deriving the
    client URL from the same variable keeps client and server in sync.

    Args:
        default: URL to return when ``OLLAMA_HOST`` is empty or unset.

    Returns:
        A full ``http(s)://host:port`` URL.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return default
    if "://" in host:
        return host
    if ":" not in host:
        host = f"{host}:11434"
    return f"http://{host}"


DEFAULT_MODELS: dict[str, str] = {
    "ollama": _get_default_model("ollama", "qwen3:4b-instruct"),
    "openai": _get_default_model("openai", "gpt-4o-mini"),
    "anthropic": _get_default_model("anthropic", "claude-sonnet-4-20250514"),
    "gemini": _get_default_model("gemini", "gemini-2.0-flash"),
    "github_copilot": _get_default_model(
        "github_copilot", "gpt-4o"
    ),  # GitHub Copilot (OAuth device flow)
    "github": _get_default_model("github", "gpt-4o"),  # GitHub Models marketplace
    "groq": _get_default_model("groq", "llama-3.3-70b-versatile"),
    "mistral": _get_default_model("mistral", "mistral-large-latest"),
    "deepseek": _get_default_model("deepseek", "deepseek-chat"),
    "openrouter": _get_default_model("openrouter", "openai/gpt-4o-mini"),
    "together_ai": _get_default_model(
        "together_ai", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    ),
    "cohere": _get_default_model("cohere", "command-r-plus"),
    "huggingface": _get_default_model(
        "huggingface", "meta-llama/Llama-3.2-3B-Instruct"
    ),
}


# =============================================================================
# max_tokens parameter handling per provider/model
# =============================================================================

# Aliases that mean "max output tokens" across providers. If the user supplies
# any of these via llm_kwargs (or a per-call override), we treat the token
# budget as already specified and do not inject our own.
TOKEN_LIMIT_ALIASES: frozenset[str] = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",  # OpenAI reasoning models (o1/o3/o4, gpt-5, ...)
        "num_predict",  # Ollama
        "max_new_tokens",  # HuggingFace text-generation
        "maxOutputTokens",  # Google Vertex AI (gemini)
    }
)

# OpenAI / Azure model name prefixes that require `max_completion_tokens`
# instead of the legacy `max_tokens`.
_OPENAI_REASONING_PREFIXES: tuple[str, ...] = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def _is_openai_reasoning_model(model: str) -> bool:
    m = model.lower()
    # Strip optional provider prefix like "openai/o3-mini".
    if "/" in m:
        m = m.split("/", 1)[1]
    return any(m == p or m.startswith(p + "-") for p in _OPENAI_REASONING_PREFIXES)


def max_tokens_param_name(provider: str, model: str) -> str:
    """Return the kwarg name the given provider/model expects for output cap.

    Defaults to `max_tokens` for providers that accept it (Anthropic, Gemini,
    Mistral, Groq, Cohere, etc. — litellm handles further normalization).
    Returns `num_predict` for Ollama and `max_completion_tokens` for OpenAI /
    Azure reasoning models.
    """
    p = provider.lower()
    if p == "ollama":
        return "num_predict"
    if p in ("openai", "azure") and _is_openai_reasoning_model(model):
        return "max_completion_tokens"
    return "max_tokens"


def litellm_model_string(provider: str, model: str) -> str:
    """Build the model string passed to ``litellm.completion``.

    For Ollama we route through ``ollama_chat/<model>`` instead of
    ``ollama/<model>``. The ``ollama`` provider in litellm calls Ollama's
    ``/api/generate`` endpoint (single-shot prompt; messages get
    concatenated and the model's chat template may not apply). The
    ``ollama_chat`` provider calls ``/api/chat``: native chat templating,
    proper role-based message arrays, the loaded model stays warm via
    Ollama's keep_alive, and re-tokenisation is incremental — making
    multi-turn negotiations both faster and more reliable while preserving
    context. The public ``self.provider`` value stays ``"ollama"``; only
    the litellm model identifier is rewritten.
    """
    if provider.lower() == "ollama":
        return f"ollama_chat/{model}"
    return f"{provider}/{model}"


def apply_max_tokens(
    kwargs: dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int | None,
) -> None:
    """Inject the token cap into `kwargs` under the correct provider-specific name.

    Skips injection if the user already supplied any known token-limit alias
    (e.g., passed `num_predict` directly in `llm_kwargs`) — that override wins.
    Pass `max_tokens=None` to skip entirely.
    """
    if max_tokens is None:
        return
    if any(alias in kwargs for alias in TOKEN_LIMIT_ALIASES):
        return
    kwargs[max_tokens_param_name(provider, model)] = max_tokens

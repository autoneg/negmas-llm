import os


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

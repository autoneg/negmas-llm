import os

DEFAULT_MODELS: dict[str, str] = {
    "ollama": os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct"),
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "github_copilot": "gpt-4o",  # GitHub Copilot (OAuth device flow)
    "github": "gpt-4o",  # GitHub Models marketplace
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "openai/gpt-4o-mini",
    "together_ai": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "cohere": "command-r-plus",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
}

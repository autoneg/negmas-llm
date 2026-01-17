"""LLM-based negotiators for the negmas framework."""

from negmas.registry import component_registry, negotiator_registry

from negmas_llm.components import (
    # Provider-specific convenience classes
    AnthropicAcceptancePolicy,
    AnthropicOfferingPolicy,
    # Base classes
    LLMAcceptancePolicy,
    LLMComponentMixin,
    LLMNegotiationSupporter,
    LLMOfferingPolicy,
    LLMValidator,
    OllamaAcceptancePolicy,
    OllamaOfferingPolicy,
    OpenAIAcceptancePolicy,
    OpenAIOfferingPolicy,
)
from negmas_llm.negotiator import (
    AnthropicNegotiator,
    AWSBedrockNegotiator,
    AzureOpenAINegotiator,
    CohereNegotiator,
    DeepSeekNegotiator,
    GeminiNegotiator,
    GroqNegotiator,
    HuggingFaceNegotiator,
    LLMNegotiator,
    LMStudioNegotiator,
    MistralNegotiator,
    OllamaNegotiator,
    OpenAINegotiator,
    OpenRouterNegotiator,
    TextGenWebUINegotiator,
    TogetherAINegotiator,
    VLLMNegotiator,
)

# =============================================================================
# Register negotiators with negmas registry
# =============================================================================

_NEGOTIATOR_SOURCE = "negmas-llm"

# Base LLM negotiator (abstract, but register for discovery)
negotiator_registry.register(
    LLMNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao"},
)

# Cloud provider negotiators
negotiator_registry.register(
    OpenAINegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "openai", "cloud"},
)

negotiator_registry.register(
    AnthropicNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "anthropic", "cloud"},
)

negotiator_registry.register(
    GeminiNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "gemini", "google", "cloud"},
)

negotiator_registry.register(
    CohereNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "cohere", "cloud"},
)

negotiator_registry.register(
    MistralNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "mistral", "cloud"},
)

negotiator_registry.register(
    GroqNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "groq", "cloud"},
)

negotiator_registry.register(
    TogetherAINegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "together-ai", "cloud"},
)

negotiator_registry.register(
    AzureOpenAINegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "azure", "openai", "cloud"},
)

negotiator_registry.register(
    AWSBedrockNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "aws", "bedrock", "cloud"},
)

negotiator_registry.register(
    OpenRouterNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "openrouter", "cloud"},
)

negotiator_registry.register(
    DeepSeekNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "deepseek", "cloud"},
)

negotiator_registry.register(
    HuggingFaceNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "huggingface", "cloud"},
)

# Local/Open-source negotiators
negotiator_registry.register(
    OllamaNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "ollama", "local"},
)

negotiator_registry.register(
    VLLMNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "vllm", "local"},
)

negotiator_registry.register(
    LMStudioNegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "lm-studio", "local"},
)

negotiator_registry.register(
    TextGenWebUINegotiator,
    source=_NEGOTIATOR_SOURCE,
    tags={"llm", "sao", "text-gen-webui", "oobabooga", "local"},
)

# =============================================================================
# Register components with negmas registry
# =============================================================================

_COMPONENT_SOURCE = "negmas-llm"

# Base LLM components
component_registry.register(
    LLMAcceptancePolicy,
    source=_COMPONENT_SOURCE,
    component_type="acceptance",
    tags={"llm", "acceptance"},
)

component_registry.register(
    LLMOfferingPolicy,
    source=_COMPONENT_SOURCE,
    component_type="offering",
    tags={"llm", "offering"},
)

component_registry.register(
    LLMNegotiationSupporter,
    source=_COMPONENT_SOURCE,
    component_type="supporter",
    tags={"llm", "supporter"},
)

component_registry.register(
    LLMValidator,
    source=_COMPONENT_SOURCE,
    component_type="validator",
    tags={"llm", "validator"},
)

# Provider-specific acceptance policies
component_registry.register(
    OpenAIAcceptancePolicy,
    source=_COMPONENT_SOURCE,
    component_type="acceptance",
    tags={"llm", "acceptance", "openai"},
)

component_registry.register(
    OllamaAcceptancePolicy,
    source=_COMPONENT_SOURCE,
    component_type="acceptance",
    tags={"llm", "acceptance", "ollama", "local"},
)

component_registry.register(
    AnthropicAcceptancePolicy,
    source=_COMPONENT_SOURCE,
    component_type="acceptance",
    tags={"llm", "acceptance", "anthropic"},
)

# Provider-specific offering policies
component_registry.register(
    OpenAIOfferingPolicy,
    source=_COMPONENT_SOURCE,
    component_type="offering",
    tags={"llm", "offering", "openai"},
)

component_registry.register(
    OllamaOfferingPolicy,
    source=_COMPONENT_SOURCE,
    component_type="offering",
    tags={"llm", "offering", "ollama", "local"},
)

component_registry.register(
    AnthropicOfferingPolicy,
    source=_COMPONENT_SOURCE,
    component_type="offering",
    tags={"llm", "offering", "anthropic"},
)

__all__ = [
    # Base negotiator class
    "LLMNegotiator",
    # Cloud provider negotiators
    "OpenAINegotiator",
    "AnthropicNegotiator",
    "GeminiNegotiator",
    "CohereNegotiator",
    "MistralNegotiator",
    "GroqNegotiator",
    "TogetherAINegotiator",
    "AzureOpenAINegotiator",
    "AWSBedrockNegotiator",
    "OpenRouterNegotiator",
    "DeepSeekNegotiator",
    "HuggingFaceNegotiator",
    # Local/Open-source negotiators
    "OllamaNegotiator",
    "VLLMNegotiator",
    "LMStudioNegotiator",
    "TextGenWebUINegotiator",
    # Components - Base classes
    "LLMComponentMixin",
    "LLMAcceptancePolicy",
    "LLMOfferingPolicy",
    "LLMNegotiationSupporter",
    "LLMValidator",
    # Components - Provider convenience classes
    "OpenAIAcceptancePolicy",
    "OpenAIOfferingPolicy",
    "OllamaAcceptancePolicy",
    "OllamaOfferingPolicy",
    "AnthropicAcceptancePolicy",
    "AnthropicOfferingPolicy",
]

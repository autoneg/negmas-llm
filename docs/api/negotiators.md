# API Reference

## Base Class

::: negmas_llm.LLMNegotiator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - get_model_string
        - format_outcome_space
        - format_own_ufun
        - format_partner_ufun
        - format_nmi_info
        - format_state
        - format_response_instructions
        - build_system_prompt
        - build_user_message
        - counter
        - propose
        - respond
        - on_negotiation_start

## Cloud Providers

### OpenAI

::: negmas_llm.OpenAINegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Anthropic

::: negmas_llm.AnthropicNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Google Gemini

::: negmas_llm.GeminiNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Cohere

::: negmas_llm.CohereNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Mistral

::: negmas_llm.MistralNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Groq

::: negmas_llm.GroqNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Together AI

::: negmas_llm.TogetherAINegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Azure OpenAI

::: negmas_llm.AzureOpenAINegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### AWS Bedrock

::: negmas_llm.AWSBedrockNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### OpenRouter

::: negmas_llm.OpenRouterNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### DeepSeek

::: negmas_llm.DeepSeekNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Hugging Face

::: negmas_llm.HuggingFaceNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

## Local Providers

### Ollama

::: negmas_llm.OllamaNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### vLLM

::: negmas_llm.VLLMNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### LM Studio

::: negmas_llm.LMStudioNegotiator
    options:
      show_root_heading: true
      members:
        - __init__

### Text Generation WebUI

::: negmas_llm.TextGenWebUINegotiator
    options:
      show_root_heading: true
      members:
        - __init__

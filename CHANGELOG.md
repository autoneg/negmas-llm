# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres (loosely) to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Environment variable support for default models: `NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL`
  (e.g., `NEGMAS_LLM_OLLAMA_DEFAULT_MODEL`, `NEGMAS_LLM_OPENAI_DEFAULT_MODEL`)
- Environment variable support for timeout and retries:
  - `NEGMAS_LLM_TIMEOUT` - Default timeout for LLM calls
  - `NEGMAS_LLM_NUM_RETRIES` - Default number of retries for LLM calls
- `timeout` and `num_retries` parameters for `LLMNegotiator` and `LLMMetaNegotiator`
- Documentation page for environment variables (`docs/guide/environment-variables.md`)
- Tests for environment variable model selection
- Per-call `max_tokens` override on `_call_llm` in `LLMNegotiator`,
  `LLMMetaNegotiator`, and the component classes (`LLMAcceptancePolicy`,
  `LLMOfferingPolicy`, `LLMNegotiationSupporter`, `LLMValidator`)
- `negmas_llm.common.apply_max_tokens` helper that picks the correct
  provider-specific output-cap kwarg (`max_tokens`,
  `max_completion_tokens`, or `num_predict`) and respects any alias the
  user supplies via `llm_kwargs`
- `negmas_llm.common.litellm_model_string` helper that routes Ollama
  through litellm's `ollama_chat` provider so requests hit `/api/chat`
  rather than the stateless `/api/generate`
- Unit tests verifying every provider sends
  `messages=[{role, content}, ...]` to `litellm.completion` (chat
  interface) rather than a flat prompt string
- Opt-in env var `NEGMAS_LLM_GITHUB_COPILOT_TEST` to enable GitHub
  Copilot provider tests when no cached OAuth token is present

### Changed

- Legacy `OLLAMA_MODEL` environment variable is still supported for
  backwards compatibility but `NEGMAS_LLM_OLLAMA_DEFAULT_MODEL` takes
  precedence
- All default prompts and tag-rendered context (`outcome-space`,
  `utility-function`, `opponent-utility-function`, `nmi`,
  `current-state`, history/trace tags) are now natural-language text
  without Markdown markers (`##`, `**`, `- ` bullets). "Key: value"
  lines are rewritten as "Key is value." statements so smaller local
  LLMs parse the context more reliably.
- Default prompts trimmed to reduce token usage on local/offline
  models without changing strategy semantics
- OpenAI/Azure reasoning models (o1, o3, o4, gpt-5 families) now
  correctly receive `max_completion_tokens` instead of the deprecated
  `max_tokens`
- Ruff `E501` (line-too-long) is now ignored so it never blocks
  commits; `ruff format` continues to handle line wrapping automatically
- GitHub Actions bumped to their Node.js 24 majors
  (`actions/checkout@v5`, `astral-sh/setup-uv@v6`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8`)
- Pre-commit hooks updated to current latest stable versions
  (ruff-pre-commit v0.15.12, pre-commit-hooks v6.0.0, blacken-docs 1.20.0)

### Fixed

- Ollama on Linux returning HTTP 503 "unknown option `max_tokens`":
  litellm was calling Ollama's `/api/generate` endpoint with the
  OpenAI-style `max_tokens` kwarg. Requests now go through
  `ollama_chat/<model>` (chat endpoint) and the token cap is sent as
  `num_predict`, which Ollama accepts on all platforms. The public
  `provider="ollama"` attribute is unchanged.
- GitHub Copilot provider tests no longer attempt interactive
  device-flow OAuth (and fail after 3 attempts) when no cached token
  exists locally — the tests now skip cleanly unless credentials are
  available or the new opt-in env var is set

## [0.4.2] - 2026-03-15

### Added

- `HybridWithTextNegotiator` - Hybrid negotiator (HybridOfferingPolicy + ACNext acceptance)
  with template-based text messages (no LLM calls)
- `verbose` parameter to `LLMNegotiator` and `LLMMetaNegotiator` (default `False`) to print
  LLM prompts and responses to stdout with rich formatting and timing information
- Non-LLM template-based negotiators for generating human-readable text without LLM calls:
  - `TemplateBasedAdapterNegotiator` - Base class that wraps any negotiator with template text
  - `BoulwareWithTextNegotiator` - Tough strategy with template text messages
  - `ConcederWithTextNegotiator` - Soft strategy with template text messages
  - `LinearWithTextNegotiator` - Linear concession with template text messages
- Template constants for customizing negotiation messages: `ACCEPTANCE_MESSAGES`,
  `CHANGE_PHRASES`, `COMPARISON_WORDS`, `REJECTION_STARTERS`, `REJECTION_ENDERS`,
  `OPENING_OFFER_STARTERS`, `OPENING_OFFER_ENDERS`
- Strategic negotiation guidance in LLM prompts (start strong, concede slowly, protect reservation value)
- Improved utility function display showing weighted contributions for clearer LLM reasoning

### Fixed

- **CRITICAL BUG**: Removed "wait" as a valid response type to prevent infinite wait loops
  - LLM negotiators now must respond with only: accept, reject, or end
  - Updated JSON schema to exclude "wait" from valid response types
  - Updated prompts to explicitly forbid waiting and clarify required actions
  - When no offer is on the table, negotiators must make a proposal (reject with outcome)

### Changed

- Bumped negmas dependency to >= 0.15.2

## [0.4.1] - 2026-03-09

### Changed

- Centralized default model configuration in `src/negmas_llm/common.py`
- Updated default Ollama model from `qwen3:0.6b` to `qwen3:4b-instruct`
- All LLM-wrapped negotiators now use `DEFAULT_OLLAMA_MODEL` from centralized config

## [0.4.0] - 2026-02-14

### Added

- `include_reasoning` parameter to `LLMNegotiator` (default `False`) to control whether
  LLM reasoning is forwarded to the negotiation partner in `SAOResponse.data`
- Tag discovery and documentation functions:
  - `get_available_tags()` - Returns list of all available tag names
  - `get_tag_documentation(tag_name)` - Get markdown documentation for a specific tag
  - `print_tag_help(tag_name=None)` - Print formatted help for all tags or a specific tag

### Changed

- Refactored `meta.py`: replaced factory pattern with direct class definitions for all
  23 LLM-wrapped negotiator classes, improving code clarity and IDE support
- Removed obsolete `SAOMetaNegotiator` availability checks (negmas >= 0.15.1 is required)
- `is_meta_negotiator_available()` now always returns `True` (kept for backwards compatibility)

## [0.3.0] - 2026-02-13

### Added

- Tag processing system for dynamic prompt templating with `{{tag_name:format(params)}}` syntax
- Supported tags: `outcome-space`, `utility-function`, `opponent-utility-function`, `nmi`,
  `current-state`, `reserved-value`, `opponent-reserved-value`, `my-last-offer`,
  `my-first-offer`, `opponent-last-offer`, `opponent-first-offer`, `partner-offers`,
  `history`, `trace`, `extended-trace`, `full-trace`, `utility`
- `LLMMetaNegotiator` that wraps native negmas negotiators with LLM text generation
- 23 pre-configured LLM-wrapped negotiator classes:
  - Time-based: `LLMAspirationNegotiator`, `LLMBoulwareTBNegotiator`, `LLMConcederTBNegotiator`,
    `LLMLinearTBNegotiator`, `LLMTimeBasedConcedingNegotiator`, `LLMTimeBasedNegotiator`
  - Nice/Tough: `LLMNiceNegotiator`, `LLMToughNegotiator`
  - Tit-for-tat: `LLMNaiveTitForTatNegotiator`
  - Random: `LLMRandomNegotiator`, `LLMRandomAlwaysAcceptingNegotiator`
  - Curve-based: `LLMCABNegotiator`, `LLMCANNegotiator`, `LLMCARNegotiator`
  - MiCRO: `LLMMiCRONegotiator`, `LLMFastMiCRONegotiator`
  - Utility-based: `LLMUtilBasedNegotiator`
  - WAR/WAN/WAB: `LLMWARNegotiator`, `LLMWANNegotiator`, `LLMWABNegotiator`
  - Limited outcomes: `LLMLimitedOutcomesNegotiator`, `LLMLimitedOutcomesAcceptor`
  - Hybrid: `LLMHybridNegotiator`
- Conversation history support for multi-round context in `LLMNegotiator`
- Structured output/JSON mode support for reliable parsing (OpenAI, Azure, Gemini, Anthropic)
- `process_prompt()` method for tag substitution in prompts
- `on_preferences_changed` and `on_negotiation_start` callbacks
- Configurable LLM provider testing infrastructure

### Changed

- Switched `LLMNegotiator` base class from `SAONegotiator` to `SAOCallNegotiator`
- All prompts now processed through `process_prompt()` for tag substitution
- Bumped negmas dependency to >= 0.15.1 for `SAOMetaNegotiator` support

## [0.2.2] - 2026-01-17

### Changed

- Bumped negmas dependency to >= 0.14.0 for registry module support

## [0.2.1] - 2026-01-17

### Added

- Registration of all negotiators and components with negmas registry system
- Negotiators tagged by provider type (cloud/local) and capabilities
- Components registered with appropriate component types (acceptance, offering, supporter, validator)

## [0.2.0] - 2026-01-09

### Added

- Modular LLM components for `MAPNegotiator` (negmas's modular architecture):
  - `LLMAcceptancePolicy` - LLM-based acceptance decisions
  - `LLMOfferingPolicy` - LLM-based offer generation
  - `LLMNegotiationSupporter` - Generate supporting text for negotiation actions
  - `LLMValidator` - Validate consistency between text and actions
- Provider convenience classes:
  - `OpenAIAcceptancePolicy`, `OpenAIOfferingPolicy`
  - `OllamaAcceptancePolicy`, `OllamaOfferingPolicy`
  - `AnthropicAcceptancePolicy`, `AnthropicOfferingPolicy`
- `LLMComponentMixin` base class for shared LLM functionality

## [0.1.2] - 2026-01-09

### Added

- Python 3.14 classifier in package metadata

## [0.1.1] - 2026-01-09

### Added

- PyPI classifiers, keywords, and project URLs
- Documentation link in README badges

## [0.1.0] - 2026-01-08

### Added

- Initial release of negmas-llm
- Core `LLMNegotiator` class with customizable `build_system_prompt()` and
  `build_user_message()` methods for prompt engineering
- 16 provider-specific negotiator subclasses:
  - Cloud providers: `OpenAINegotiator`, `AnthropicNegotiator`, `GeminiNegotiator`,
    `CohereNegotiator`, `MistralNegotiator`, `GroqNegotiator`, `TogetherAINegotiator`,
    `AzureOpenAINegotiator`, `AWSBedrockNegotiator`, `OpenRouterNegotiator`,
    `DeepSeekNegotiator`, `HuggingFaceNegotiator`
  - Local providers: `OllamaNegotiator`, `VLLMNegotiator`, `LMStudioNegotiator`,
    `TextGenWebUINegotiator`
- Full documentation with mkdocs-material
- GitHub Actions workflows for CI and PyPI publishing
- Pre-commit hooks for code quality (ruff, blacken-docs)
- Python 3.11, 3.12, 3.13, 3.14 support
- AGPL-3.0 license

[Unreleased]: https://github.com/autoneg/negmas-llm/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/autoneg/negmas-llm/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/autoneg/negmas-llm/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/autoneg/negmas-llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/autoneg/negmas-llm/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/autoneg/negmas-llm/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/autoneg/negmas-llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/autoneg/negmas-llm/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/autoneg/negmas-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/autoneg/negmas-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/autoneg/negmas-llm/releases/tag/v0.1.0

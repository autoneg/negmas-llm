# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/autoneg/negmas-llm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/autoneg/negmas-llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/autoneg/negmas-llm/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/autoneg/negmas-llm/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/autoneg/negmas-llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/autoneg/negmas-llm/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/autoneg/negmas-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/autoneg/negmas-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/autoneg/negmas-llm/releases/tag/v0.1.0

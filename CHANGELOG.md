# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres (loosely) to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-02

The headline of this release is **PABLO-ve**, a component architecture that
generalizes BOA, plus an LLM implementation of every slot in it. Requires
negmas >= 0.16.0.

### Added

- **PABLO-ve component architecture** (`negmas_llm.pablove`,
  `negmas_llm.pablove_components`): Perception, Acceptance, Bidding, Language,
  Opponent-modeling, plus optional validation and ending. BOA has no slot for
  understanding what the partner *said*, for deciding to stop, or for producing
  what we say back; these become ordinary negmas components, inheriting
  lifecycle callbacks, ufun/NMI sharing and registry support. Every component is
  optional, which makes the central claim exact and testable: with
  perception/language/validation/ending absent, `PABLOveNegotiator` produces a
  trace identical to `make_boa(...)`. Inter-step data flows through a
  turn-scoped `TurnContext` on the negotiator rather than through component
  signatures, so existing negmas policies keep working unchanged;
  `negotiator.shared` holds cross-turn state. `ctx.their_offer` and `ctx.bid`
  are deliberately separate fields -- negmas' `respond()` passes the partner's
  offer while `propose()` passes ours. Also adds `TemplateLanguage`
  (deterministic verbalization, no LLM -- the control that separates "language
  helps" from "this model's language helps") and `make_pablove`.
- **An LLM implementation of every BOA and PABLO-ve slot**, so a negotiator can
  be assembled entirely from LLM components, entirely from classical ones, or
  any mixture: `LLMOffering`, `LLMAcceptance`, `LLMUFunModel` (a
  text-conditioned opponent model -- unlike a frequency model it can use what
  the partner said, not only what they offered), `LLMPerception`, `LLMLanguage`,
  `LLMValidation`, `LLMEnding`. Each component resolves its configuration on
  its own class name, so components route independently
  (`NEGMAS_LLM_LLMOffering_MODEL=<big>` alongside
  `NEGMAS_LLM_LLMPerception_MODEL=<small>`). Every component enforces the
  invariants the LLM cannot be trusted with and counts each intervention:
  `snap_outcome()` repairs near-miss outcomes rather than discarding them,
  offering/acceptance refuse outcomes at or below the reserved value,
  `LLMEnding` will not walk away from an offer that already beats no deal, and
  `LLMUFunModel` scores 0 before its first estimate. Also adds
  `outcome_space_of()`, since components that consult only the NMI see nothing
  outside a live negotiation.
- Composable PABLO-ve slots: `Perception`, `Language`, `Validation` and
  `Ending` components now support `a | b | c`, producing a `PerceptionChain`,
  `LanguageChain`, `ValidationChain` or `EndingChain` that fills the slot as a
  single component of that kind. Chaining is kind-checked (`perception |
  language` raises `TypeError`), flat and associative, and never mutates its
  operands. Semantics per kind: perceivers all run and their readings are
  merged (acts/commitments unioned in order, first non-empty wins for
  single-valued fields, best `source` reported); language components each
  revise the previous one's draft through `ctx.utterance`; validators must all
  pass and their rewrites compose; ending components stop when any of them says
  so (`mode="all"` requires unanimity). A chain is not itself registered with
  the negotiator -- its leaves are, individually, so each one keeps its own
  lifecycle callbacks, ufun/NMI sharing and `memory_mode` override.
- Every PABLO-ve slot (`acceptance`, `offering`, `model`, `perception`,
  `language`, `validation`, `ending`, on both `PABLOveNegotiator` and
  `make_pablove`) now accepts a **type**, a **class name**, a
  `(spec, params)` pair or a **list** of any of those, in addition to a
  ready-made component -- matching how negmas' own `*_type`/`*_params` pairs
  work. Each slot gained a matching `*_params` argument (`perception_params`,
  `language_params`, ...), which takes one dict for all components or one dict
  per component against a list spec. A list spec builds the same chain `|`
  would, and `A | B` over *types* (which Python evaluates to a union before it
  reaches us) is treated the same way, so the two spellings never disagree.
  New public helpers: `make_component`, `component_leaves`,
  `resolve_component_type`, and the `ComponentSpec` alias documenting the
  grammar.
- `LLMComponent.memory_mode` with three strategies for how a PABLO-ve component
  learns the negotiation's setup and history, selectable per component or
  uniformly via `PABLOveNegotiator(memory_mode=...)`: `"none"` (a fresh,
  memoryless completion every call -- the previous behavior), `"conversation"`
  (the new default: each component keeps its own persistent chat, seeded once
  through the same preferences template and tag renderer `LLMNegotiator` uses)
  and `"shared"` (the setup cached once on `PABLOveNegotiator.memory` and pulled
  fresh into each per-call message). Every mode except `"none"` also gets a
  short "team briefing" in its system prompt, since a component that does not
  know it is one part of a pipeline tends to over-explain or second-guess
  decisions made elsewhere.
- Opt-in conversation summarizer (`negmas_llm.summarize.maybe_summarize`) for
  `LLMNegotiator` and for `LLMComponent`'s `"conversation"` mode, bounding token
  growth in long negotiations: `summarize_every` (exchange count) and
  `summarize_over_chars` (total conversation length) both default to disabled,
  either trigger firing is enough, and everything older than `summarize_keep`
  exchanges collapses into one LLM-generated summary. Never wall-clock time.
  `LLMMetaNegotiator` is not wired up -- it has no persistent conversation to
  summarize.
- In-process ufun tool-calling (`negmas_llm.ufun_tools`): OpenAI/litellm-style
  function-tool specs and a dispatcher covering `evaluate_utility`,
  `utility_min`/`utility_max`, `best_outcome`/`worst_outcome`, `reserved_value`
  and the inverter operations
  `invert_some`/`invert_all`/`invert_one_in`/`invert_best_in`/`invert_worst_in`,
  so the LLM can compute exact utilities instead of estimating them from the
  prompt. No server, transport or MCP layer involved. Available to
  `LLMNegotiator`, `LLMMetaNegotiator` and every PABLO-ve `LLMComponent` through
  `use_ufun_tools`, with one shared call/tool-round-trip loop
  (`ufun_tools.run_llm_call`, capped at `ufun_tools.MAX_TOOL_ROUNDS`).
- MCP server (`negmas_llm.mcp`, optional `[mcp]` extra, `negmas-mcp` entry point)
  exposing any registered negmas or negmas-llm negotiator to an MCP client.
  Tools: `list_negotiators`, `describe_negotiator`, `start_negotiation`,
  `counter`, `negotiation_state`, `end_negotiation`, `run_negotiation`. Sessions
  run a real `SAOMechanism` with the caller driving one side, so time-based
  strategies concede on a genuine schedule rather than against a synthesized
  state. Serves stdio and streamable-http. This is the mirror of
  `negmas_llm.ufun_tools`: there the negotiator's LLM calls tools, here the
  negotiator *is* the tool. Documented in `docs/guide/mcp.md` with a worked
  step-by-step walkthrough in `docs/guide/mcp-tutorial.md`.
- LLM-as-judge scoring (`negmas_llm.judging`): `judge_negotiation_messages`
  scores one party's own messages on `score_leakage` (did its words reveal
  private preference information) and `format_compliance`, via a single
  `litellm.completion` call with a fixed judge system prompt;
  `describe_private_info` formats a ufun's issue weights for that prompt. Takes
  primitive arguments rather than any downstream project's negotiation-context
  type, so this package has no dependency on its consumers.
- Single source of truth for LLM configuration (`negmas_llm.config`): every
  negotiator, meta negotiator and component resolves its
  provider/model/effort/etc. in one place, driven by environment variables so
  experiments can switch models without code changes. Three configuration axes
  (per-provider, global `NEGMAS_LLM_<VAR>`, per-type
  `NEGMAS_LLM_<ClassName>_<VAR>`) with precedence explicit > per-type > global >
  built-in default, and two coherence rules that keep a model off the wrong
  provider. Adds **model types** (tiers): `resolve_llm_config(model_type="fast")`
  appends the uppercased tier as an env-var suffix, so different parts of one
  negotiator can request different (provider, model, effort) bundles; `_call_llm`
  gained a `model_type` parameter. `effort` is plumbed to litellm as
  `reasoning_effort`. Historical defaults are unchanged.
- Per-call token usage and response-time tracking: `TokenUsage`, a cumulative
  accumulator (prompt/completion/total tokens, wall-clock seconds, call count)
  fed from litellm's `response.usage` at every call site. Read it off
  `.token_usage` on any negotiator or component after a negotiation.
- Recommender mode for `LLMMetaNegotiator` via `enforce_base_offer` /
  `enforce_base_response`: when `False`, the base negotiator's output is a
  *recommendation* -- the LLM picks the outcome and response, falling back to
  the base proposal whenever its choice is unparseable or not a valid outcome,
  so agreements stay structurally valid either way. Adds
  `LLMNegotiatorWithRecommender`, which presents several sub-negotiators'
  proposals to the LLM as candidate recommendations.
- Mechanism-limit awareness: `TurnContext` now carries `n_steps`/`time_limit`
  from the NMI alongside `step`/`relative_time`, and a new
  `common.time_status()` states what `relative_time` means (0% = start, 100% =
  deadline) together with the actual step/time limits (treating negmas's
  infinite `time_limit` as unlimited). Sent by default from every PABLO-ve
  component, `LLMMetaNegotiator`, and the older BOA-style components.
- Prompt tags `{{annotation}}` (the mechanism's shared
  `negotiator.nmi.annotation`) and `{{private-annotation}}` (this negotiator's
  own `annotation`/`private_info`, with the internal `opponent_ufun` key
  filtered out so it never leaks). Both are self-contained blocks that render as
  an empty string when there is nothing to show, and both support dotted-path
  access (`{{annotation.<key>}}`).
- `describe_reserved_value()`, the single canonical phrasing of what a reserved
  value *is* -- the payoff on no agreement, not merely a floor to obey -- routed
  through every prompt site: the `{{reserved-value}}` tag, the linear-additive
  ufun renderer, `LLMNegotiator`'s preferences/system/start prompts,
  `format_own_ufun`/`format_partner_ufun`, `LLMComponent`, and the PABLO-ve
  offering/acceptance prompts. On scenarios where every reserved value is 0.0
  this is invisible; with per-party sampled reservations it is the whole game.
- `max_words` (env `NEGMAS_LLM_MAX_WORDS`, default 60) and
  `word_limit_instruction`: response length is now bounded by a word budget
  stated in the prompt rather than by a token cap, because reasoning models
  spend hidden deliberation against the same output budget as the visible answer.
- LLM sampling now takes part in negmas's global reproducibility switch. When
  `NEGMAS_RAND_SEED` is set (or `negmas.helpers.rand.seed_all()` is called),
  every `litellm.completion` this package makes -- negotiators, meta
  negotiators, PABLO-ve components and the leakage judge -- carries that seed
  as its sampling `seed`. With no seed in effect (the default) nothing is sent
  and requests are byte-for-byte what they were. The parameter is injected only
  for providers litellm reports as accepting it (Anthropic and Gemini do not),
  and an explicit `seed` in `llm_kwargs` always wins. Best-effort: providers
  treat a sampling seed as a hint, so this makes a run more repeatable, not
  exactly reproducible.
- `PABLOveNegotiator.opponent_model`, delegating to the `opponent_ufun` the base
  `Negotiator` already exposes, so negmas policies that read a differently-named
  attribute (e.g. `NiceTitForTatOfferingPolicy`) can be dropped into a
  PABLO-ve slot without a silent fallback.
- `LLMLanguage` now includes any existing draft (`ctx.utterance`) in its prompt
  and asks the model to improve it, so it works as a later link in a
  `LanguageChain` without any chain-specific code. A lone `LLMLanguage` always
  sees `None` there, so its behavior is unchanged.
- `negmas_llm.common.resolve_ollama_api_base()` helper exposing the same
  `OLLAMA_HOST` resolution rule for reuse by downstream code.
- Ollama Cloud / Turbo support in `OllamaNegotiator`: when an `api_key` is
  provided (or `OLLAMA_API_KEY` is set) and `api_base` is not pinned, the
  negotiator routes to `https://ollama.com` instead of a local server.
- `DashScopeNegotiator` for Alibaba DashScope (Qwen) models, reading
  `DASHSCOPE_API_KEY` from the environment.
- Pre-commit hooks (the same pre-commit-hooks/ruff/ruff-format/blacken-docs
  suite negmas itself uses) now run on every commit.

### Changed

- **Minimum negmas version is now 0.16.0** (was 0.15.2). 0.16.0 reworked ufun
  inversion (`negmas.preferences.inv_ufun` is now a package of inverters and
  `ufun.invert()` returns an adaptive inverter that picks the best one for the
  ufun), which `negmas_llm.ufun_tools` depends on directly.
- `use_ufun_tools` now defaults to `True` for `LLMNegotiator`, `LLMMetaNegotiator`
  (and every named wrapper/subclass of both), and every PABLO-ve `LLMComponent`
  (`LLMOffering`, `LLMAcceptance`, `LLMPerception`, `LLMLanguage`,
  `LLMUFunModel`, `LLMValidation`, `LLMEnding`). An LLM reasoning about its own
  utility function from the serialized prompt alone is prone to misreading it
  -- e.g. defaulting to a "larger number is better" prior on an ordinal-looking
  issue whose mapping actually runs the other way -- and the tool gives it a
  way to check instead of guess. Pass `use_ufun_tools=False` to opt out for a
  provider/model that does not handle tool-calling reliably.
- `temperature` and `max_tokens` now default to `None` on `LLMNegotiator`,
  `LLMMetaNegotiator` and the components, and a model-appropriate value is
  resolved at call time: reasoning models get a larger budget (8192) so hidden
  deliberation cannot starve the visible response, classic instruct models keep
  1024, and OpenAI reasoning models omit `temperature` entirely (they reject
  non-defaults). Explicit constructor/per-call values and `llm_kwargs` aliases
  still win. Overridable via `NEGMAS_LLM_DEFAULT_MAX_TOKENS` /
  `NEGMAS_LLM_DEFAULT_TEMPERATURE`. Adds `is_reasoning_model`,
  `default_*`/`resolve_*` and `apply_temperature` helpers.
- The truncation/depth limits every PABLO-ve component applies to its prompt
  (`text_limit`, `history_turns`, `history_offers`, `domain_values_limit`) are
  now per-component `attrs` fields on `LLMComponent` rather than hardcoded
  constants, so they can be tuned per negotiator/component.
- `OllamaNegotiator` now derives its default `api_base` from the `OLLAMA_HOST`
  environment variable (host:port, bare host, or full URL), matching how
  `ollama serve` chooses its listen address. Explicit `api_base` arguments
  still take precedence; with no env var set, behavior is unchanged
  (`http://localhost:11434`).
- The standalone "Your reserved value follows." section is gone from the default
  preferences prompt -- `{{utility-function:text}}` already ends with the full
  explanation, and keeping both rendered the same paragraph twice.

### Fixed

- `_format_linear_additive_ufun` renders both the reader's own ufun *and* the
  opponent's but hard-coded first-person wording, so an opponent-model
  negotiator was told "Your reserved value ... NEVER accept or offer anything at
  or below this value" about the **opponent's** number -- substituting the
  opponent's walk-away point for its own. It now takes an `own` flag and
  attributes the value to the right party.
- `LLMMetaNegotiator` never learned its own reserved value at all: it has no
  `on_preferences_changed` hook, so it never rendered the preferences prompt, and
  its LLM saw the concept only in rules telling it to keep the value secret.
  This matters most on the override path (`enforce_base_response=False`), where
  the LLM can accept a deal the base strategy would have refused. Both the
  decision prompts and the text-generation prompt now carry it.
- `{{opponent-utility-function}}` returned the non-empty string "Opponent
  utility function unknown", so the preferences prompt's static "The opponent's
  utility function follows." heading always appeared even with no opponent ufun
  known. The heading now lives inside the tag's own output.
- `PABLOveNegotiator.respond` reported `their_offer`/`their_data` as always
  `None`: `GBState` has no `current_offer` attribute, and under the
  `SAOMechanism` compatibility shim `state.threads` ends up empty entirely, so
  the offer and data are now rescued from the raw `SAOState` before the shim
  drops them. `LLMPerception` had the same broken lookup.
- `TurnContext.history` stored a fresh tuple copy of every prior turn on each
  new turn (O(n) per turn, O(n^2) over a negotiation). It is now a lazy view
  over the negotiator's shared, growing turn list.
- Three of six PABLO-ve components (`LLMOffering`, `LLMAcceptance`,
  `LLMEnding`) surfaced the opponent's offer but not their words, and
  `LLMLanguage`/`LLMUFunModel` dropped the text entirely when no `Perception`
  component was configured. Every component now sees both the opponent's outcome
  and their text.
- `PABLOveNegotiator.opponent_model` returned `self._models[0]` verbatim,
  duplicating the mechanism instead of reusing it and dropping
  `MAPNegotiator`'s own `UFunModel` type-filtering; it now delegates to
  `opponent_ufun`.
- `summarize_every`/`summarize_over_chars` used a `bool(...)` disable check,
  silently conflating a `0` threshold with "disabled".
- `LLMMetaNegotiator._build_system_prompt` now returns a custom system prompt
  untouched, with the word-budget instruction appended at LLM-call time (matching
  `LLMNegotiator`).
- `negmas_llm.ufun_tools` against negmas >= 0.16.0, whose inverters *clamp*
  (returning the nearest achievable outcome for a range that holds none) and
  whose default inverter no longer implements `all()`. `invert_all` falls back to
  the equivalent strict scan when `all()` is absent, and
  `invert_one_in`/`invert_best_in`/`invert_worst_in` keep their documented
  contract of a null answer for an empty range.
- A ufun tool for the reserved value: none of the existing tools exposed it, yet
  `invert_some`'s `normalized` flag is documented as being relative to it.


## [0.4.3] - 2026-05-14

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
- Handle dict-shaped outcomes returned by some LLM responses in
  `components.py` and `negotiator.py` (normalised to issue-order tuples;
  invalid values surface a warning unless `raise_on_parsing_error`)

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

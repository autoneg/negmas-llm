"""Meta-negotiator that adds LLM-generated text to any base negotiator's offers."""

from __future__ import annotations

import json
import re
import textwrap
import time
import warnings
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm import ModelResponse
from negmas.gb.common import ExtendedResponseType
from negmas.inout import serialize
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOState
from negmas.sao.negotiators.meta import SAOMetaNegotiator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from negmas_llm.common import (
    DEFAULT_MODELS,
    apply_effort,
    apply_max_tokens,
    apply_temperature,
    litellm_model_string,
    resolve_max_words,
    time_status,
    word_limit_instruction,
)
from negmas_llm.config import (
    DEFAULT_PROVIDER,
    effective_llm_config,
    resolve_llm_config,
)
from negmas_llm.tags import process_prompt
from negmas_llm.token_usage import TokenUsage
from negmas_llm.ufun_tools import (
    MAX_TOOL_ROUNDS,
    UFUN_TOOL_SPECS,
    assistant_tool_call_entry,
    tool_result_messages,
)

DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS.get("ollama", "qwen3:4b-instruct")

if TYPE_CHECKING:
    from litellm.types.utils import Choices

__all__ = [
    "LLMMetaNegotiator",
    "is_meta_negotiator_available",
    # Recommender-based meta negotiators
    "LLMNegotiatorWithRecommender",
    "LLMEnhancedNegotiator",
    "LLMNegotiatorWithMultipleRecommenders",
    # LLM-wrapped native negotiators
    "LLMAspirationNegotiator",
    "LLMBoulwareTBNegotiator",
    "LLMConcederTBNegotiator",
    "LLMLinearTBNegotiator",
    "LLMTimeBasedConcedingNegotiator",
    "LLMTimeBasedNegotiator",
    "LLMNiceNegotiator",
    "LLMToughNegotiator",
    "LLMNaiveTitForTatNegotiator",
    "LLMRandomNegotiator",
    "LLMRandomAlwaysAcceptingNegotiator",
    "LLMCABNegotiator",
    "LLMCANNegotiator",
    "LLMCARNegotiator",
    "LLMMiCRONegotiator",
    "LLMFastMiCRONegotiator",
    "LLMUtilBasedNegotiator",
    "LLMWARNegotiator",
    "LLMWANNegotiator",
    "LLMWABNegotiator",
    "LLMLimitedOutcomesNegotiator",
    "LLMLimitedOutcomesAcceptor",
    "LLMHybridNegotiator",
]


def _dedent(text: str) -> str:
    """Dedent a multi-line string, stripping the first line if empty."""
    if text.startswith("\n"):
        text = text[1:]
    return textwrap.dedent(text)


def is_meta_negotiator_available() -> bool:
    """Check if LLMMetaNegotiator is available.

    This function exists for backwards compatibility. SAOMetaNegotiator
    is now always assumed to be available (requires negmas >= 0.15.1).

    Returns:
        Always returns True.
    """
    return True


class LLMMetaNegotiator(SAOMetaNegotiator):
    """A meta-negotiator that wraps a base negotiator and adds LLM-generated text.

    This negotiator inherits from `SAOMetaNegotiator` and manages a single base
    negotiator as its child. It delegates the core negotiation strategy (what
    offers to make, when to accept) to the base negotiator, while using an LLM
    to generate persuasive or explanatory text to accompany each offer.

    The base negotiator receives all lifecycle callbacks (on_negotiation_start,
    on_round_start, etc.) through the standard `SAOMetaNegotiator` mechanism,
    ensuring it functions correctly.

    This allows combining proven negotiation strategies with natural language
    capabilities, enabling human-like communication without changing the
    underlying negotiation logic.

    Provider, model, and the other LLM settings are resolved in one place (see
    ``negmas_llm.config``): an explicit argument wins, otherwise the value comes
    from ``NEGMAS_LLM_<ClassName>_<VAR>`` (per negotiator type), then
    ``NEGMAS_LLM_<VAR>`` (global), then the built-in default. See
    ``docs/guide/environment-variables.md``.

    Args:
        base_negotiator: The negotiator that handles the core negotiation logic.
            This negotiator's propose/respond methods determine the actual offers
            and acceptance decisions.
        provider: The LLM provider (e.g., "openai", "anthropic", "ollama").
            None (default) resolves from the environment/defaults.
        model: The model name (e.g., "gpt-4", "claude-3-opus"). None (default)
            resolves from the environment/defaults.
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Hard ceiling on tokens the model may spend. None (default)
            sends no cap at all, so hidden reasoning can never starve the
            visible response. Use ``max_words`` to bound answer length.
        max_words: Approximate word budget for the generated message, stated in
            the prompt. None (default) uses
            :data:`negmas_llm.common.DEFAULT_MAX_WORDS`; 0 removes the limit.
        enforce_base_offer: When True (default) the structured outcome of every
            proposal is taken verbatim from the base negotiator and the LLM only
            fills ``data["text"]``. When False the LLM may override the outcome,
            treating the base proposal as a recommendation; if the LLM's outcome
            is invalid or unparseable it falls back to the base proposal.
        enforce_base_response: When True (default) the accept/reject/end decision
            is taken verbatim from the base negotiator and the LLM only fills
            ``data["text"]``. When False the LLM may override the decision,
            treating the base response as a recommendation; if the LLM's response
            is unparseable it falls back to the base response.
        verbose: If True, print LLM prompts and responses to stdout. Useful for
            debugging and understanding the LLM's text generation process.
            Default is False.
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process (evaluate an outcome, min/max,
            best/worst, and the inverter operations some/all/one_in/best_in/
            worst_in -- see :mod:`negmas_llm.ufun_tools`), instead of leaving
            it to estimate utilities from the prompt. Applies to every LLM
            call this negotiator makes (text generation and, when
            ``enforce_base_offer``/``enforce_base_response`` is False, the
            outcome/response decision). Default is True: an LLM reasoning
            about its own utility function from the serialized prompt alone
            is prone to misreading it, and the tool gives it a way to check
            instead of guess. Set False for a provider/model that does not
            handle tool-calling reliably; it also requires ``self.ufun`` to
            be set to have any effect. ``share_ufun=True``
            (the default) propagates *this* negotiator's own ufun DOWN to the
            base negotiator on join -- not the other way around -- so give
            the ufun to this constructor (``ufun=``/``preferences=``, which
            lands in ``**kwargs``) or via ``mechanism.add(negotiator,
            ufun=...)``; a ufun given only to ``base_negotiator`` does not
            reach ``self.ufun`` and tools will not fire. Tool calls are not
            added to the stored conversation history, so they happen fresh on
            every call.
        system_prompt: Custom system prompt for text generation.
            If not provided, a default prompt focused on generating
            persuasive negotiation messages is used.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to SAOMetaNegotiator.

    Example:
        >>> from negmas.sao import BoulwareTBNegotiator
        >>> from negmas_llm import LLMMetaNegotiator
        >>> base = BoulwareTBNegotiator()
        >>> meta = LLMMetaNegotiator(
        ...     base_negotiator=base,
        ...     provider="openai",
        ...     model="gpt-4o",
        ... )
        >>> # meta will use BoulwareTBNegotiator's strategy but add
        >>> # LLM-generated text to each offer

    See Also:
        :class:`LLMNegotiator`: A negotiator where the LLM controls both
            strategy and text generation.
        :class:`negmas.sao.negotiators.meta.SAOMetaNegotiator`: The base class
            for meta-negotiators in SAO protocols.
    """

    #: Built-in fallback provider for this class (see negmas_llm.config). The
    #: strategy subclasses are provider-agnostic, so this defaults to None and
    #: resolves to the global default provider ("ollama").
    DEFAULT_PROVIDER: str | None = None
    #: Built-in fallback model. None defers to the per-provider default table.
    DEFAULT_MODEL: str | None = None
    #: When True the global NEGMAS_LLM_PROVIDER is ignored for this class.
    LOCK_PROVIDER: bool = False

    def __init__(
        self,
        base_negotiator: SAONegotiator,
        provider: str | None = None,
        model: str | None = None,
        *,
        effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        timeout: float | int | None = None,
        num_retries: int | None = None,
        enforce_base_offer: bool = True,
        enforce_base_response: bool = True,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize with the base negotiator as our single child
        super().__init__(
            negotiators=[base_negotiator],
            negotiator_names=["base"],
            share_ufun=True,
            share_nmi=True,
            **kwargs,
        )
        # Single source of truth: resolve provider/model/etc. from explicit
        # arguments, per-type and global environment variables, and this class's
        # built-in fallbacks. See negmas_llm.config for the precedence rules.
        resolved = resolve_llm_config(
            type(self).__name__,
            provider=provider,
            model=model,
            effort=effort,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_words=max_words,
            timeout=timeout,
            num_retries=num_retries,
            default_provider=self.DEFAULT_PROVIDER or DEFAULT_PROVIDER,
            default_model=self.DEFAULT_MODEL,
            lock_provider=self.LOCK_PROVIDER,
        )
        self._store_llm_config(
            resolved,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )
        self.enforce_base_offer = enforce_base_offer
        self.enforce_base_response = enforce_base_response
        self.token_usage = TokenUsage()

        # Track received messages for context
        self._received_messages: list[dict[str, Any]] = []

    def _store_llm_config(
        self,
        resolved: Any,
        *,
        verbose: bool,
        use_ufun_tools: bool = True,
        system_prompt: str | None,
        llm_kwargs: dict[str, Any] | None,
    ) -> None:
        """Store resolved LLM configuration on this negotiator.

        Factored out of :meth:`__init__` so subclasses that register their
        sub-negotiators differently (e.g. multiple recommenders) can still reuse
        the same configuration wiring without re-running the single-base
        constructor path.

        Args:
            resolved: The resolved LLM config from :func:`resolve_llm_config`.
            verbose: If True, print LLM prompts and responses to stdout.
            use_ufun_tools: If True, offer the LLM function tools that compute
                its own utility function in-process (see
                :mod:`negmas_llm.ufun_tools`) instead of estimating it from the
                prompt. Requires ``self.ufun`` to be set on *this* negotiator
                (``share_ufun=True``, the default, then propagates it down to
                the base negotiator on join -- not the other way around).
            system_prompt: Custom system prompt for text generation (if any).
            llm_kwargs: Additional keyword arguments passed to litellm.completion.
        """
        self.provider = resolved.provider
        self.model = resolved.model
        self.effort = resolved.effort
        self.api_key = resolved.api_key
        self.api_base = resolved.api_base
        self.temperature = resolved.temperature
        self.max_tokens = resolved.max_tokens
        self.max_words = resolve_max_words(resolved.max_words)
        self.timeout: float | int | None = resolved.timeout
        self.num_retries: int | None = resolved.num_retries
        self.verbose = verbose
        self.use_ufun_tools = use_ufun_tools
        self._custom_system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs or {}

    @property
    def base_negotiator(self) -> SAONegotiator:
        """The underlying negotiator that handles core negotiation logic."""
        return self._negotiators[0]  # type: ignore[return-value]

    # =========================================================================
    # SAOMetaNegotiator abstract method implementations
    # =========================================================================

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Get proposal from base negotiator and add LLM-generated text.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            An ExtendedOutcome with the base proposal and LLM-generated text.
        """
        # Get proposal from base negotiator
        base_proposal = self.base_negotiator.propose(state, dest=dest)

        if base_proposal is None:
            return None

        # Extract the outcome if it's already an ExtendedOutcome
        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            base_data = base_proposal.data or {}
        else:
            outcome = base_proposal
            base_data = {}

        if outcome is None:
            return None

        # Extract any text received from the other party
        received_text = self._extract_received_text(state)

        if not self.enforce_base_offer:
            # The LLM may override the structured outcome, treating the base
            # proposal as a recommendation. Falls back to the base outcome if
            # the LLM's output is invalid or unparseable.
            decided_outcome, generated_text = self._decide_outcome(
                state, [("base", outcome)], received_text
            )
            if decided_outcome is not None:
                outcome = decided_outcome
        else:
            # Generate text to accompany the offer (base decides the outcome)
            generated_text = self._generate_text(
                state, "propose", outcome, received_text
            )

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedOutcome(outcome=outcome, data=data)

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Get response from base negotiator and add LLM-generated text.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            An ExtendedResponseType with the base response and LLM-generated text.
        """
        # Get response from base negotiator
        base_response = self.base_negotiator.respond(state, source=source)

        # Extract any received text for context
        received_text = self._extract_received_text(state)
        if received_text:
            self._received_messages.append(
                {
                    "step": state.step,
                    "source": source,
                    "text": received_text,
                    "offer": state.current_offer,
                }
            )

        # Extract the response type if it's already an ExtendedResponseType
        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = base_response.data or {}
        else:
            response_type = base_response
            base_data = {}

        if not self.enforce_base_response:
            # The LLM may override the accept/reject/end decision, treating the
            # base response as a recommendation. Falls back to the base
            # response_type if the LLM's output is unparseable. The LLM always
            # gets to decide here, even for a rejection with no received text.
            decided_response, generated_text = self._decide_response(
                state, [("base", response_type)], received_text
            )
            if decided_response is not None:
                response_type = decided_response
        else:
            # Base decides the action; the LLM only adds text. For a rejection
            # with no received text there is nothing to say, so return the raw
            # base response unchanged (no LLM call, no wrapping).
            if response_type == ResponseType.ACCEPT_OFFER:
                action = "accept"
            elif response_type == ResponseType.END_NEGOTIATION:
                action = "end"
            else:
                if received_text:
                    action = "reject"
                else:
                    return base_response
            generated_text = self._generate_text(
                state, action, state.current_offer, received_text
            )

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedResponseType(response=response_type, data=data)

    # =========================================================================
    # LLM text generation methods
    # =========================================================================

    def get_model_string(self) -> str:
        """Get the model string for litellm.

        Returns:
            The full model string in litellm format (provider/model).
        """
        return litellm_model_string(self.provider, self.model)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for text generation.

        Returns the configured prompt verbatim — a custom ``system_prompt`` stays
        untouched on the attribute. The word budget (``max_words``) is appended at
        call time in :meth:`_generate_text`, so bounding the *answer* by
        instruction is kept separate from this attribute; the token budget is left
        open so reasoning models are not cut off mid-thought.

        Returns:
            The system prompt string.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt

        return _dedent("""
            You generate concise, persuasive text to accompany negotiation actions.

            Guidelines:
                1. Justify the offer naturally.
                2. Acknowledge any message received from the other party.
                3. Keep it brief (1-3 sentences) and professional.

            Respond with ONLY this JSON:
            {
                "text": "your message"
            }
            """)

    def _time_status(self, state: SAOState) -> str:
        """:func:`time_status` for ``state``, with limits from ``self.nmi``.

        ``self.nmi`` is ``None`` outside a live negotiation, in which case the
        limits are reported as unlimited rather than guessed at.
        """
        nmi = self.nmi
        return time_status(
            state.step,
            state.relative_time,
            getattr(nmi, "n_steps", None) if nmi is not None else None,
            getattr(nmi, "time_limit", None) if nmi is not None else None,
        )

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Build the user message for the LLM.

        Args:
            state: The current negotiation state.
            action: The action being taken ("propose", "accept", "reject", "end").
            outcome: The outcome being proposed (if any).
            received_text: Text received from the other party (if any).

        Returns:
            The user message string.
        """
        parts = [
            self._time_status(state),
            "",
        ]

        if received_text:
            parts.append(f'The other party said: "{received_text}"')
            parts.append("")

        if action == "propose":
            parts.append(f"You are making the offer {outcome}.")
            parts.append("Write a brief message for this offer.")
        elif action == "accept":
            parts.append(f"You are ACCEPTING the offer {state.current_offer}.")
            parts.append("Write a brief acceptance message.")
        elif action == "reject":
            if outcome:
                parts.append(
                    "You are REJECTING the current offer and "
                    f"counter-proposing {outcome}."
                )
                parts.append("Write a brief message explaining the counter-offer.")
            else:
                parts.append("You are REJECTING the current offer.")
                parts.append("Write a brief rejection message.")
        elif action == "end":
            parts.append("You are ENDING the negotiation without agreement.")
            parts.append("Write a brief closing message.")

        return "\n".join(parts)

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        state: SAOState | None = None,
        max_tokens: int | None = None,
        model_type: str | None = None,
    ) -> str:
        """Call the LLM and get a response.

        Args:
            messages: The conversation messages.
            state: The current negotiation state (for tag processing).
            max_tokens: Per-call override for the output token cap. If None,
                uses the resolved max tokens. A provider-specific alias in
                ``self.llm_kwargs`` always takes precedence.
            model_type: Optional model type/tier (e.g. ``"fast"``) to use for
                this call; re-resolves provider/model/effort/etc. from the
                ``NEGMAS_LLM_<...>_<VAR>_<TYPE>`` variables. When None, the
                negotiator's own construction-time settings are used.

        Returns:
            The LLM response text.
        """
        # Process all message contents with process_prompt
        processed_messages = []
        for msg in messages:
            processed_content = process_prompt(msg["content"], self, state)
            processed_messages.append({**msg, "content": processed_content})

        cfg = effective_llm_config(self, model_type)
        # Utility-function tool-use is offered only when enabled and a ufun is
        # actually available to compute against.
        tools_enabled = self.use_ufun_tools and self.ufun is not None
        call_messages = list(processed_messages)
        kwargs: dict[str, Any] = {
            "model": litellm_model_string(cfg.provider, cfg.model),
            "messages": call_messages,
            **self.llm_kwargs,
        }
        # Model-dependent parameters: explicit values win; None resolves a
        # model-appropriate default. llm_kwargs aliases take precedence.
        apply_temperature(kwargs, cfg.provider, cfg.model, cfg.temperature)
        apply_max_tokens(
            kwargs,
            cfg.provider,
            cfg.model,
            max_tokens if max_tokens is not None else cfg.max_tokens,
        )
        apply_effort(kwargs, cfg.effort)

        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.timeout is not None:
            kwargs["timeout"] = cfg.timeout
        if cfg.num_retries is not None:
            kwargs["num_retries"] = cfg.num_retries

        if tools_enabled:
            kwargs["tools"] = UFUN_TOOL_SPECS

        # Print prompt if verbose mode is enabled (using rich)
        console = Console() if self.verbose else None
        if self.verbose and console:
            console.print()
            # Create a table for the prompt header
            header = Table.grid(padding=(0, 1))
            header.add_column(style="bold cyan")
            header.add_row(f"LLM PROMPT ({cfg.provider}/{cfg.model})")
            console.print(Panel(header, style="cyan"))

            for msg in processed_messages:
                role_style = "bold green" if msg["role"] == "assistant" else "bold blue"
                console.print(f"[{role_style}][{msg['role'].upper()}][/{role_style}]")
                console.print(msg["content"])
                console.print()

        # Time the LLM call(s). When tool-use is enabled this loops: run any
        # requested ufun tools in-process and feed results back until the
        # model gives a final (non-tool-call) answer.
        start_time = time.perf_counter()
        response_text = ""
        for _round in range(MAX_TOOL_ROUNDS + 1):
            call_start = time.perf_counter()
            response = litellm.completion(**kwargs)
            self.token_usage.add(response, seconds=time.perf_counter() - call_start)
            model_response = cast(ModelResponse, response)
            choices = cast(list["Choices"], model_response.choices)
            message = choices[0].message
            tool_calls = getattr(message, "tool_calls", None) if tools_enabled else None

            if tool_calls:
                call_messages.append(assistant_tool_call_entry(message, tool_calls))

                def _log_tool_call(name: str, arguments: str, result: Any) -> None:
                    if self.verbose and console:
                        console.print(
                            f"[dim green]ufun tool {name}"
                            f"({arguments}) -> {result}[/dim green]"
                        )

                call_messages.extend(
                    tool_result_messages(
                        tool_calls,
                        self.ufun,  # type: ignore[arg-type]
                        on_call=_log_tool_call,
                    )
                )
                continue

            response_text = message.content or ""
            break
        elapsed_time = time.perf_counter() - start_time

        # Print response if verbose mode is enabled (using rich)
        if self.verbose and console:
            console.print()
            # Create a panel for the response with timing info
            header = Table.grid(padding=(0, 1))
            header.add_column(style="bold green")
            header.add_column(justify="right", style="bold yellow")
            header.add_row(
                f"LLM RESPONSE ({cfg.provider}/{cfg.model})",
                f"[{elapsed_time:.2f}s]",
            )
            console.print(Panel(header, style="green"))

            # Try to format as JSON if it looks like JSON
            stripped = response_text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    # Parse and re-format for pretty printing
                    parsed = json.loads(stripped)
                    formatted = json.dumps(parsed, indent=2)
                    syntax = Syntax(
                        formatted, "json", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                except json.JSONDecodeError:
                    console.print(response_text)
            else:
                console.print(response_text)
            console.print()

        return response_text

    def _parse_text_response(self, response_text: str) -> str:
        """Parse the LLM response to extract the text message.

        Args:
            response_text: The raw LLM response.

        Returns:
            The extracted text message.
        """
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "text" in data:
                    return str(data["text"])
            except json.JSONDecodeError:
                pass

        # Fallback: use the response as-is (stripped)
        return response_text.strip()

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Generate text to accompany an offer or response.

        Args:
            state: The current negotiation state.
            action: The action being taken.
            outcome: The outcome being proposed (if any).
            received_text: Text received from the other party (if any).

        Returns:
            The generated text message.
        """
        system_prompt = self._build_system_prompt()
        # The word budget is appended here rather than stored, so a custom
        # ``system_prompt`` stays verbatim on the attribute. Bounding the answer
        # by instruction keeps the token budget free for hidden reasoning.
        limit = word_limit_instruction(self.max_words)
        if limit:
            system_prompt = f"{system_prompt}\n{limit}\n"
        user_message = self._build_user_message(state, action, outcome, received_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response_text = self._call_llm(messages, state)
        return self._parse_text_response(response_text)

    def _extract_received_text(self, state: SAOState) -> str | None:
        """Extract text from the most recent offer received.

        Args:
            state: The current negotiation state.

        Returns:
            The text from the received offer, or None if not available.
        """
        # Check current_data for text
        if state.current_data and isinstance(state.current_data, dict):
            text = state.current_data.get("text")
            if text:
                return str(text)

        # Check new_data for the most recent text
        if state.new_data:
            for data in reversed(state.new_data):
                if data and isinstance(data, dict):
                    text = data.get("text")
                    if text:
                        return str(text)

        return None

    # =========================================================================
    # LLM-decides helpers (override / synthesis path)
    # =========================================================================
    #
    # Used when ``enforce_base_offer``/``enforce_base_response`` is False, and by
    # the recommender subclasses. The LLM is asked to choose an outcome (or a
    # response) from one or more base recommendations and return strict JSON. An
    # invalid/unparseable result yields ``None`` so the caller can fall back to
    # the base.

    def _format_outcome_space(self) -> str:
        """Format the outcome space for the LLM.

        Returns:
            A string describing the outcome space, or an empty string if the
            NMI or outcome space is unavailable.
        """
        if self.nmi is None or self.nmi.outcome_space is None:
            return ""
        outcome_space = self.nmi.outcome_space
        try:
            os_dict = serialize(outcome_space)
            os_dict.pop("__python_class__", None)
            parts = [
                "The outcome space follows.",
                "",
                f"```json\n{json.dumps(os_dict, indent=2, default=str)}\n```",
                "",
                "An outcome is a mapping of each issue name to one of its values.",
            ]
            return "\n".join(parts)
        except Exception:
            return f"The outcome space follows.\n\n{outcome_space}\n"

    def _format_outcome(self, outcome: Outcome | None) -> str:
        """Format a single outcome for display to the LLM.

        Args:
            outcome: The outcome to format.

        Returns:
            A human-readable ``{issue=value, ...}`` string when the outcome space
            is known, else ``str(outcome)``.
        """
        if outcome is None:
            return "None"
        if self.nmi is not None and self.nmi.outcome_space is not None:
            try:
                issues = self.nmi.outcome_space.issues  # type: ignore[attr-defined]
                if issues:
                    parts = []
                    for i, value in enumerate(outcome):
                        if i < len(issues):
                            parts.append(f"{issues[i].name}={value}")
                        else:
                            parts.append(str(value))
                    return "{" + ", ".join(parts) + "}"
            except AttributeError:
                pass
        return str(outcome)

    def _build_decision_system_prompt(self, for_response: bool) -> str:
        """Build the system prompt for the override/synthesis path.

        Args:
            for_response: True for the respond path (choose a response), False
                for the propose path (choose an outcome).

        Returns:
            The system prompt string.
        """
        if for_response:
            return _dedent("""
                You are the decision-maker in a negotiation. You receive one or
                more recommended responses from base strategies and must decide
                whether to ACCEPT the current offer, REJECT it, or END the
                negotiation.

                You may follow a recommendation or choose differently.

                Respond with ONLY this JSON:
                {
                    "response": "accept" | "reject" | "end",
                    "text": "a brief message for the counterpart"
                }
                """)
        return _dedent("""
            You are the decision-maker in a negotiation. You receive one or more
            recommended outcomes from base strategies and must decide the
            outcome to propose.

            You may follow a recommendation or choose a different valid outcome
            from the outcome space.

            Respond with ONLY this JSON:
            {
                "outcome": {"issue_name": value, ...},
                "text": "a brief message for the counterpart"
            }
            """)

    def _build_decision_user_message(
        self,
        state: SAOState,
        recommendations: list[tuple[str, Any]],
        received_text: str | None,
        for_response: bool,
    ) -> str:
        """Build the user message for the override/synthesis path.

        Args:
            state: The current negotiation state.
            recommendations: ``(name, value)`` pairs where ``value`` is an
                outcome (propose) or a :class:`ResponseType` (respond).
            received_text: Text received from the other party (if any).
            for_response: True for respond, False for propose.

        Returns:
            The user message string.
        """
        parts = [
            self._time_status(state),
            "",
        ]
        if for_response and state.current_offer is not None:
            parts.append(
                f"The current offer on the table is {self._format_outcome(state.current_offer)}."
            )
            parts.append("")
        if received_text:
            parts.append(f'The other party said: "{received_text}"')
            parts.append("")
        if not for_response:
            os_text = self._format_outcome_space()
            if os_text:
                parts.append(os_text)
                parts.append("")

        response_label = {
            ResponseType.ACCEPT_OFFER: "accept",
            ResponseType.REJECT_OFFER: "reject",
            ResponseType.END_NEGOTIATION: "end",
        }
        parts.append("Recommendations from base strategies:")
        for name, value in recommendations:
            if for_response:
                label = response_label.get(value, str(value))
                parts.append(f"  - {name}: {label}")
            else:
                parts.append(f"  - {name}: {self._format_outcome(value)}")
        parts.append("")

        if for_response:
            parts.append(
                "Decide whether to accept, reject, or end the negotiation and "
                "write a brief message."
            )
        else:
            parts.append(
                "Choose an outcome to propose and write a brief message for it."
            )
        return "\n".join(parts)

    def _parse_decision_response(
        self, response_text: str, for_response: bool
    ) -> tuple[Any | None, str]:
        """Parse the LLM decision response.

        Args:
            response_text: The raw LLM response.
            for_response: True for respond, False for propose.

        Returns:
            A ``(value, text)`` tuple. For propose, ``value`` is an
            :class:`Outcome` or ``None``. For respond, ``value`` is a
            :class:`ResponseType` or ``None``. ``text`` is the message (possibly
            the stripped raw response on parse failure).
        """
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        text: str | None = None
        if not json_match:
            return None, response_text.strip()
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None, response_text.strip()
        if isinstance(data, dict) and "text" in data:
            text = str(data["text"])

        if for_response:
            response_str = str(data.get("response", "")).lower()
            response_map = {
                "accept": ResponseType.ACCEPT_OFFER,
                "reject": ResponseType.REJECT_OFFER,
                "end": ResponseType.END_NEGOTIATION,
            }
            value: Any | None = response_map.get(response_str)
        else:
            value = self._outcome_from_data(data.get("outcome"))

        if text is None:
            text = response_text.strip()
        return value, text

    def _outcome_from_data(self, outcome_data: Any) -> Outcome | None:
        """Convert raw LLM outcome data into an outcome tuple.

        Mirrors :meth:`LLMNegotiator._parse_llm_response`: a list becomes a
        tuple directly; a dict is ordered by the outcome-space issues when
        available, else by dict insertion order.

        Args:
            outcome_data: The raw parsed ``outcome`` value from the LLM JSON.

        Returns:
            An outcome tuple, or ``None`` if it cannot be interpreted.
        """
        if outcome_data is None:
            return None
        if isinstance(outcome_data, list):
            return tuple(outcome_data)
        if isinstance(outcome_data, dict):
            if self.nmi is not None and self.nmi.outcome_space is not None:
                try:
                    issues = self.nmi.outcome_space.issues  # type: ignore[attr-defined]
                    if issues:
                        values = []
                        for issue in issues:
                            if issue.name in outcome_data:
                                values.append(outcome_data[issue.name])
                            else:
                                # Case-insensitive fallback
                                found = False
                                for key, val in outcome_data.items():
                                    if key.lower() == issue.name.lower():
                                        values.append(val)
                                        found = True
                                        break
                                if not found:
                                    break
                        if len(values) == len(issues):
                            return tuple(values)
                except AttributeError:
                    pass
            return tuple(outcome_data.values())
        return None

    def _validate_outcome(self, outcome: Outcome | None) -> Outcome | None:
        """Validate an outcome against the outcome space.

        Args:
            outcome: The outcome to validate.

        Returns:
            The outcome if valid, ``None`` if it has None values or is not valid
            in the outcome space (a warning is emitted in the latter cases).
        """
        if outcome is None:
            return None
        if any(v is None for v in outcome):
            warnings.warn(
                f"LLM returned outcome with None values: {outcome}. "
                "Falling back to the base recommendation.",
                stacklevel=2,
            )
            return None
        if self.nmi is not None and self.nmi.outcome_space is not None:
            try:
                if self.nmi.outcome_space.is_valid(outcome):  # type: ignore[attr-defined]
                    return outcome
                warnings.warn(
                    f"LLM returned invalid outcome: {outcome}. Not valid in "
                    "the outcome space. Falling back to the base recommendation.",
                    stacklevel=2,
                )
                return None
            except (AttributeError, TypeError):
                pass
        return outcome

    def _decide_outcome(
        self,
        state: SAOState,
        recommendations: list[tuple[str, Outcome | None]],
        received_text: str | None,
    ) -> tuple[Outcome | None, str]:
        """Ask the LLM to choose an outcome from recommendations.

        Args:
            state: The current negotiation state.
            recommendations: ``(name, outcome)`` pairs from base strategies.
            received_text: Text received from the other party (if any).

        Returns:
            A ``(validated_outcome, text)`` tuple. ``validated_outcome`` is
            ``None`` if the LLM's outcome is invalid or unparseable (caller
            falls back to the base).
        """
        system_prompt = self._build_decision_system_prompt(for_response=False)
        limit = word_limit_instruction(self.max_words)
        if limit:
            system_prompt = f"{system_prompt}\n{limit}\n"
        user_message = self._build_decision_user_message(
            state, recommendations, received_text, for_response=False
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response_text = self._call_llm(messages, state)
        outcome, text = self._parse_decision_response(response_text, for_response=False)
        return self._validate_outcome(outcome), text

    def _decide_response(
        self,
        state: SAOState,
        recommendations: list[tuple[str, ResponseType]],
        received_text: str | None,
    ) -> tuple[ResponseType | None, str]:
        """Ask the LLM to choose a response from recommendations.

        Args:
            state: The current negotiation state.
            recommendations: ``(name, response_type)`` pairs from base strategies.
            received_text: Text received from the other party (if any).

        Returns:
            A ``(response_type, text)`` tuple. ``response_type`` is ``None`` if
            the LLM's response is unparseable (caller falls back to the base).
        """
        system_prompt = self._build_decision_system_prompt(for_response=True)
        limit = word_limit_instruction(self.max_words)
        if limit:
            system_prompt = f"{system_prompt}\n{limit}\n"
        user_message = self._build_decision_user_message(
            state, recommendations, received_text, for_response=True
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response_text = self._call_llm(messages, state)
        response_type, text = self._parse_decision_response(
            response_text, for_response=True
        )
        return response_type, text

    def on_negotiation_start(self, state) -> None:
        """Reset received messages when negotiation starts.

        Args:
            state: The initial negotiation state.
        """
        super().on_negotiation_start(state)
        self._received_messages = []


# =============================================================================
# LLM-wrapped versions of native negmas negotiators
# =============================================================================
#
# These classes provide convenient wrappers around native negmas negotiators,
# adding LLM-generated text to their offers while preserving the original
# negotiation strategy.


class LLMAspirationNegotiator(LLMMetaNegotiator):
    """LLM-wrapped AspirationNegotiator.

    This negotiator uses the aspiration-based time-dependent strategy for making
    offers and acceptance decisions, while adding LLM-generated persuasive text
    to accompany each offer.

    The aspiration negotiator starts with high aspirations (demanding offers)
    and gradually lowers them over time.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import AspirationNegotiator

        base_negotiator = AspirationNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMBoulwareTBNegotiator(LLMMetaNegotiator):
    """LLM-wrapped BoulwareTBNegotiator.

    This negotiator uses the Boulware time-based concession strategy, which
    concedes slowly at first and more rapidly near the deadline. LLM-generated
    persuasive text accompanies each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import BoulwareTBNegotiator

        base_negotiator = BoulwareTBNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMConcederTBNegotiator(LLMMetaNegotiator):
    """LLM-wrapped ConcederTBNegotiator.

    This negotiator uses the Conceder time-based strategy, which concedes
    rapidly at first and slows down near the deadline. LLM-generated text
    accompanies each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import ConcederTBNegotiator

        base_negotiator = ConcederTBNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMLinearTBNegotiator(LLMMetaNegotiator):
    """LLM-wrapped LinearTBNegotiator.

    This negotiator uses a linear time-based concession strategy, conceding
    at a constant rate over time. LLM-generated text accompanies each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import LinearTBNegotiator

        base_negotiator = LinearTBNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMTimeBasedConcedingNegotiator(LLMMetaNegotiator):
    """LLM-wrapped TimeBasedConcedingNegotiator.

    A configurable time-based conceding negotiator with LLM-generated text
    accompanying each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import TimeBasedConcedingNegotiator

        base_negotiator = TimeBasedConcedingNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMTimeBasedNegotiator(LLMMetaNegotiator):
    """LLM-wrapped TimeBasedNegotiator.

    A general time-based negotiator with LLM-generated text accompanying
    each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.timebased import TimeBasedNegotiator

        base_negotiator = TimeBasedNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMNiceNegotiator(LLMMetaNegotiator):
    """LLM-wrapped NiceNegotiator.

    A cooperative negotiator that makes nice offers with LLM-generated
    persuasive text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.nice import NiceNegotiator

        base_negotiator = NiceNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMToughNegotiator(LLMMetaNegotiator):
    """LLM-wrapped ToughNegotiator.

    An aggressive negotiator that makes tough offers with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.tough import ToughNegotiator

        base_negotiator = ToughNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMNaiveTitForTatNegotiator(LLMMetaNegotiator):
    """LLM-wrapped NaiveTitForTatNegotiator.

    A tit-for-tat negotiator that mirrors the opponent's behavior with
    LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.titfortat import NaiveTitForTatNegotiator

        base_negotiator = NaiveTitForTatNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMRandomNegotiator(LLMMetaNegotiator):
    """LLM-wrapped RandomNegotiator.

    A negotiator that makes random offers with LLM-generated text. Useful
    for testing and baselines.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.randneg import RandomNegotiator

        base_negotiator = RandomNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMRandomAlwaysAcceptingNegotiator(LLMMetaNegotiator):
    """LLM-wrapped RandomAlwaysAcceptingNegotiator.

    A negotiator that makes random offers but always accepts with LLM-generated
    text. Useful for testing.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.randneg import RandomAlwaysAcceptingNegotiator

        base_negotiator = RandomAlwaysAcceptingNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMCABNegotiator(LLMMetaNegotiator):
    """LLM-wrapped CABNegotiator.

    Curve-based Aspiration Boulware negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.cab import CABNegotiator

        base_negotiator = CABNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMCANNegotiator(LLMMetaNegotiator):
    """LLM-wrapped CANNegotiator.

    Curve-based Aspiration Nice negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.cab import CANNegotiator

        base_negotiator = CANNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMCARNegotiator(LLMMetaNegotiator):
    """LLM-wrapped CARNegotiator.

    Curve-based Aspiration Random negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.cab import CARNegotiator

        base_negotiator = CARNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMMiCRONegotiator(LLMMetaNegotiator):
    """LLM-wrapped MiCRONegotiator.

    MiCRO (Mixed strategy with CRoss Offers) negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.micro import MiCRONegotiator

        base_negotiator = MiCRONegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMFastMiCRONegotiator(LLMMetaNegotiator):
    """LLM-wrapped FastMiCRONegotiator.

    Fast version of MiCRO negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.micro import FastMiCRONegotiator

        base_negotiator = FastMiCRONegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMUtilBasedNegotiator(LLMMetaNegotiator):
    """LLM-wrapped UtilBasedNegotiator.

    Utility-based negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.utilbased import UtilBasedNegotiator

        base_negotiator = UtilBasedNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMWARNegotiator(LLMMetaNegotiator):
    """LLM-wrapped WARNegotiator.

    War negotiator (aggressive) with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.war import WARNegotiator

        base_negotiator = WARNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMWANNegotiator(LLMMetaNegotiator):
    """LLM-wrapped WANNegotiator.

    War-Aspiration-Nice negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.war import WANNegotiator

        base_negotiator = WANNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMWABNegotiator(LLMMetaNegotiator):
    """LLM-wrapped WABNegotiator.

    War-Aspiration-Boulware negotiator with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.war import WABNegotiator

        base_negotiator = WABNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMLimitedOutcomesNegotiator(LLMMetaNegotiator):
    """LLM-wrapped LimitedOutcomesNegotiator.

    A negotiator that works with a limited set of outcomes, with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.limited import LimitedOutcomesNegotiator

        base_negotiator = LimitedOutcomesNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMLimitedOutcomesAcceptor(LLMMetaNegotiator):
    """LLM-wrapped LimitedOutcomesAcceptor.

    An acceptor that works with a limited set of outcomes, with LLM-generated text.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.limited import LimitedOutcomesAcceptor

        base_negotiator = LimitedOutcomesAcceptor(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


class LLMHybridNegotiator(LLMMetaNegotiator):
    """LLM-wrapped HybridNegotiator.

    A hybrid negotiator that combines multiple strategies using HybridOfferingPolicy
    and ACNext acceptance policy, with LLM-generated text accompanying each offer.

    Args:
        provider: The LLM provider (default: "ollama").
        model: The model name (default: DEFAULT_OLLAMA_MODEL).
        api_key: API key for the provider (if required).
        api_base: Base URL for the API.
        temperature: Sampling temperature for the LLM. None (default) selects
            a model-appropriate value.
        max_tokens: Maximum tokens in the LLM response. None (default) selects
            a model-appropriate budget (larger for reasoning/thinking models so
            hidden deliberation cannot starve the visible response).
        verbose: If True, print LLM prompts and responses to stdout (default: False).
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to the base negotiator.
            Supports `alpha` and `beta` parameters for the acceptance policy.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from negmas.gb.negotiators.hybrid import HybridNegotiator

        base_negotiator = HybridNegotiator(**kwargs)
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )


# =============================================================================
# Recommender-based meta negotiators
# =============================================================================
#
# These expose the "LLM decides from base recommendation(s)" mechanism built
# into LLMMetaNegotiator as named classes. ``LLMNegotiatorWithRecommender`` and
# ``LLMEnhancedNegotiator`` wrap a single base; ``LLMNegotiatorWithMultipleRecommenders``
# wraps several named bases and synthesizes one outcome/response from all their
# recommendations.


class LLMNegotiatorWithRecommender(LLMMetaNegotiator):
    """An LLMMetaNegotiator that lets the LLM decide the outcome and response.

    The single base negotiator acts as a *recommender*: its proposal/response is
    passed to the LLM, which may follow it or override it. This is exactly
    :class:`LLMMetaNegotiator` with both ``enforce_*`` flags set to ``False``;
    the flags are not exposed in the constructor.

    Args:
        base_negotiator: The recommender negotiator.
        provider: The LLM provider (None resolves from the environment).
        model: The model name (None resolves from the environment).
        effort: Optional reasoning effort.
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Hard ceiling on tokens the model may spend.
        max_words: Approximate word budget for the generated message.
        timeout: Request timeout in seconds.
        num_retries: Number of retries on transient failures.
        verbose: If True, print LLM prompts and responses to stdout.
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for the LLM decision.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to SAOMetaNegotiator.
    """

    def __init__(
        self,
        base_negotiator: SAONegotiator,
        provider: str | None = None,
        model: str | None = None,
        *,
        effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        timeout: float | int | None = None,
        num_retries: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            effort=effort,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_words=max_words,
            timeout=timeout,
            num_retries=num_retries,
            enforce_base_offer=False,
            enforce_base_response=False,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )


class LLMEnhancedNegotiator(LLMMetaNegotiator):
    """An LLMMetaNegotiator where the base decides and the LLM only adds text.

    The structured outcome of every proposal and the accept/reject/end decision
    come verbatim from the base negotiator; the LLM only generates the
    accompanying ``data["text"]``. This is exactly :class:`LLMMetaNegotiator`
    with both ``enforce_*`` flags set to ``True`` (the default); the flags are
    not exposed in the constructor.

    Args:
        base_negotiator: The base negotiator that decides offers and responses.
        provider: The LLM provider (None resolves from the environment).
        model: The model name (None resolves from the environment).
        effort: Optional reasoning effort.
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Hard ceiling on tokens the model may spend.
        max_words: Approximate word budget for the generated message.
        timeout: Request timeout in seconds.
        num_retries: Number of retries on transient failures.
        verbose: If True, print LLM prompts and responses to stdout.
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for text generation.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to SAOMetaNegotiator.
    """

    def __init__(
        self,
        base_negotiator: SAONegotiator,
        provider: str | None = None,
        model: str | None = None,
        *,
        effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        timeout: float | int | None = None,
        num_retries: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            effort=effort,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_words=max_words,
            timeout=timeout,
            num_retries=num_retries,
            enforce_base_offer=True,
            enforce_base_response=True,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )


class LLMNegotiatorWithMultipleRecommenders(LLMMetaNegotiator):
    """A meta-negotiator that synthesizes one outcome from several recommenders.

    Unlike :class:`LLMMetaNegotiator`, this wraps **several** named base
    negotiators ("recommenders"). On each step it collects every recommender's
    proposal (or response), feeds all the recommendations to the LLM, and the
    LLM decides the single outcome (or response) to return. There are no
    ``enforce_*`` flags: this class always produces its result from the
    recommendations. When the LLM's output is invalid or unparseable it falls
    back to the first valid recommendation for proposals, and to the majority
    vote of the recommenders for responses (ties broken by first occurrence).

    Args:
        recommenders: The base negotiators that provide recommendations. At
            least one is required.
        recommender_names: Optional names for the recommenders, aligned with
            ``recommenders``. If omitted, each recommender's own name is used.
        provider: The LLM provider (None resolves from the environment).
        model: The model name (None resolves from the environment).
        effort: Optional reasoning effort.
        api_key: API key for the provider (if required).
        api_base: Base URL for the API (useful for local deployments).
        temperature: Sampling temperature for the LLM.
        max_tokens: Hard ceiling on tokens the model may spend.
        max_words: Approximate word budget for the generated message.
        timeout: Request timeout in seconds.
        num_retries: Number of retries on transient failures.
        verbose: If True, print LLM prompts and responses to stdout.
        use_ufun_tools: If True, offer the LLM function tools that compute its
            own utility function in-process instead of estimating it from the
            prompt (default: False). See :class:`LLMMetaNegotiator`.
        system_prompt: Custom system prompt for the LLM decision.
        llm_kwargs: Additional keyword arguments passed to litellm.completion.
        **kwargs: Additional arguments passed to SAOMetaNegotiator.
    """

    def __init__(
        self,
        recommenders: list[SAONegotiator],
        recommender_names: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        *,
        effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_words: int | None = None,
        timeout: float | int | None = None,
        num_retries: int | None = None,
        verbose: bool = False,
        use_ufun_tools: bool = True,
        system_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not recommenders:
            raise ValueError(
                "LLMNegotiatorWithMultipleRecommenders requires >= 1 recommender"
            )
        if recommender_names is None:
            recommender_names = [
                neg.name or f"recommender_{i}" for i, neg in enumerate(recommenders)
            ]
        if len(recommender_names) != len(recommenders):
            raise ValueError(
                "recommender_names must have the same length as recommenders"
            )
        # Register all recommenders as children, bypassing the single-base
        # constructor path of LLMMetaNegotiator.
        SAOMetaNegotiator.__init__(
            self,
            negotiators=recommenders,
            negotiator_names=recommender_names,
            share_ufun=True,
            share_nmi=True,
            **kwargs,
        )
        resolved = resolve_llm_config(
            type(self).__name__,
            provider=provider,
            model=model,
            effort=effort,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_words=max_words,
            timeout=timeout,
            num_retries=num_retries,
            default_provider=self.DEFAULT_PROVIDER or DEFAULT_PROVIDER,
            default_model=self.DEFAULT_MODEL,
            lock_provider=self.LOCK_PROVIDER,
        )
        self._store_llm_config(
            resolved,
            verbose=verbose,
            use_ufun_tools=use_ufun_tools,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
        )
        # No enforce flags: this class always decides from recommendations.
        self.enforce_base_offer = False
        self.enforce_base_response = False
        self._received_messages: list[dict[str, Any]] = []

    @property
    def recommenders(self) -> tuple[SAONegotiator, ...]:
        """The recommender sub-negotiators, in registration order."""
        return tuple(self._negotiators)

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Synthesize a proposal from all recommenders via the LLM.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            An ExtendedOutcome with the LLM-chosen outcome and text, falling
            back to the first valid recommender outcome if the LLM fails.
        """
        received_text = self._extract_received_text(state)
        recommendations: list[tuple[str, Outcome | None]] = []
        for name, neg in zip(self.negotiator_names, self._negotiators, strict=True):
            proposal = neg.propose(state, dest=dest)
            outcome = (
                proposal.outcome if isinstance(proposal, ExtendedOutcome) else proposal
            )
            recommendations.append((name, outcome))

        decided_outcome, generated_text = self._decide_outcome(
            state, recommendations, received_text
        )
        if decided_outcome is None:
            # Fall back to the first non-None recommender outcome.
            decided_outcome = next(
                (o for _, o in recommendations if o is not None), None
            )
        if decided_outcome is None:
            return None
        return ExtendedOutcome(outcome=decided_outcome, data={"text": generated_text})

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Synthesize a response from all recommenders via the LLM.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            An ExtendedResponseType with the LLM-chosen response and text,
            falling back to the majority vote of the recommenders if the LLM
            fails.
        """
        received_text = self._extract_received_text(state)
        if received_text:
            self._received_messages.append(
                {
                    "step": state.step,
                    "source": source,
                    "text": received_text,
                    "offer": state.current_offer,
                }
            )
        recommendations: list[tuple[str, ResponseType]] = []
        response_types: list[ResponseType] = []
        for name, neg in zip(self.negotiator_names, self._negotiators, strict=True):
            base_response = neg.respond(state, source=source)
            response_type = (
                base_response.response
                if isinstance(base_response, ExtendedResponseType)
                else base_response
            )
            recommendations.append((name, response_type))
            response_types.append(response_type)

        decided_response, generated_text = self._decide_response(
            state, recommendations, received_text
        )
        if decided_response is None:
            decided_response = self._majority_response(response_types)
        return ExtendedResponseType(
            response=decided_response, data={"text": generated_text}
        )

    @staticmethod
    def _majority_response(response_types: list[ResponseType]) -> ResponseType:
        """Pick the majority response type, breaking ties by first occurrence.

        Args:
            response_types: The recommenders' response types, in order.

        Returns:
            The majority response type, or REJECT_OFFER for an empty list.
        """
        if not response_types:
            return ResponseType.REJECT_OFFER
        counts: dict[ResponseType, int] = {}
        for rt in response_types:
            counts[rt] = counts.get(rt, 0) + 1
        best = response_types[0]
        best_count = 0
        seen: set[ResponseType] = set()
        for rt in response_types:
            if rt in seen:
                continue
            seen.add(rt)
            if counts[rt] > best_count:
                best = rt
                best_count = counts[rt]
        return best

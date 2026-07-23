"""Single source of truth for resolving the LLM provider/model of a negotiator.

Every LLM negotiator, meta-negotiator, and component in ``negmas-llm`` resolves
its effective LLM configuration through :func:`resolve_llm_config` in this
module. There is exactly one resolution algorithm and it is documented here so
the behavior is predictable and easy to drive from the environment when running
experiments.

Three independent axes of environment configuration
====================================================

1. **Per-provider default model** — ``NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL``
   (e.g. ``NEGMAS_LLM_OPENAI_DEFAULT_MODEL``). Sets the default model used
   *whenever that provider is selected*, regardless of negotiator type.
   (:func:`negmas_llm.common.default_model_for_provider` exposes the same
   per-provider lookup for standalone use.)

2. **Global defaults** — ``NEGMAS_LLM_PROVIDER``, ``NEGMAS_LLM_MODEL``,
   ``NEGMAS_LLM_TEMPERATURE``, ``NEGMAS_LLM_MAX_TOKENS``, ``NEGMAS_LLM_TIMEOUT``,
   ``NEGMAS_LLM_NUM_RETRIES``, ``NEGMAS_LLM_API_KEY``, ``NEGMAS_LLM_API_BASE``.
   Apply to *every* negotiator that does not specify the value explicitly.
   This is the knob for "run the whole experiment on model X".

3. **Per-negotiator-type overrides** — ``NEGMAS_LLM_<ClassName>_<VAR>`` where
   ``<ClassName>`` is the *exact* concrete class name of the negotiator (e.g.
   ``NEGMAS_LLM_LLMBoulwareTBNegotiator_MODEL``) and ``<VAR>`` is one of
   ``PROVIDER``, ``MODEL``, ``EFFORT``, ``TEMPERATURE``, ``MAX_TOKENS``,
   ``TIMEOUT``, ``NUM_RETRIES``, ``API_KEY``, ``API_BASE``. Apply to *only that
   one class*. This is the knob for "everyone on model X, but this one
   negotiator on model Y from a different provider".

Model types (tiers)
====================

A *model type* is a named ``(provider, model, effort)`` bundle — e.g. ``fast``,
``accurate``, ``friendly`` — so different parts of a negotiator can ask for
different models. When a caller passes ``model_type="fast"`` to
:func:`resolve_llm_config`, the (uppercased) type name is **appended** as a
suffix to every environment variable it reads::

    NEGMAS_LLM_MODEL_FAST                       (global, fast tier)
    NEGMAS_LLM_PROVIDER_FAST                    (global, fast tier)
    NEGMAS_LLM_EFFORT_FAST                      (global, fast tier)
    NEGMAS_LLM_<ClassName>_MODEL_FAST           (per-type, fast tier)
    ...

Type names must be identifier-safe (letters, digits, underscores) because they
become part of an environment variable name. ``model_type=None`` reads the
un-suffixed variables and reproduces the historical behavior exactly.

Precedence (highest wins)
=========================

For every variable, with the requested tier ``<T>`` (absent when
``model_type=None``)::

    explicit constructor argument
      >  NEGMAS_LLM_<ClassName>_<VAR>_<T>       (per-type, tier)
      >  NEGMAS_LLM_<ClassName>_<VAR>           (per-type, base)
      >  NEGMAS_LLM_<VAR>_<T>                   (global, tier)
      >  NEGMAS_LLM_<VAR>                       (global, base)
      >  built-in default

This ordering is **scope-dominant**: the class scope outranks the tier
qualifier, so a per-type base (``NEGMAS_LLM_<ClassName>_MODEL``) beats a global
tier (``NEGMAS_LLM_MODEL_FAST``). Set the fully-qualified
``NEGMAS_LLM_<ClassName>_MODEL_<T>`` for unambiguous control.

Two provider/model coherence rules keep a model from landing on the wrong
provider:

* The global ``NEGMAS_LLM_PROVIDER`` (and its tier form ``NEGMAS_LLM_PROVIDER_<T>``)
  is ignored for *provider-locked* classes (the provider-named convenience
  classes such as ``OpenAINegotiator``). Their provider is part of their
  identity; use a per-type override to change it.

* A *global* model (``NEGMAS_LLM_MODEL_<T>`` or ``NEGMAS_LLM_MODEL``) is applied
  only when the negotiator's effective provider equals the global provider
  resolved at the same tier (``NEGMAS_LLM_PROVIDER_<T>`` → ``NEGMAS_LLM_PROVIDER``
  → the default ``ollama``). If a per-type/explicit override put the negotiator
  on a different provider, its model falls back to *that* provider's default
  instead of a global model string meant for another provider. Per-type
  ``NEGMAS_LLM_<ClassName>_MODEL`` is always honored (you named the class, so we
  trust you). ``effort`` carries no such guard — it is not provider-coupled.

Built-in fallbacks live next to the classes that own them, as the class
attributes :data:`DEFAULT_PROVIDER`/:data:`DEFAULT_MODEL`, with the per-provider
hardcoded model table in :data:`negmas_llm.common._MODEL_FALLBACKS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from negmas_llm.common import _MODEL_FALLBACKS

__all__ = [
    "DEFAULT_PROVIDER",
    "CONFIGURABLE_VARS",
    "ResolvedLLMConfig",
    "resolve_llm_config",
    "effective_llm_config",
    "per_type_env_var",
]

#: Global fallback provider used when nothing else selects one. Matches the
#: historical default of the strategy meta-negotiators.
DEFAULT_PROVIDER: str = "ollama"

#: The per-variable suffixes recognized in ``NEGMAS_LLM_<ClassName>_<VAR>`` and
#: ``NEGMAS_LLM_<VAR>`` (optionally followed by a ``_<MODELTYPE>`` tier suffix).
#: Public so callers/tests can enumerate the knobs.
CONFIGURABLE_VARS: tuple[str, ...] = (
    "PROVIDER",
    "MODEL",
    "EFFORT",
    "TEMPERATURE",
    "MAX_TOKENS",
    "TIMEOUT",
    "NUM_RETRIES",
    "API_KEY",
    "API_BASE",
)


def _norm_type(model_type: str | None) -> str | None:
    """Uppercase the model-type name for use as an env-var suffix (None -> None)."""
    return model_type.upper() if model_type else None


def per_type_env_var(
    negotiator_type: str, var: str, model_type: str | None = None
) -> str:
    """Name of the per-negotiator-type environment variable for ``var``.

    Args:
        negotiator_type: Exact concrete class name (e.g. ``"OpenAINegotiator"``).
        var: One of the suffixes in :data:`CONFIGURABLE_VARS` (case-insensitive).
        model_type: Optional model-type/tier name (e.g. ``"fast"``); appended,
            uppercased, as a suffix.

    Returns:
        The environment variable name, e.g.
        ``"NEGMAS_LLM_OpenAINegotiator_MODEL"`` or, with ``model_type="fast"``,
        ``"NEGMAS_LLM_OpenAINegotiator_MODEL_FAST"``.
    """
    name = f"NEGMAS_LLM_{negotiator_type}_{var.upper()}"
    mt = _norm_type(model_type)
    return f"{name}_{mt}" if mt else name


def _per_type_names(negotiator_type: str, var: str, mt: str | None) -> list[str]:
    """Per-type env var names for ``var``, most specific first (tier then base)."""
    base = f"NEGMAS_LLM_{negotiator_type}_{var.upper()}"
    return [f"{base}_{mt}", base] if mt else [base]


def _global_names(var: str, mt: str | None) -> list[str]:
    """Global env var names for ``var``, most specific first (tier then base)."""
    base = f"NEGMAS_LLM_{var.upper()}"
    return [f"{base}_{mt}", base] if mt else [base]


def _first_env(names: list[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_per_type(negotiator_type: str, var: str, mt: str | None) -> str | None:
    """First set of the per-type (tier then base) env vars for ``var``."""
    return _first_env(_per_type_names(negotiator_type, var, mt))


def _env_scoped(negotiator_type: str, var: str, mt: str | None) -> str | None:
    """First set of per-type then global env vars for ``var`` (scope-dominant)."""
    return _first_env(
        _per_type_names(negotiator_type, var, mt) + _global_names(var, mt)
    )


def _global_provider(mt: str | None) -> str:
    """The global provider for a tier: PROVIDER_<T> -> PROVIDER -> default."""
    return _first_env(_global_names("PROVIDER", mt)) or DEFAULT_PROVIDER


@dataclass(frozen=True)
class ResolvedLLMConfig:
    """The fully-resolved LLM settings for one negotiator instance.

    ``temperature`` and ``max_tokens`` remain ``None`` when neither an explicit
    argument nor an environment override sets them, so the model-dependent
    defaults in :mod:`negmas_llm.common` still apply at call time. ``effort`` is
    the reasoning effort (e.g. ``"low"``/``"medium"``/``"high"``) sent verbatim
    as ``reasoning_effort``; ``None`` omits it.
    """

    provider: str
    model: str
    api_key: str | None
    api_base: str | None
    temperature: float | None
    max_tokens: int | None
    timeout: float | int | None
    num_retries: int | None
    effort: str | None = None


def _resolve_provider(
    negotiator_type: str,
    explicit: str | None,
    default_provider: str,
    lock_provider: bool,
    mt: str | None,
) -> str:
    if explicit is not None:
        return explicit
    # Per-type overrides (tier then base) are always honored.
    per_type = _env_per_type(negotiator_type, "PROVIDER", mt)
    if per_type:
        return per_type
    if not lock_provider:
        # Global overrides (tier then base). Skipped for provider-locked classes.
        global_provider = _first_env(_global_names("PROVIDER", mt))
        if global_provider:
            return global_provider
    return default_provider


def _resolve_model(
    negotiator_type: str,
    explicit: str | None,
    provider: str,
    default_model: str | None,
    mt: str | None,
) -> str:
    if explicit is not None:
        return explicit
    # Per-type overrides (tier then base) are always honored.
    per_type = _env_per_type(negotiator_type, "MODEL", mt)
    if per_type:
        return per_type
    # A global model applies only when this negotiator's provider matches the
    # global provider resolved at the same tier; otherwise a global model string
    # meant for provider A would be sent to provider B. See module docstring.
    for names, guard_provider in (
        (_global_names("MODEL", mt)[:1] if mt else [], _global_provider(mt)),
        (["NEGMAS_LLM_MODEL"], _global_provider(None)),
    ):
        if names and provider == guard_provider:
            global_model = _first_env(names)
            if global_model:
                return global_model
    # Per-provider env override: NEGMAS_LLM_<PROVIDER>_DEFAULT_MODEL (and the
    # legacy OLLAMA_MODEL). This intentionally wins over the class's built-in
    # DEFAULT_MODEL: it means "whenever this provider is used, use this model".
    per_provider_env = os.environ.get(f"NEGMAS_LLM_{provider.upper()}_DEFAULT_MODEL")
    if per_provider_env:
        return per_provider_env
    if provider == "ollama":
        legacy = os.environ.get("OLLAMA_MODEL")
        if legacy:
            return legacy
    # The class's own documented default (e.g. OpenAINegotiator -> "gpt-4o").
    if default_model:
        return default_model
    # Last resort: the hardcoded per-provider fallback table.
    return _MODEL_FALLBACKS.get(provider, "")


def resolve_llm_config(
    negotiator_type: str,
    *,
    model_type: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | int | None = None,
    num_retries: int | None = None,
    default_provider: str = DEFAULT_PROVIDER,
    default_model: str | None = None,
    lock_provider: bool = False,
) -> ResolvedLLMConfig:
    """Resolve the effective LLM configuration for a negotiator.

    See the module docstring for the full precedence and coherence rules.

    Args:
        negotiator_type: Exact concrete class name, used as the per-type env
            key. Callers pass ``type(self).__name__``.
        model_type: Optional model-type/tier name (e.g. ``"fast"``,
            ``"accurate"``). When given, every environment variable is read with
            the uppercased type appended as a suffix (``NEGMAS_LLM_MODEL_FAST``,
            ``NEGMAS_LLM_<ClassName>_MODEL_FAST``, …), falling back to the
            un-suffixed variable. Must be identifier-safe.
        provider: Explicit provider from the constructor (``None`` to resolve).
        model: Explicit model from the constructor (``None`` to resolve).
        effort: Explicit reasoning effort (``None`` to resolve).
        api_key: Explicit API key (``None`` to resolve).
        api_base: Explicit API base URL (``None`` to resolve).
        temperature: Explicit temperature (``None`` keeps model-dependent
            default at call time unless an env override is set).
        max_tokens: Explicit max tokens (``None`` keeps model-dependent default
            at call time unless an env override is set).
        timeout: Explicit timeout in seconds (``None`` to resolve).
        num_retries: Explicit retry count (``None`` to resolve).
        default_provider: The class's built-in fallback provider.
        default_model: The class's built-in fallback model (``None`` to defer to
            the per-provider default table).
        lock_provider: When ``True``, the global ``NEGMAS_LLM_PROVIDER`` (and its
            tier form) is ignored (used by provider-named classes); per-type
            provider overrides still apply.

    Returns:
        A :class:`ResolvedLLMConfig` with every field resolved.
    """
    mt = _norm_type(model_type)

    resolved_provider = _resolve_provider(
        negotiator_type, provider, default_provider, lock_provider, mt
    )
    resolved_model = _resolve_model(
        negotiator_type, model, resolved_provider, default_model, mt
    )

    effort_value = effort or _env_scoped(negotiator_type, "EFFORT", mt)
    api_key_value = api_key or _env_scoped(negotiator_type, "API_KEY", mt)
    api_base_value = api_base or _env_scoped(negotiator_type, "API_BASE", mt)

    temperature_value = temperature
    if temperature_value is None:
        env_temp = _env_scoped(negotiator_type, "TEMPERATURE", mt)
        if env_temp is not None:
            temperature_value = float(env_temp)

    max_tokens_value = max_tokens
    if max_tokens_value is None:
        # Global base NEGMAS_LLM_MAX_TOKENS is also read model-dependently in
        # common.default_max_tokens; reading it here as well yields the same
        # value and additionally supports per-type and tier overrides.
        env_mt = _env_scoped(negotiator_type, "MAX_TOKENS", mt)
        if env_mt is not None:
            max_tokens_value = int(env_mt)

    timeout_value = timeout
    if timeout_value is None:
        env_timeout = _env_scoped(negotiator_type, "TIMEOUT", mt)
        if env_timeout is not None:
            timeout_value = float(env_timeout)

    num_retries_value = num_retries
    if num_retries_value is None:
        env_retries = _env_scoped(negotiator_type, "NUM_RETRIES", mt)
        if env_retries is not None:
            num_retries_value = int(env_retries)

    return ResolvedLLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key_value,
        api_base=api_base_value,
        temperature=temperature_value,
        max_tokens=max_tokens_value,
        timeout=timeout_value,
        num_retries=num_retries_value,
        effort=effort_value,
    )


def effective_llm_config(
    obj: object, model_type: str | None = None
) -> ResolvedLLMConfig:
    """The LLM config an object should use for a call, honoring model types.

    When ``model_type`` is ``None`` this returns the object's own
    construction-time settings (``obj.provider``, ``obj.model``, …) unchanged.
    When a model type is requested it re-resolves through
    :func:`resolve_llm_config` using the object's concrete class name and its
    ``DEFAULT_PROVIDER``/``DEFAULT_MODEL``/``LOCK_PROVIDER`` class attributes, so
    different parts of the same negotiator can request different tiers.

    Args:
        obj: A negotiator/component instance carrying the resolved fields and the
            ``DEFAULT_PROVIDER``/``DEFAULT_MODEL``/``LOCK_PROVIDER`` attributes.
        model_type: Optional tier name (e.g. ``"fast"``).

    Returns:
        The :class:`ResolvedLLMConfig` to use for this call.
    """
    if model_type is None:
        return ResolvedLLMConfig(
            provider=getattr(obj, "provider", ""),
            model=getattr(obj, "model", ""),
            api_key=getattr(obj, "api_key", None),
            api_base=getattr(obj, "api_base", None),
            temperature=getattr(obj, "temperature", None),
            max_tokens=getattr(obj, "max_tokens", None),
            timeout=getattr(obj, "timeout", None),
            num_retries=getattr(obj, "num_retries", None),
            effort=getattr(obj, "effort", None),
        )
    # Re-resolve for the requested tier. The instance's own construction-time
    # settings (not just the class defaults) are the fallback floor, so an
    # explicitly-configured provider/model/api_base/etc. is preserved on tier
    # calls; a tier-suffixed env var still overrides it.
    tier = resolve_llm_config(
        type(obj).__name__,
        model_type=model_type,
        default_provider=(
            getattr(obj, "provider", None)
            or getattr(obj, "DEFAULT_PROVIDER", None)
            or DEFAULT_PROVIDER
        ),
        default_model=getattr(obj, "model", None)
        or getattr(obj, "DEFAULT_MODEL", None),
        lock_provider=getattr(obj, "LOCK_PROVIDER", False),
    )

    return ResolvedLLMConfig(
        provider=tier.provider,
        model=tier.model,
        api_key=tier.api_key
        if tier.api_key is not None
        else getattr(obj, "api_key", None),
        api_base=tier.api_base
        if tier.api_base is not None
        else getattr(obj, "api_base", None),
        temperature=tier.temperature
        if tier.temperature is not None
        else getattr(obj, "temperature", None),
        max_tokens=tier.max_tokens
        if tier.max_tokens is not None
        else getattr(obj, "max_tokens", None),
        timeout=tier.timeout
        if tier.timeout is not None
        else getattr(obj, "timeout", None),
        num_retries=tier.num_retries
        if tier.num_retries is not None
        else getattr(obj, "num_retries", None),
        effort=tier.effort if tier.effort is not None else getattr(obj, "effort", None),
    )

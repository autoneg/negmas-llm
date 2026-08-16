"""JSON <-> negmas conversions for the MCP layer.

Imports only from ``negmas`` so this module stays extractable.
"""

from __future__ import annotations

from typing import Any

from negmas.outcomes import (
    CartesianOutcomeSpace,
    Issue,
    Outcome,
    make_issue,
    make_os,
)
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun

__all__ = [
    "ScenarioError",
    "build_issues",
    "build_outcome_space",
    "build_ufun",
    "describe_outcome_space",
    "outcome_from_json",
    "outcome_to_json",
]


class ScenarioError(ValueError):
    """A scenario/outcome spec could not be interpreted."""


def build_issues(specs: list[dict[str, Any]]) -> list[Issue]:
    """Build issues from ``[{"name": ..., "values": [...] | {"min":..,"max":..}}]``."""
    if not specs:
        raise ScenarioError("at least one issue is required")
    issues: list[Issue] = []
    for spec in specs:
        name = spec.get("name")
        if not name:
            raise ScenarioError("every issue needs a 'name'")
        values = spec.get("values")
        if isinstance(values, dict):
            if "min" not in values or "max" not in values:
                raise ScenarioError(f"range issue {name!r} needs both 'min' and 'max'")
            lo, hi = values["min"], values["max"]
            if lo > hi:
                raise ScenarioError(f"issue {name!r}: min {lo} > max {hi}")
            issues.append(make_issue((lo, hi), name=name))
        elif isinstance(values, list):
            if not values:
                raise ScenarioError(f"issue {name!r} has no values")
            issues.append(make_issue(values, name=name))
        else:
            raise ScenarioError(
                f"issue {name!r}: 'values' must be a list or a "
                "{'min':..,'max':..} object"
            )
    names = [i.name for i in issues]
    if len(set(names)) != len(names):
        raise ScenarioError(f"duplicate issue names: {names}")
    return issues


def build_outcome_space(specs: list[dict[str, Any]]) -> CartesianOutcomeSpace:
    """Build a Cartesian outcome space from issue specs."""
    return make_os(build_issues(specs))


def _coerce(issue: Issue, value: Any) -> Any:
    """Coerce a JSON value to ``issue``'s value type.

    JSON object keys are always strings, so ``{"100": 0.5}`` on an int issue must
    be matched back to ``100``.
    """
    if issue.is_discrete():
        allowed = list(issue.all)
        if value in allowed:
            return value
        # match by string form -- covers int/float values arriving as JSON keys
        matches = [v for v in allowed if str(v) == str(value)]
        if len(matches) == 1:
            return matches[0]
        raise ScenarioError(
            f"issue {issue.name!r}: {value!r} is not one of {allowed[:10]}"
        )
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"issue {issue.name!r}: {value!r} is not numeric") from exc
    if not (issue.min_value <= coerced <= issue.max_value):
        raise ScenarioError(
            f"issue {issue.name!r}: {coerced} outside "
            f"[{issue.min_value}, {issue.max_value}]"
        )
    return coerced


def build_ufun(
    spec: dict[str, Any], outcome_space: CartesianOutcomeSpace
) -> LinearAdditiveUtilityFunction:
    """Build a linear-additive ufun.

    ``spec`` is ``{"weights": {issue: w}, "values": {issue: [...] | {value: u}},
    "reserved_value": float}``. Per-issue ``values`` may be a list aligned with the
    issue's declared value order, or a mapping from issue value to utility.
    """
    issues = {i.name: i for i in outcome_space.issues}
    raw_values = spec.get("values") or {}
    if not isinstance(raw_values, dict):
        raise ScenarioError("ufun 'values' must be an object keyed by issue name")
    unknown = set(raw_values) - set(issues)
    if unknown:
        raise ScenarioError(f"ufun 'values' names unknown issues: {sorted(unknown)}")

    values: dict[str, dict[Any, float]] = {}
    for name, issue in issues.items():
        entry = raw_values.get(name)
        if entry is None:
            if not issue.is_discrete():
                raise ScenarioError(f"issue {name!r} is continuous and needs 'values'")
            allowed = list(issue.all)
            span = max(len(allowed) - 1, 1)
            values[name] = {v: i / span for i, v in enumerate(allowed)}
            continue
        if isinstance(entry, list):
            if not issue.is_discrete():
                raise ScenarioError(
                    f"issue {name!r} is continuous; give 'values' as a mapping"
                )
            allowed = list(issue.all)
            if len(entry) != len(allowed):
                raise ScenarioError(
                    f"issue {name!r}: got {len(entry)} utilities for "
                    f"{len(allowed)} values"
                )
            values[name] = {v: float(u) for v, u in zip(allowed, entry, strict=True)}
        elif isinstance(entry, dict):
            values[name] = {_coerce(issue, k): float(v) for k, v in entry.items()}
        else:
            raise ScenarioError(f"issue {name!r}: 'values' must be a list or object")

    raw_weights = spec.get("weights") or {}
    if not isinstance(raw_weights, dict):
        raise ScenarioError("ufun 'weights' must be an object keyed by issue name")
    unknown = set(raw_weights) - set(issues)
    if unknown:
        raise ScenarioError(f"ufun 'weights' names unknown issues: {sorted(unknown)}")
    weights = {name: float(raw_weights.get(name, 1.0)) for name in issues}

    try:
        reserved = float(spec.get("reserved_value", 0.0))
    except (TypeError, ValueError) as exc:
        raise ScenarioError("'reserved_value' must be a number") from exc

    return LinearAdditiveUtilityFunction(
        values={name: TableFun(table) for name, table in values.items()},
        weights=weights,
        outcome_space=outcome_space,
        reserved_value=reserved,
    )


def outcome_to_json(
    outcome: Outcome | None, issues: list[Issue]
) -> dict[str, Any] | None:
    """Render an outcome tuple as ``{issue_name: value}``."""
    if outcome is None:
        return None
    return {issue.name: value for issue, value in zip(issues, outcome, strict=False)}


def outcome_from_json(obj: Any, issues: list[Issue]) -> Outcome:
    """Parse an outcome from a ``{issue_name: value}`` object or a positional list."""
    if obj is None:
        raise ScenarioError("an outcome is required")
    if isinstance(obj, list | tuple):
        if len(obj) != len(issues):
            raise ScenarioError(f"expected {len(issues)} values, got {len(obj)}")
        return tuple(_coerce(i, v) for i, v in zip(issues, obj, strict=True))
    if not isinstance(obj, dict):
        raise ScenarioError(
            f"outcome must be an object or list, got {type(obj).__name__}"
        )
    missing = [i.name for i in issues if i.name not in obj]
    if missing:
        raise ScenarioError(f"outcome is missing issues: {missing}")
    unknown = set(obj) - {i.name for i in issues}
    if unknown:
        raise ScenarioError(f"outcome names unknown issues: {sorted(unknown)}")
    return tuple(_coerce(i, obj[i.name]) for i in issues)


def describe_outcome_space(outcome_space: CartesianOutcomeSpace) -> dict[str, Any]:
    """Summarize an outcome space for an LLM."""
    issues: list[dict[str, Any]] = []
    for issue in outcome_space.issues:
        entry: dict[str, Any] = {"name": issue.name, "type": type(issue).__name__}
        if issue.is_discrete():
            entry["values"] = list(issue.all)
        else:
            entry["range"] = {"min": issue.min_value, "max": issue.max_value}
        issues.append(entry)
    return {"issues": issues, "cardinality": str(outcome_space.cardinality)}

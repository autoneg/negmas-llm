"""MCP tool definitions exposing negmas / negmas-llm negotiators."""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server import MCPServer
from negmas.registry import negotiator_registry
from negmas.sao import SAOMechanism

from negmas_llm.mcp.scenario import (
    build_outcome_space,
    build_ufun,
    describe_outcome_space,
    outcome_from_json,
    outcome_to_json,
)
from negmas_llm.mcp.session import (
    NegotiationSession,
    SessionStore,
    create_negotiator,
)

__all__ = ["make_server"]

_INSTRUCTIONS = """\
Consult negmas negotiation strategies. Use list_negotiators to find one, \
start_negotiation to open a session against it, then counter repeatedly. \
Genius (Java-bridge) negotiators are hidden by default because they need an \
external bridge running."""


def _find_info(name: str) -> Any | None:
    """Resolve a registry entry by unique key or short name."""
    info = negotiator_registry.get(name)
    if info is not None:
        return info
    matches = negotiator_registry.get_by_short_name(name) or []
    return matches[0] if matches else None


def _params_of(cls: type) -> list[str]:
    """Named constructor parameters, gathered across the MRO.

    Negotiators typically declare ``__init__(self, *args, **kwargs)`` and put the
    real parameters on base classes, so the own signature alone is empty.
    """
    seen: list[str] = []
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic signatures
            continue
        for pname, param in params.items():
            if pname == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if pname not in seen:
                seen.append(pname)
    return seen


def _summarize(key: str, info: Any) -> dict[str, Any]:
    doc = (info.description or "").strip()
    return {
        "name": info.short_name,
        "key": key,
        "source": info.source,
        "tags": sorted(info.tags),
        "description": doc.split("\n")[0][:200],
    }


def make_server(name: str = "negmas", store: SessionStore | None = None) -> MCPServer:
    """Build the MCP server. Pass ``store`` to share session state with tests."""
    mcp = MCPServer(name, instructions=_INSTRUCTIONS)
    sessions = store if store is not None else SessionStore()

    @mcp.tool()
    def list_negotiators(
        tags: list[str] | None = None,
        any_tags: list[str] | None = None,
        source: str | None = None,
        include_genius: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List available negotiation strategies.

        Args:
            tags: keep only negotiators having all of these tags.
            any_tags: keep only negotiators having at least one of these tags.
            source: filter by package, e.g. "negmas" or "negmas-llm".
            include_genius: include Genius negotiators, which need a running
                Java bridge and will fail to start without one.
            limit: maximum number returned.
        """
        found = negotiator_registry.query(
            tags=set(tags) if tags else None,
            any_tags=set(any_tags) if any_tags else None,
            exclude_tags=None if include_genius else {"genius"},
        )
        items = [_summarize(k, v) for k, v in found.items()]
        if source:
            items = [i for i in items if i["source"] == source]
        items.sort(key=lambda i: i["name"])
        return {
            "total": len(items),
            "returned": min(len(items), max(limit, 0)),
            "negotiators": items[: max(limit, 0)],
        }

    @mcp.tool()
    def describe_negotiator(name: str) -> dict[str, Any]:
        """Describe one negotiator: its type, tags, and constructor parameters.

        Args:
            name: negotiator name as returned by list_negotiators.
        """
        cls = negotiator_registry.get_class(name)
        if cls is None:
            raise ValueError(f"unknown negotiator {name!r}; use list_negotiators")
        info = _find_info(name)
        return {
            "name": info.short_name if info else name,
            "full_type_name": f"{cls.__module__}.{cls.__qualname__}",
            "source": info.source if info else None,
            "tags": sorted(info.tags) if info else [],
            "parameters": _params_of(cls),
            "doc": (inspect.getdoc(cls) or "").strip()[:1500],
        }

    @mcp.tool()
    def start_negotiation(
        negotiator: str,
        issues: list[dict[str, Any]],
        my_ufun: dict[str, Any],
        opponent_ufun: dict[str, Any],
        n_steps: int = 20,
        time_limit: float | None = None,
        negotiator_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Open a negotiation session against a negotiator. You move first.

        Args:
            negotiator: name from list_negotiators.
            issues: e.g. [{"name": "price", "values": [100, 150, 200]},
                {"name": "delivery", "values": {"min": 1, "max": 30}}].
            my_ufun: your utility, as {"weights": {issue: w}, "values":
                {issue: [u, ...] or {value: u}}, "reserved_value": float}.
                Utilities are per issue value; weights combine them.
            opponent_ufun: the opponent's utility, same shape. The opponent
                needs one to negotiate.
            n_steps: total party-turns. The two sides alternate, so you get
                about half of them.
            time_limit: optional wall-clock limit in seconds.
            negotiator_params: extra constructor arguments, e.g.
                {"model": "gpt-4o"} for an LLM negotiator.
        """
        outcome_space = build_outcome_space(issues)
        session = NegotiationSession(
            opponent=create_negotiator(negotiator, negotiator_params),
            opponent_name=negotiator,
            outcome_space=outcome_space,
            my_ufun=build_ufun(my_ufun, outcome_space),
            opponent_ufun=build_ufun(opponent_ufun, outcome_space),
            n_steps=n_steps,
            time_limit=time_limit,
        )
        sessions.add(session)
        state = session.state_dict()
        state["outcome_space"] = describe_outcome_space(outcome_space)
        state["your_turn"] = True
        return state

    @mcp.tool()
    def counter(
        session_id: str, response: str, outcome: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Play one turn, then return the opponent's reply.

        Note two negmas semantics: n_steps counts party-turns, so you get about
        half of them; and offering back the opponent's standing offer accepts
        it and ends the negotiation.

        Args:
            session_id: from start_negotiation.
            response: "offer" to counter-offer, "accept" to take the
                opponent's standing offer, "end" to walk away.
            outcome: required for "offer", as {issue_name: value}.
        """
        session = sessions.get(session_id)
        parsed = (
            outcome_from_json(outcome, session.issues) if response == "offer" else None
        )
        return session.counter(response, parsed)

    @mcp.tool()
    def negotiation_state(session_id: str) -> dict[str, Any]:
        """Get the current state and full offer history of a session.

        Args:
            session_id: from start_negotiation.
        """
        session = sessions.get(session_id)
        state = session.state_dict()
        state["trace"] = session.trace()
        return state

    @mcp.tool()
    def end_negotiation(session_id: str) -> dict[str, Any]:
        """Abandon a session and free its resources.

        Args:
            session_id: from start_negotiation.
        """
        sessions.remove(session_id)
        return {"session_id": session_id, "ended": True}

    @mcp.tool()
    def run_negotiation(
        negotiator_a: str,
        negotiator_b: str,
        issues: list[dict[str, Any]],
        ufun_a: dict[str, Any],
        ufun_b: dict[str, Any],
        n_steps: int = 40,
        time_limit: float | None = None,
        params_a: dict[str, Any] | None = None,
        params_b: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run two negotiators against each other and return the result.

        Slow, and costs money if either side is an LLM negotiator.

        Args:
            negotiator_a: first negotiator's name.
            negotiator_b: second negotiator's name.
            issues: as in start_negotiation.
            ufun_a: utility for negotiator_a, as in start_negotiation.
            ufun_b: utility for negotiator_b.
            n_steps: negotiation rounds; both sides act in each one. (This
                differs from start_negotiation, where n_steps counts
                party-turns.)
            time_limit: optional wall-clock limit in seconds.
            params_a: extra constructor arguments for negotiator_a.
            params_b: extra constructor arguments for negotiator_b.
        """
        outcome_space = build_outcome_space(issues)
        issue_list = list(outcome_space.issues)
        ua = build_ufun(ufun_a, outcome_space)
        ub = build_ufun(ufun_b, outcome_space)
        a = create_negotiator(negotiator_a, params_a)
        b = create_negotiator(negotiator_b, params_b)

        mechanism = SAOMechanism(
            outcome_space=outcome_space, n_steps=n_steps, time_limit=time_limit
        )
        mechanism.add(a, ufun=ua)
        mechanism.add(b, ufun=ub)
        mechanism.run()

        state = mechanism.state
        agreement = state.agreement
        names = {a.id: negotiator_a, b.id: negotiator_b}
        return {
            "agreement": outcome_to_json(agreement, issue_list),
            "utilities": {
                negotiator_a: round(
                    float(ua(agreement)) if agreement else float(ua.reserved_value), 6
                ),
                negotiator_b: round(
                    float(ub(agreement)) if agreement else float(ub.reserved_value), 6
                ),
            },
            "steps": state.step,
            "timedout": state.timedout,
            "broken": state.broken,
            "trace": [
                {
                    "step": step,
                    "by": names.get(nid, nid),
                    "outcome": outcome_to_json(outcome, issue_list),
                }
                for step, nid, outcome in mechanism.extended_trace
            ],
        }

    return mcp

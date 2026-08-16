"""MCP server exposing negmas and negmas-llm negotiators as LLM tools.

Complements :mod:`negmas_llm.ufun_tools`, which points the other way (in-process
tools for a negotiator's own LLM). Install with ``pip install negmas-llm[mcp]``
and run ``negmas-mcp``.

``scenario.py`` and ``session.py`` import only from ``negmas``; nothing here
imports the rest of ``negmas_llm``. Split this out as a standalone ``negmas-mcp``
package if it grows a CLI beyond the one entry point, needs its own release
cadence, or needs a different Python floor.
"""

from __future__ import annotations

try:
    import mcp as _mcp  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "negmas_llm.mcp needs the MCP SDK: pip install 'negmas-llm[mcp]'"
    ) from exc

from negmas_llm.mcp.scenario import (
    ScenarioError,
    build_issues,
    build_outcome_space,
    build_ufun,
    describe_outcome_space,
    outcome_from_json,
    outcome_to_json,
)
from negmas_llm.mcp.server import make_server
from negmas_llm.mcp.session import (
    NegotiationSession,
    SessionError,
    SessionStore,
    create_negotiator,
)

__all__ = [
    "NegotiationSession",
    "ScenarioError",
    "SessionError",
    "SessionStore",
    "build_issues",
    "build_outcome_space",
    "build_ufun",
    "create_negotiator",
    "describe_outcome_space",
    "make_server",
    "outcome_from_json",
    "outcome_to_json",
]

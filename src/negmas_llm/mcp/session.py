"""Mechanism-backed negotiation sessions driven from outside.

Each session runs a real :class:`~negmas.sao.SAOMechanism`, so time-based
strategies see genuine ``relative_time``. The MCP caller's side is a proxy that
is never consulted: every step that is its turn supplies an explicit action.

Imports only from ``negmas`` so this module stays extractable.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from negmas.outcomes import CartesianOutcomeSpace, Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.registry import negotiator_registry
from negmas.sao import (
    ResponseType,
    SAOMechanism,
    SAONegotiator,
    SAOResponse,
)

from negmas_llm.mcp.scenario import ScenarioError, outcome_to_json

__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_WAIT",
    "DEFAULT_SESSION_TTL",
    "NegotiationSession",
    "SessionError",
    "SessionStore",
    "create_negotiator",
]

#: Sessions kept before the oldest idle one is evicted.
DEFAULT_MAX_SESSIONS = 64
#: Seconds a session may sit idle before eviction.
DEFAULT_SESSION_TTL = 3600.0
#: Consecutive WAITs allowed before negmas ends the negotiation. negmas defaults
#: to ``sys.maxsize``, which lets a WAITing opponent spin forever -- and every
#: spin is a paid API call when that opponent is an LLM negotiator.
DEFAULT_MAX_WAIT = 8

_RESPONSES = {
    "offer": ResponseType.REJECT_OFFER,
    "accept": ResponseType.ACCEPT_OFFER,
    "end": ResponseType.END_NEGOTIATION,
}


class SessionError(RuntimeError):
    """A session could not be found, or was used in an invalid way."""


class _ProxyNegotiator(SAONegotiator):
    """Stands in for the MCP caller.

    Never consulted -- ``propose``/``respond`` raise so a plumbing bug surfaces
    loudly instead of silently inventing an offer.
    """

    def propose(self, state, dest=None):  # type: ignore[no-untyped-def]  # noqa: ARG002
        raise AssertionError("proxy negotiator consulted: no action was injected")

    def respond(self, state, source=None):  # type: ignore[no-untyped-def]  # noqa: ARG002
        raise AssertionError("proxy negotiator consulted: no action was injected")


def create_negotiator(name: str, params: dict[str, Any] | None = None) -> SAONegotiator:
    """Instantiate a registered negotiator by name."""
    if negotiator_registry.get_class(name) is None:
        raise ScenarioError(f"unknown negotiator {name!r}; use list_negotiators")
    try:
        return negotiator_registry.create(name, **(params or {}))
    except Exception as exc:
        raise ScenarioError(f"could not create {name!r}: {exc}") from exc


class NegotiationSession:
    """One negotiation the caller drives a single step at a time."""

    def __init__(
        self,
        *,
        opponent: SAONegotiator,
        opponent_name: str,
        outcome_space: CartesianOutcomeSpace,
        my_ufun: BaseUtilityFunction,
        opponent_ufun: BaseUtilityFunction,
        n_steps: int | None = 20,
        time_limit: float | None = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.opponent_name = opponent_name
        self.outcome_space = outcome_space
        self.my_ufun = my_ufun
        self.n_steps = n_steps
        self.max_wait = max_wait
        self.touched = time.monotonic()

        self.mechanism = SAOMechanism(
            outcome_space=outcome_space,
            n_steps=n_steps,
            time_limit=time_limit,
            one_offer_per_step=True,
            max_wait=max_wait,
        )
        self.proxy = _ProxyNegotiator(name="mcp-client")
        self.opponent = opponent
        # Add order decides who opens: the proxy must be added first so the
        # caller moves first. Asserted below so a reorder fails loudly.
        self.mechanism.add(self.proxy, ufun=my_ufun)
        self.mechanism.add(self.opponent, ufun=opponent_ufun)
        if self.proxy.id not in self.mechanism.next_negotitor_ids():
            raise SessionError("internal error: proxy is not the first to act")

    # -- helpers ---------------------------------------------------------

    @property
    def issues(self) -> list:
        return list(self.outcome_space.issues)

    def _utility(self, outcome: Outcome | None) -> float | None:
        if outcome is None:
            return None
        try:
            # rounded: raw floats carry binary noise like 0.48999999999999994
            return round(float(self.my_ufun(outcome)), 6)
        except Exception:
            return None

    def state_dict(self) -> dict[str, Any]:
        """Current state as JSON."""
        state = self.mechanism.state
        agreement = state.agreement
        return {
            "session_id": self.id,
            "opponent": self.opponent_name,
            # `running` is also False before the first step, so `ended` is the
            # terminal predicate.
            "done": state.ended,
            "step": state.step,
            "relative_time": round(float(state.relative_time), 6),
            "n_steps": self.n_steps,
            "current_offer": outcome_to_json(state.current_offer, self.issues),
            "current_offer_utility": self._utility(state.current_offer),
            "offer_is_from_opponent": state.current_proposer == self.opponent.id,
            "agreement": outcome_to_json(agreement, self.issues),
            "agreement_utility": self._utility(agreement),
            "timedout": state.timedout,
            "broken": state.broken,
        }

    def trace(self) -> list[dict[str, Any]]:
        """Full offer trace, most recent last."""
        return [
            {
                "step": step,
                "by": "opponent" if nid == self.opponent.id else "you",
                "outcome": outcome_to_json(outcome, self.issues),
                "your_utility": self._utility(outcome),
            }
            for step, nid, outcome in self.mechanism.extended_trace
        ]

    # -- driving ---------------------------------------------------------

    def _step_opponent(self) -> None:
        """Let the opponent act until it is the caller's turn again.

        A WAITing opponent is bounded by the mechanism's ``max_wait``; the guard
        here only backstops a mechanism that never yields the turn at all.
        """
        guard = min(2 * (self.max_wait + 2), 1000)
        for _ in range(guard):
            if self.mechanism.state.ended:
                return
            if self.proxy.id in self.mechanism.next_negotitor_ids():
                return
            self.mechanism.step()
        raise SessionError(f"opponent did not yield the turn within {guard} steps")

    def counter(self, response: str, outcome: Outcome | None = None) -> dict[str, Any]:
        """Play one action for the caller, then let the opponent reply.

        ``response`` is ``offer``, ``accept`` or ``end``.
        """
        self.touched = time.monotonic()
        if self.mechanism.state.ended:
            raise SessionError("this negotiation has already ended")
        if response not in _RESPONSES:
            raise ScenarioError(
                f"response must be one of {sorted(_RESPONSES)}, got {response!r}"
            )
        if response == "offer" and outcome is None:
            raise ScenarioError("response 'offer' requires an outcome")
        if response == "accept":
            outcome = self.mechanism.state.current_offer
            if outcome is None:
                raise ScenarioError("there is no offer to accept yet")

        if self.proxy.id not in self.mechanism.next_negotitor_ids():
            self._step_opponent()

        # Fresh dict every step: the mechanism pops from the action it is given.
        action = {self.proxy.id: SAOResponse(_RESPONSES[response], outcome)}
        result = self.mechanism.step(action)

        # Either side's step can end it -- with offering_is_accepting=True,
        # re-offering the standing offer accepts it.
        if result.running:
            self._step_opponent()
        return self.state_dict()

    def close(self) -> None:
        if not self.mechanism.state.ended:
            self.mechanism.abort()


class SessionStore:
    """Thread-safe session registry with a size cap and idle TTL."""

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl: float = DEFAULT_SESSION_TTL,
    ) -> None:
        self._sessions: dict[str, NegotiationSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl = ttl

    def add(self, session: NegotiationSession) -> NegotiationSession:
        with self._lock:
            self._evict()
            if len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.touched)
                self._drop(oldest.id)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> NegotiationSession:
        with self._lock:
            self._evict()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError(
                    f"unknown or expired session {session_id!r}; "
                    "call start_negotiation first"
                )
            session.touched = time.monotonic()
            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionError(f"unknown or expired session {session_id!r}")
            self._drop(session_id)

    def _drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [
            s.id for s in self._sessions.values() if now - s.touched > self.ttl
        ]:
            self._drop(sid)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

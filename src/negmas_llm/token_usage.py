"""Cumulative LLM token-consumption tracking, shared by every call site."""

from __future__ import annotations

from typing import Any

from attrs import define, field

__all__ = ["TokenUsage"]


@define
class TokenUsage:
    """Running totals of tokens and wall-clock time spent on ``litellm.completion`` calls.

    Attributes:
        prompt_tokens: Sum of input tokens across all calls counted so far.
        completion_tokens: Sum of output tokens across all calls counted so far.
        total_tokens: Sum of ``prompt_tokens + completion_tokens`` as reported
            per call (kept separate from the running sums above since a
            provider that omits ``usage`` entirely still counts as a call).
        seconds: Sum of wall-clock time spent inside ``litellm.completion``
            across all calls counted so far (the caller times each call and
            passes the elapsed seconds to :meth:`add`).
        calls: Number of ``litellm.completion`` calls counted so far, whether
            or not each one reported usage.
    """

    prompt_tokens: int = field(default=0)
    completion_tokens: int = field(default=0)
    total_tokens: int = field(default=0)
    seconds: float = field(default=0.0)
    calls: int = field(default=0)

    def add(self, response: Any, *, seconds: float = 0.0) -> None:
        """Fold in the usage and timing of one ``litellm.completion`` response.

        Some providers/models omit ``usage`` (or return it as ``None``); the
        call is still counted so ``calls`` reflects the true number of round
        trips made, even when no token figures are available for it. Fields
        are checked with ``isinstance(..., int)`` rather than a bare
        ``getattr(..., default) or 0``: a ``unittest.mock.MagicMock`` response
        (used throughout this project's own test suite) auto-vivifies
        ``.usage.prompt_tokens`` as a truthy mock object rather than raising
        or returning ``None``, which would otherwise silently replace a real
        running total with a mock via its supported ``__radd__``.

        Args:
            response: The raw ``litellm.completion`` return value.
            seconds: Wall-clock time the caller measured for this one call.
        """
        self.calls += 1
        self.seconds += seconds
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if isinstance(prompt, int):
            self.prompt_tokens += prompt
        if isinstance(completion, int):
            self.completion_tokens += completion
        if isinstance(total, int):
            self.total_tokens += total

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, convenient for JSON records."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
            "calls": self.calls,
        }

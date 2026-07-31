"""Unit tests for negmas_llm.summarize.maybe_summarize.

Pure-function tests: no negotiator, no mechanism, no mocked litellm --
just history lists and a fake ``raw_call``.
"""

from __future__ import annotations

from negmas_llm.summarize import maybe_summarize


def _exchange(user: str, assistant: str = "ok") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _history(n: int, content: str = "x") -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for i in range(n):
        history += _exchange(f"{content}-{i}")
    return history


def _raw_call_recording(calls: list[tuple[str, str]], summary: str = "SUMMARY"):
    def _call(system: str, user: str) -> str:
        calls.append((system, user))
        return summary

    return _call


def test_both_triggers_disabled_leaves_history_untouched():
    history = _history(50)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history,
        every=None,
        over_chars=None,
        keep=3,
        raw_call=_raw_call_recording(calls),
    )
    assert result is history
    assert calls == []


def test_under_both_thresholds_does_nothing():
    history = _history(5)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history,
        every=10,
        over_chars=10_000,
        keep=3,
        raw_call=_raw_call_recording(calls),
    )
    assert result == history
    assert calls == []


def test_every_trigger_fires_alone():
    history = _history(6)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history, every=5, over_chars=None, keep=2, raw_call=_raw_call_recording(calls)
    )
    assert len(calls) == 1
    assert len(result) // 2 == 3  # 1 summary exchange + keep=2
    assert "Summary of earlier turns" in result[0]["content"]


def test_over_chars_trigger_fires_alone_even_when_every_is_none():
    """The user's own scenario: no round-count trigger set, but the
    conversation has grown past a character-size threshold."""
    history = _history(4, content="y" * 50)  # well over 100 chars total
    calls: list[tuple[str, str]] = []
    total_chars = sum(len(m["content"]) for m in history)
    result = maybe_summarize(
        history,
        every=None,
        over_chars=total_chars - 1,
        keep=1,
        raw_call=_raw_call_recording(calls),
    )
    assert len(calls) == 1
    assert len(result) // 2 == 2  # 1 summary exchange + keep=1


def test_over_chars_trigger_does_not_fire_when_under_threshold():
    history = _history(4, content="y" * 50)
    total_chars = sum(len(m["content"]) for m in history)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history,
        every=None,
        over_chars=total_chars + 1,
        keep=1,
        raw_call=_raw_call_recording(calls),
    )
    assert result == history
    assert calls == []


def test_either_trigger_is_sufficient():
    """A high `every` that would not fire on its own, paired with a low
    `over_chars` that does -- summarization still runs."""
    history = _history(4, content="y" * 50)
    total_chars = sum(len(m["content"]) for m in history)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history,
        every=1000,  # nowhere near close to firing
        over_chars=total_chars - 1,  # already past this
        keep=1,
        raw_call=_raw_call_recording(calls),
    )
    assert len(calls) == 1
    assert len(result) // 2 == 2


def test_keep_is_respected_regardless_of_which_trigger_fired():
    history = _history(10)
    calls: list[tuple[str, str]] = []
    result = maybe_summarize(
        history, every=None, over_chars=1, keep=4, raw_call=_raw_call_recording(calls)
    )
    # 1 summary exchange + keep=4 verbatim exchanges
    assert len(result) // 2 == 5
    assert result[-8:] == history[-8:]


def test_raw_call_receives_the_collapsed_transcript_not_the_kept_part():
    history = _history(6)
    calls: list[tuple[str, str]] = []
    maybe_summarize(
        history, every=4, over_chars=None, keep=2, raw_call=_raw_call_recording(calls)
    )
    assert len(calls) == 1
    _, user_prompt = calls[0]
    # The 2 oldest exchanges (x-0, x-1) should be in the transcript...
    assert "x-0" in user_prompt
    assert "x-1" in user_prompt
    # ...but the 2 most recent (kept verbatim, not summarized) should not be.
    assert "x-4" not in user_prompt
    assert "x-5" not in user_prompt


def test_empty_summary_response_leaves_history_unchanged():
    history = _history(6)
    result = maybe_summarize(
        history, every=4, over_chars=None, keep=2, raw_call=lambda s, u: "   "
    )
    assert result == history


def test_summary_pair_replaces_only_the_collapsed_prefix():
    history = _history(6)
    result = maybe_summarize(
        history,
        every=4,
        over_chars=None,
        keep=2,
        raw_call=lambda s, u: "the gist",
    )
    assert result[0] == {
        "role": "user",
        "content": (
            "[Summary of earlier turns in this conversation, "
            "replacing them to save space]\nthe gist"
        ),
    }
    assert result[1] == {"role": "assistant", "content": "Understood."}
    assert result[2:] == history[-4:]

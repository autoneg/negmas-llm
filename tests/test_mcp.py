"""Tests for the MCP server exposing negmas negotiators."""

from __future__ import annotations

import itertools
import json
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import pytest
from negmas.registry import negotiator_registry
from negmas.sao import ResponseType, SAOMechanism, SAONegotiator

# the MCP SDK is an optional extra; skip the whole module without it
pytest.importorskip("mcp")

from negmas_llm.mcp import (  # noqa: E402
    NegotiationSession,
    ScenarioError,
    SessionError,
    SessionStore,
    build_issues,
    build_outcome_space,
    build_ufun,
    create_negotiator,
    make_server,
    outcome_from_json,
    outcome_to_json,
)

ISSUES: list[dict[str, Any]] = [
    {"name": "price", "values": [100, 150, 200]},
    {"name": "kind", "values": ["a", "b"]},
]
MY_UFUN: dict[str, Any] = {
    "weights": {"price": 0.7, "kind": 0.3},
    "values": {"price": [0.0, 0.5, 1.0], "kind": [0.2, 1.0]},
    "reserved_value": 0.0,
}
OPP_UFUN: dict[str, Any] = {
    "weights": {"price": 0.3, "kind": 0.7},
    "values": {"price": [1.0, 0.5, 0.0], "kind": [1.0, 0.0]},
    "reserved_value": 0.0,
}


def make_session(negotiator: str = "BoulwareTBNegotiator", n_steps: int = 10):
    """A session against `negotiator` over the shared test scenario."""
    outcome_space = build_outcome_space(ISSUES)
    return NegotiationSession(
        opponent=create_negotiator(negotiator),
        opponent_name=negotiator,
        outcome_space=outcome_space,
        my_ufun=build_ufun(MY_UFUN, outcome_space),
        opponent_ufun=build_ufun(OPP_UFUN, outcome_space),
        n_steps=n_steps,
    )


# =============================================================================
# 1. Spec round-trip
# =============================================================================


class TestScenario:
    def test_issue_kinds(self):
        issues = build_issues(
            [
                {"name": "price", "values": [100, 150, 200]},
                {"name": "kind", "values": ["a", "b"]},
                {"name": "days", "values": {"min": 1, "max": 30}},
                {"name": "ratio", "values": {"min": 0.0, "max": 1.0}},
            ]
        )
        assert [i.name for i in issues] == ["price", "kind", "days", "ratio"]
        assert issues[0].value_type is int
        assert issues[1].value_type is str
        assert not issues[3].is_discrete()

    @pytest.mark.parametrize(
        ("values", "outcome"),
        [
            ([100, 150, 200], 150),
            (["a", "b", "c"], "b"),
            ([0.5, 1.5, 2.5], 1.5),
        ],
    )
    def test_outcome_round_trip_preserves_type(self, values, outcome):
        issues = build_issues([{"name": "x", "values": values}])
        as_json = outcome_to_json((outcome,), issues)
        assert json.loads(json.dumps(as_json)) == as_json
        back = outcome_from_json(as_json, issues)
        assert back == (outcome,)
        assert type(back[0]) is type(outcome)

    def test_string_keyed_values_match_typed_ones(self):
        """JSON object keys are strings; they must still match int issue values."""
        outcome_space = build_outcome_space(ISSUES)
        stringy = build_ufun(
            {
                "weights": {"price": 0.7, "kind": 0.3},
                "values": {
                    "price": {"100": 0.0, "150": 0.5, "200": 1.0},
                    "kind": {"a": 0.2, "b": 1.0},
                },
            },
            outcome_space,
        )
        listy = build_ufun(MY_UFUN, outcome_space)
        for outcome in itertools.product(*(i.all for i in outcome_space.issues)):
            assert float(stringy(outcome)) == pytest.approx(float(listy(outcome)))

    def test_outcome_accepts_positional_list(self):
        issues = build_issues(ISSUES)
        assert outcome_from_json([200, "b"], issues) == (200, "b")

    def test_reserved_value_is_kept(self):
        outcome_space = build_outcome_space(ISSUES)
        ufun = build_ufun({**MY_UFUN, "reserved_value": 0.25}, outcome_space)
        assert ufun.reserved_value == pytest.approx(0.25)

    @pytest.mark.parametrize(
        "bad",
        [
            {"price": 999, "kind": "a"},  # not a legal value
            {"price": 100},  # missing issue
            {"price": 100, "kind": "a", "extra": 1},  # unknown issue
            [100],  # wrong arity
            None,
        ],
    )
    def test_bad_outcomes_raise_clearly(self, bad):
        issues = build_issues(ISSUES)
        with pytest.raises(ScenarioError):
            outcome_from_json(bad, issues)

    @pytest.mark.parametrize(
        "bad",
        [
            [],
            [{"name": "x"}],
            [{"values": [1, 2]}],
            [{"name": "x", "values": []}],
            [{"name": "x", "values": {"min": 5, "max": 1}}],
            [{"name": "x", "values": [1]}, {"name": "x", "values": [2]}],
        ],
    )
    def test_bad_issue_specs_raise_clearly(self, bad):
        with pytest.raises(ScenarioError):
            build_issues(bad)

    def test_ufun_rejects_unknown_issue(self):
        outcome_space = build_outcome_space(ISSUES)
        with pytest.raises(ScenarioError):
            build_ufun({"weights": {"nope": 1.0}}, outcome_space)

    def test_ufun_rejects_wrong_length_list(self):
        outcome_space = build_outcome_space(ISSUES)
        with pytest.raises(ScenarioError):
            build_ufun({"values": {"price": [0.0, 1.0]}}, outcome_space)


# =============================================================================
# 2. Non-perturbation: driving via the session matches a plain mechanism
# =============================================================================


class ScriptedNegotiator(SAONegotiator):
    """Offers a fixed sequence, then repeats the last outcome."""

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self._i = 0

    def propose(self, state, dest=None):
        outcome = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return outcome

    def respond(self, state, source=None):
        return ResponseType.REJECT_OFFER


SCRIPT = [(100, "b"), (150, "b"), (200, "b"), (200, "a"), (150, "a"), (100, "a")]


def test_session_does_not_perturb_the_mechanism():
    """The same script through the session and through a plain mechanism agree.

    This shows proxy injection does not disturb the mechanism. It cannot catch a
    fidelity bug affecting both runs identically -- that is what the time-fidelity
    test below is for.
    """
    session = make_session(n_steps=12)
    for outcome in SCRIPT:
        state = session.counter("offer", outcome)
        if state["done"]:
            break
    via_mcp = [(t["by"], t["outcome"]) for t in session.trace()]
    mcp_agreement = session.state_dict()["agreement"]

    outcome_space = build_outcome_space(ISSUES)
    mechanism = SAOMechanism(
        outcome_space=outcome_space, n_steps=12, one_offer_per_step=True
    )
    scripted = ScriptedNegotiator(SCRIPT, name="scripted")
    opponent = create_negotiator("BoulwareTBNegotiator")
    mechanism.add(scripted, ufun=build_ufun(MY_UFUN, outcome_space))
    mechanism.add(opponent, ufun=build_ufun(OPP_UFUN, outcome_space))
    mechanism.run()

    issues = list(outcome_space.issues)
    direct = [
        ("opponent" if nid == opponent.id else "you", outcome_to_json(o, issues))
        for _, nid, o in mechanism.extended_trace
    ]
    assert via_mcp == direct[: len(via_mcp)]
    assert mcp_agreement == outcome_to_json(mechanism.state.agreement, issues)


# =============================================================================
# 3. Time fidelity -- the guard against faking mechanism state
# =============================================================================


class TestTimeFidelity:
    def test_relative_time_advances_per_party_turn(self):
        session = make_session(n_steps=10)
        times = []
        while True:
            state = session.counter("offer", (200, "b"))
            times.append(state["relative_time"])
            if state["done"]:
                break
        assert times == sorted(times)
        assert times[0] > 0.0
        assert times[-1] > times[0]

    def test_caller_gets_about_half_of_n_steps(self):
        """n_steps counts party-turns, not rounds."""
        session = make_session(n_steps=10)
        turns = 0
        while True:
            turns += 1
            if session.counter("offer", (200, "b"))["done"]:
                break
        assert 4 <= turns <= 7, f"expected ~5 caller turns for n_steps=10, got {turns}"

    def test_time_based_opponent_concedes_over_time(self):
        """Boulware must hold near its best early and concede late.

        Against a hand-faked state it would concede on the wrong schedule, and
        this is what would catch it.
        """
        session = make_session(n_steps=30)
        outcome_space = build_outcome_space(ISSUES)
        opponent_ufun = build_ufun(OPP_UFUN, outcome_space)
        utilities = []
        while True:
            state = session.counter("offer", (200, "b"))
            offer = state["current_offer"]
            if offer is not None and state["offer_is_from_opponent"]:
                utilities.append(float(opponent_ufun((offer["price"], offer["kind"]))))
            if state["done"]:
                break
        assert len(utilities) >= 5
        # never concedes back upwards
        assert all(
            b <= a + 1e-9 for a, b in zip(utilities, utilities[1:], strict=False)
        )
        # starts at its maximum and ends strictly lower
        assert utilities[0] == pytest.approx(max(utilities))
        assert utilities[-1] < utilities[0]


def test_offering_back_the_standing_offer_accepts_it():
    """negmas runs with offering_is_accepting=True."""
    session = make_session(n_steps=20)
    # (200, "b") is the opponent's worst, so it counters rather than accepting
    state = session.counter("offer", (200, "b"))
    standing = state["current_offer"]
    assert standing is not None and not state["done"]

    state = session.counter("offer", (standing["price"], standing["kind"]))
    assert state["done"]
    assert state["agreement"] == standing


def test_reported_utilities_are_free_of_float_noise():
    """Raw ufun output carries noise like 0.48999999999999994."""
    session = make_session(n_steps=20)
    state = session.counter("offer", (200, "b"))
    for value in (state["current_offer_utility"], state["agreement_utility"]):
        if value is not None:
            assert value == round(value, 6)


def test_explicit_accept_ends_with_that_agreement():
    session = make_session(n_steps=20)
    state = session.counter("offer", (200, "b"))
    standing = state["current_offer"]
    state = session.counter("accept")
    assert state["done"]
    assert state["agreement"] == standing


def test_end_walks_away_without_agreement():
    session = make_session(n_steps=20)
    state = session.counter("end")
    assert state["done"]
    assert state["agreement"] is None


class WaitingNegotiator(SAONegotiator):
    """Always WAITs. negmas' default max_wait would let this spin forever."""

    calls = 0

    def propose(self, state, dest=None):
        return None

    def respond(self, state, source=None):
        type(self).calls += 1
        return ResponseType.WAIT


def test_a_waiting_opponent_cannot_spin_forever():
    """max_wait bounds a WAIT loop; without it every spin is a paid LLM call."""
    WaitingNegotiator.calls = 0
    outcome_space = build_outcome_space(ISSUES)
    session = NegotiationSession(
        opponent=WaitingNegotiator(name="waiter"),
        opponent_name="WaitingNegotiator",
        outcome_space=outcome_space,
        my_ufun=build_ufun(MY_UFUN, outcome_space),
        opponent_ufun=build_ufun(OPP_UFUN, outcome_space),
        n_steps=20,
        max_wait=4,
    )
    state = session.counter("offer", (200, "b"))
    assert state["done"], "a permanently waiting opponent must end the negotiation"
    assert WaitingNegotiator.calls <= 8, (
        f"opponent was consulted {WaitingNegotiator.calls} times for max_wait=4"
    )


class TestSessionErrors:
    def test_offer_requires_an_outcome(self):
        with pytest.raises(ScenarioError):
            make_session().counter("offer")

    def test_unknown_response(self):
        with pytest.raises(ScenarioError):
            make_session().counter("shrug", (100, "a"))

    def test_counter_after_end(self):
        session = make_session()
        session.counter("end")
        with pytest.raises(SessionError):
            session.counter("offer", (100, "a"))

    def test_accept_before_any_offer(self):
        with pytest.raises(ScenarioError):
            make_session().counter("accept")

    def test_unknown_negotiator(self):
        with pytest.raises(ScenarioError):
            create_negotiator("DefinitelyNotANegotiator")


class TestSessionStore:
    def test_get_and_remove(self):
        store = SessionStore()
        session = store.add(make_session())
        assert store.get(session.id) is session
        store.remove(session.id)
        with pytest.raises(SessionError):
            store.get(session.id)

    def test_unknown_id(self):
        with pytest.raises(SessionError):
            SessionStore().get("nope")

    def test_capacity_evicts_oldest(self):
        store = SessionStore(max_sessions=2)
        first = store.add(make_session())
        store.add(make_session())
        store.add(make_session())
        assert len(store) == 2
        with pytest.raises(SessionError):
            store.get(first.id)

    def test_idle_sessions_expire(self):
        store = SessionStore(ttl=-1.0)
        session = store.add(make_session())
        with pytest.raises(SessionError):
            store.get(session.id)


# =============================================================================
# 4. Registry listing
# =============================================================================


class TestListing:
    def test_genius_excluded_by_default(self):
        result = call("list_negotiators", {"limit": 1000})
        names = {n["name"] for n in result["negotiators"]}
        genius = negotiator_registry.query(tags={"genius"})
        assert genius, "expected Genius negotiators to be registered"
        assert names.isdisjoint({i.short_name for i in genius.values()})

    def test_genius_included_on_request(self):
        without = call("list_negotiators", {"limit": 1000})["total"]
        with_genius = call("list_negotiators", {"limit": 1000, "include_genius": True})[
            "total"
        ]
        assert with_genius > without

    def test_includes_both_packages(self):
        names = {
            n["name"] for n in call("list_negotiators", {"limit": 1000})["negotiators"]
        }
        assert {"BoulwareTBNegotiator", "AspirationNegotiator"} <= names
        assert {"LLMBoulwareTBNegotiator", "OllamaNegotiator"} <= names

    def test_source_filter(self):
        result = call("list_negotiators", {"limit": 1000, "source": "negmas-llm"})
        assert result["negotiators"]
        assert {n["source"] for n in result["negotiators"]} == {"negmas-llm"}

    def test_tag_filter(self):
        result = call("list_negotiators", {"limit": 1000, "tags": ["llm"]})
        assert result["negotiators"]
        assert all("llm" in n["tags"] for n in result["negotiators"])

    def test_limit_is_respected(self):
        result = call("list_negotiators", {"limit": 3})
        assert len(result["negotiators"]) == 3
        assert result["total"] > 3

    def test_describe_reports_registry_metadata(self):
        info = call("describe_negotiator", {"name": "OllamaNegotiator"})
        assert info["source"] == "negmas-llm"
        assert "llm" in info["tags"]
        assert "model" in info["parameters"]
        assert info["full_type_name"].startswith("negmas_llm.")

    def test_describe_unknown_is_an_error(self):
        assert call_raw("describe_negotiator", {"name": "Nope"}).is_error


# =============================================================================
# 5-6. Through the real MCP tool layer
# =============================================================================


def tool_text(result) -> str:
    """Join the text parts of a tool result (content items are a union type)."""
    return " ".join(text for c in result.content if (text := getattr(c, "text", "")))


def call_raw(tool: str, args: dict[str, Any], server=None):
    """Call a tool over an in-memory MCP client and return the raw result."""
    from mcp import Client

    async def run():
        async with Client(server if server is not None else make_server()) as client:
            return await client.call_tool(tool, args)

    return anyio.run(run)


def call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = call_raw(tool, args)
    assert not result.is_error, tool_text(result)
    return result.structured_content


def test_tools_are_exposed():
    from mcp import Client

    async def run():
        async with Client(make_server()) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert set(anyio.run(run)) == {
        "list_negotiators",
        "describe_negotiator",
        "start_negotiation",
        "counter",
        "negotiation_state",
        "end_negotiation",
        "run_negotiation",
    }


def test_full_negotiation_over_mcp():
    """Drive a whole negotiation through the MCP tool layer."""
    from mcp import Client

    async def run():
        store = SessionStore()
        server = make_server(store=store)
        async with Client(server) as client:
            started = await client.call_tool(
                "start_negotiation",
                {
                    "negotiator": "BoulwareTBNegotiator",
                    "issues": ISSUES,
                    "my_ufun": MY_UFUN,
                    "opponent_ufun": OPP_UFUN,
                    "n_steps": 12,
                },
            )
            assert not started.is_error, tool_text(started)
            session_id = started.structured_content["session_id"]
            assert started.structured_content["your_turn"]
            assert started.structured_content["outcome_space"]["issues"]

            states = []
            for _ in range(20):
                result = await client.call_tool(
                    "counter",
                    {
                        "session_id": session_id,
                        "response": "offer",
                        "outcome": {"price": 200, "kind": "b"},
                    },
                )
                assert not result.is_error, tool_text(result)
                states.append(result.structured_content)
                if result.structured_content["done"]:
                    break
            assert states and states[-1]["done"]

            final = await client.call_tool(
                "negotiation_state", {"session_id": session_id}
            )
            trace = final.structured_content["trace"]
            assert trace and {t["by"] for t in trace} == {"you", "opponent"}

            ended = await client.call_tool(
                "end_negotiation", {"session_id": session_id}
            )
            assert ended.structured_content["ended"]
            return states

    states = anyio.run(run)
    assert [s["relative_time"] for s in states] == sorted(
        s["relative_time"] for s in states
    )


def test_bad_input_surfaces_as_tool_error():
    assert call_raw(
        "start_negotiation",
        {
            "negotiator": "NoSuchNegotiator",
            "issues": ISSUES,
            "my_ufun": MY_UFUN,
            "opponent_ufun": OPP_UFUN,
        },
    ).is_error
    assert call_raw(
        "counter",
        {"session_id": "nope", "response": "offer", "outcome": {"price": 100}},
    ).is_error


def test_self_play_reaches_agreement():
    result = call(
        "run_negotiation",
        {
            "negotiator_a": "BoulwareTBNegotiator",
            "negotiator_b": "AspirationNegotiator",
            "issues": ISSUES,
            "ufun_a": MY_UFUN,
            "ufun_b": OPP_UFUN,
            "n_steps": 40,
        },
    )
    assert result["agreement"] is not None
    assert set(result["utilities"]) == {
        "BoulwareTBNegotiator",
        "AspirationNegotiator",
    }
    assert all(0.0 <= u <= 1.0 for u in result["utilities"].values())
    assert {t["by"] for t in result["trace"]} == {
        "BoulwareTBNegotiator",
        "AspirationNegotiator",
    }


# =============================================================================
# 7. An LLM negotiator behind the MCP tools, with litellm mocked
# =============================================================================


def mock_llm_response(content: str) -> MagicMock:
    """A litellm response with no tool calls.

    `tool_calls` must be None explicitly: `use_ufun_tools` defaults to True and a
    bare MagicMock attribute is truthy, so every call would look like a tool call.
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    return response


LLM_REPLY = json.dumps(
    {
        "response": "reject",
        "outcome": {"price": 100, "kind": "a"},
        "reasoning": "holding out",
    }
)


def test_llm_negotiator_over_mcp_with_mocked_litellm():
    """An LLM opponent's decision reaches the caller through the MCP tools."""
    from mcp import Client

    async def run():
        server = make_server(store=SessionStore())
        async with Client(server) as client:
            started = await client.call_tool(
                "start_negotiation",
                {
                    "negotiator": "OllamaNegotiator",
                    "issues": ISSUES,
                    "my_ufun": MY_UFUN,
                    "opponent_ufun": OPP_UFUN,
                    "n_steps": 6,
                },
            )
            assert not started.is_error, tool_text(started)
            result = await client.call_tool(
                "counter",
                {
                    "session_id": started.structured_content["session_id"],
                    "response": "offer",
                    "outcome": {"price": 200, "kind": "b"},
                },
            )
            assert not result.is_error, tool_text(result)
            return result.structured_content

    with patch("litellm.completion", return_value=mock_llm_response(LLM_REPLY)) as llm:
        state = anyio.run(run)
    assert llm.call_count > 0
    assert state["current_offer"] == {"price": 100, "kind": "a"}
    assert state["offer_is_from_opponent"]


def test_llm_negotiator_replies_through_a_session():
    with patch("litellm.completion", return_value=mock_llm_response(LLM_REPLY)) as llm:
        session = make_session("OllamaNegotiator", n_steps=6)
        state = session.counter("offer", (200, "b"))
    assert llm.call_count > 0
    assert state["current_offer"] == {"price": 100, "kind": "a"}


@pytest.mark.slow
@pytest.mark.ollama
def test_llm_negotiator_over_mcp_live(ollama_config):
    """The same path against a real LLM."""
    if not ollama_config.is_available():
        pytest.skip("ollama is not available")
    outcome_space = build_outcome_space(ISSUES)
    session = NegotiationSession(
        opponent=create_negotiator(
            "OllamaNegotiator",
            {"model": ollama_config.model, "api_base": ollama_config.api_base},
        ),
        opponent_name="OllamaNegotiator",
        outcome_space=outcome_space,
        my_ufun=build_ufun(MY_UFUN, outcome_space),
        opponent_ufun=build_ufun(OPP_UFUN, outcome_space),
        n_steps=6,
    )
    state = session.counter("offer", (200, "b"))
    assert state["current_offer"] is not None or state["done"]

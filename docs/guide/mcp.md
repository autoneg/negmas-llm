# MCP Server

Expose any registered negmas or negmas-llm negotiator to an MCP client (Claude
Desktop, Claude Code, or any agent framework) as a strategy oracle.

This is the mirror of [`negmas_llm.ufun_tools`](../api/negotiators.md), which
gives a negotiator's own LLM in-process tools. Here the LLM is the client and the
negotiator is the tool.

For a worked end-to-end example, see the [MCP tutorial](mcp-tutorial.md).

## Install and run

```bash
pip install 'negmas-llm[mcp]'
negmas-mcp                                  # stdio (default)
negmas-mcp --transport streamable-http --port 8000
```

Register it with an MCP client:

```json
{
  "mcpServers": {
    "negmas": {"command": "negmas-mcp"}
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `list_negotiators` | Browse strategies by tag or source. |
| `describe_negotiator` | Type, tags and constructor parameters of one. |
| `start_negotiation` | Open a session; you move first. |
| `counter` | Play a turn, get the opponent's reply. |
| `negotiation_state` | Current state plus the full offer trace. |
| `end_negotiation` | Drop a session. |
| `run_negotiation` | Run two negotiators against each other. |

## Scenarios

Issues take either a list of values or a numeric range:

```json
[{"name": "price", "values": [100, 150, 200]},
 {"name": "delivery", "values": {"min": 1, "max": 30}}]
```

A utility function gives per-value utilities plus issue weights. `values` may be
a list aligned with the issue's declared order, or a mapping:

```json
{"weights": {"price": 0.7, "delivery": 0.3},
 "values": {"price": [0.0, 0.5, 1.0], "delivery": {"1": 1.0, "30": 0.0}},
 "reserved_value": 0.1}
```

Both sides need a utility function: the opponent cannot negotiate without one.

## Two semantics worth knowing

Sessions run a real `SAOMechanism`, so time-based strategies such as Boulware
concede on a genuine schedule. Two consequences follow from negmas' defaults:

- **`n_steps` counts party-turns, not rounds.** The two sides alternate, so
  `n_steps=20` gives you about 10 offers.
- **Offering back the opponent's standing offer accepts it**
  (`offering_is_accepting=True`), ending the negotiation with that agreement.

## Genius negotiators

The registry includes Genius strategies, which need a running Java bridge.
`list_negotiators` hides them unless you pass `include_genius=true`.

## LLM negotiators

`negmas-llm`'s own negotiators are registered too, so you can negotiate against
`LLMBoulwareTBNegotiator` or `OllamaNegotiator`. Pass provider settings through
`negotiator_params`, e.g. `{"model": "gpt-4o"}`. These call an LLM on every turn,
so they are slower and may cost money.

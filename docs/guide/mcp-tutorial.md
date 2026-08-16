# MCP Tutorial

Negotiate a used-laptop purchase against a real negmas strategy, one turn at a
time, through the MCP tools. Every output below is from an actual run.

You are the **buyer**. Your opponent is `BoulwareTBNegotiator`, a strategy that
holds near its best offer and concedes late.

## Setup

```bash
pip install 'negmas-llm[mcp]'
negmas-mcp --help          # confirm the entry point exists
```

Register the server with your MCP client:

```json
{"mcpServers": {"negmas": {"command": "negmas-mcp"}}}
```

Restart the client. Seven tools should appear: `list_negotiators`,
`describe_negotiator`, `start_negotiation`, `counter`, `negotiation_state`,
`end_negotiation`, `run_negotiation`.

## Step 1 — Pick an opponent

Browse by tag:

```json
{"tool": "list_negotiators", "arguments": {"tags": ["time-based"], "limit": 5}}
```

```json
{
  "total": 6,
  "returned": 5,
  "negotiators": [
    {"name": "AspirationNegotiator", "source": "negmas",
     "description": "Represents a time-based negotiation strategy that is independent of the offers received during the negotiation."},
    {"name": "BoulwareTBNegotiator", "source": "negmas",
     "description": "A Boulware time-based negotiator that conceeds sub-linearly"},
    {"name": "ConcederTBNegotiator", "source": "negmas",
     "description": "A Boulware time-based negotiator that conceeds super-linearly"}
  ]
}
```

`describe_negotiator` shows what you can configure:

```json
{"tool": "describe_negotiator", "arguments": {"name": "BoulwareTBNegotiator"}}
```

```json
{
  "name": "BoulwareTBNegotiator",
  "full_type_name": "negmas.gb.negotiators.timebased.BoulwareTBNegotiator",
  "source": "negmas",
  "tags": ["boulware", "builtin", "propose", "respond", "sao", "time-based"],
  "parameters": ["offering_curve", "accepting_curve", "starting_utility", "..."]
}
```

Anything in `parameters` can be passed as `negotiator_params` in the next step.

## Step 2 — Describe the scenario

Three issues:

```json
[{"name": "price",    "values": [900, 1000, 1100, 1200]},
 {"name": "warranty", "values": [0, 1, 2]},
 {"name": "delivery", "values": ["standard", "express"]}]
```

Both sides need a utility function. `values` gives a utility per issue value
(here as a list, aligned with the order declared above); `weights` says how much
each issue matters; `reserved_value` is what you get by walking away.

```json
// you, the buyer -- price dominates
{"weights": {"price": 0.7, "warranty": 0.2, "delivery": 0.1},
 "values": {"price": [1.0, 0.7, 0.35, 0.0],
            "warranty": [0.0, 0.6, 1.0],
            "delivery": [0.0, 1.0]},
 "reserved_value": 0.3}

// the seller -- warranty dominates
{"weights": {"price": 0.25, "warranty": 0.65, "delivery": 0.1},
 "values": {"price": [0.0, 0.4, 0.75, 1.0],
            "warranty": [1.0, 0.5, 0.0],
            "delivery": [1.0, 0.0]},
 "reserved_value": 0.3}
```

The weights differ, and that is what makes a deal possible: warranty is cheap
for you to give up and expensive for the seller to give. Trading it away buys
you a lower price.

## Step 3 — Open a session

```json
{"tool": "start_negotiation", "arguments": {
  "negotiator": "BoulwareTBNegotiator",
  "issues": [...], "my_ufun": {...}, "opponent_ufun": {...},
  "n_steps": 40}}
```

```json
{
  "session_id": "acbee2cc76a64b68abe007867ccdba44",
  "done": false, "step": 0, "relative_time": 0.0, "n_steps": 40,
  "current_offer": null, "agreement": null, "your_turn": true,
  "outcome_space": {"issues": [...], "cardinality": "24"}
}
```

Keep the `session_id`. You move first.

!!! note "`n_steps` counts party-turns"
    The two sides alternate and each turn consumes a step, so `n_steps: 40`
    gives you about 20 offers, not 40.

## Step 4 — Trade the issue you care least about

Each `counter` plays your turn and returns the opponent's reply.

```json
{"tool": "counter", "arguments": {
  "session_id": "acbee...", "response": "offer",
  "outcome": {"price": 900, "warranty": 2, "delivery": "express"}}}
```

Open at your ideal, then concede warranty and delivery before touching price:

| Turn | Your offer | Seller's reply | `relative_time` | Your utility of their offer |
|---|---|---|---|---|
| 1 | 900 / 2 / express | 1200 / 0 / standard | 0.07 | 0.0 |
| 2 | 900 / 1 / express | 1200 / 0 / standard | 0.12 | 0.0 |
| 3 | 900 / 0 / express | 1200 / 0 / standard | 0.17 | 0.0 |
| 4 | 900 / 0 / standard | 1200 / 0 / standard | 0.22 | 0.0 |
| 5 | 1000 / 0 / standard | 1200 / 0 / standard | 0.27 | 0.0 |
| … | (hold) | 1200 / 0 / standard | … | 0.0 |
| 12 | 1000 / 0 / standard | **1100** / 0 / standard | 0.61 | 0.245 |
| 13 | 1000 / 0 / standard | 1200 / 0 / express | 0.66 | 0.1 |
| 14 | 1000 / 0 / standard | **1000 / 0 / standard** | 0.71 | 0.49 |

This is the Boulware shape: it repeats its best offer for eleven turns, then
concedes quickly once `relative_time` climbs. Conceding early against it buys
nothing — the session runs a real mechanism, so its clock is genuine.

At turn 14 it offered exactly what you had been asking for, which closes the
deal at **1000 / 0 / standard**, utility **0.49** against your 0.3 reserve.

## Step 5 — Closing a deal yourself

Had it stopped short of your offer, you would accept its standing offer:

```json
{"tool": "counter", "arguments": {"session_id": "acbee...", "response": "accept"}}
```

`"response": "end"` walks away with no agreement.

!!! warning "Re-offering their offer accepts it"
    Sending `"offer"` with the opponent's current outcome ends the negotiation
    with that agreement — negmas treats a matching offer as acceptance. Check
    `current_offer` before you repeat an outcome.

## Step 6 — Review the trace

```json
{"tool": "negotiation_state", "arguments": {"session_id": "acbee..."}}
```

```json
{"done": true,
 "agreement": {"price": 1000, "warranty": 0, "delivery": "standard"},
 "agreement_utility": 0.49,
 "trace": [
   {"step": 0, "by": "you", "outcome": {"price": 900, "warranty": 2, "delivery": "express"}, "your_utility": 1.0},
   {"step": 1, "by": "opponent", "outcome": {"price": 1200, "warranty": 0, "delivery": "standard"}, "your_utility": 0.0}
 ]}
```

Then release the session:

```json
{"tool": "end_negotiation", "arguments": {"session_id": "acbee..."}}
```

## Step 7 — Check yourself against a machine

`run_negotiation` plays two strategies against each other on the same scenario,
which tells you whether your 0.49 was any good:

```json
{"tool": "run_negotiation", "arguments": {
  "negotiator_a": "BoulwareTBNegotiator", "negotiator_b": "AspirationNegotiator",
  "issues": [...], "ufun_a": {...buyer...}, "ufun_b": {...seller...},
  "n_steps": 30}}
```

```json
{"agreement": {"price": 900, "warranty": 0, "delivery": "standard"},
 "utilities": {"BoulwareTBNegotiator": 0.7, "AspirationNegotiator": 0.75},
 "steps": 26}
```

A Boulware buyer got **0.70** where you got 0.49 — it held out longer and won
the price issue outright. Swap in `ConcederTBNegotiator` or `NaiveTitForTatNegotiator`
to see how the frontier shifts.

!!! note "`n_steps` means rounds here"
    In `run_negotiation` both sides act within each step, so `n_steps` counts
    rounds — unlike `start_negotiation`, where it counts party-turns.

## Negotiating against an LLM

negmas-llm's own negotiators are registered, so they work as opponents too:

```json
{"tool": "start_negotiation", "arguments": {
  "negotiator": "OllamaNegotiator",
  "negotiator_params": {"model": "qwen2.5:7b"},
  "issues": [...], "my_ufun": {...}, "opponent_ufun": {...}, "n_steps": 20}}
```

`list_negotiators` with `{"source": "negmas-llm"}` shows them all, including the
`LLM*` meta-negotiators that wrap a classic strategy. Each turn costs an LLM
call, so keep `n_steps` small.

## Without an MCP client

The same machinery is importable:

```python
from negmas_llm.mcp import (
    NegotiationSession,
    build_outcome_space,
    build_ufun,
    create_negotiator,
)

space = build_outcome_space(ISSUES)
session = NegotiationSession(
    opponent=create_negotiator("BoulwareTBNegotiator"),
    opponent_name="BoulwareTBNegotiator",
    outcome_space=space,
    my_ufun=build_ufun(BUYER, space),
    opponent_ufun=build_ufun(SELLER, space),
    n_steps=40,
)

state = session.counter("offer", (900, 2, "express"))
while not state["done"] and (state["current_offer_utility"] or 0) < 0.3:
    state = session.counter("offer", (1000, 0, "standard"))

print(state["agreement"], state["agreement_utility"])
# {'price': 1000, 'warranty': 0, 'delivery': 'standard'} 0.49
```

Outcomes are plain tuples here, in issue order.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `unknown or expired session` | Sessions are capped at 64 and expire after an hour idle. Start a new one. |
| `999 is not one of [900, 1000, 1100, 1200]` | Offers must use declared issue values. |
| `outcome is missing issues: ['delivery']` | Every issue must appear in an offer. |
| Negotiation ended sooner than expected | You re-offered the opponent's standing offer, or `n_steps` ran out. |
| A negotiator fails to start | Genius strategies need a running Java bridge; they are hidden unless you pass `include_genius: true`. |

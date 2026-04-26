# Quick Start

This guide walks you through creating your first LLM-powered negotiation.

## Basic Example

```python
from negmas import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.outcomes import make_issue
from negmas_llm import OpenAINegotiator

# Step 1: Define the negotiation issues
issues = [
    make_issue("price", (0, 100)),      # Price from 0 to 100
    make_issue("quantity", (1, 10)),    # Quantity from 1 to 10
]

# Step 2: Create utility functions
# Seller wants high price and high quantity
seller_ufun = LinearAdditiveUtilityFunction(
    weights={"price": 0.6, "quantity": 0.4},
    issues=issues,
)

# Buyer wants low price but high quantity
buyer_ufun = LinearAdditiveUtilityFunction(
    weights={"price": -0.6, "quantity": 0.4},
    issues=issues,
)

# Step 3: Create LLM negotiators
seller = OpenAINegotiator(
    model="gpt-4o",
    name="seller",
    ufun=seller_ufun,
)

buyer = OpenAINegotiator(
    model="gpt-4o",
    name="buyer", 
    ufun=buyer_ufun,
)

# Step 4: Run the negotiation
mechanism = SAOMechanism(outcome_space=issues, n_steps=20)
mechanism.add(seller)
mechanism.add(buyer)
result = mechanism.run()

# Step 5: Check results
if result.agreement:
    print(f"Agreement reached: {result.agreement}")
    print(f"Seller utility: {seller_ufun(result.agreement)}")
    print(f"Buyer utility: {buyer_ufun(result.agreement)}")
else:
    print("No agreement reached")
```

## Using Local Models

For local inference with Ollama:

```python
from negmas_llm import OllamaNegotiator

negotiator = OllamaNegotiator(
    model="llama3.2",
    name="local_negotiator",
    ufun=my_ufun,
)
```

## Providing Partner Information

You can give negotiators information about their opponent's utility function:

```python
seller = OpenAINegotiator(
    model="gpt-4o",
    name="seller",
    ufun=seller_ufun,
    private_info={"opponent_ufun": buyer_ufun},  # Share buyer's preferences
)
```

## Customizing LLM Parameters

```python
negotiator = OpenAINegotiator(
    model="gpt-4o",
    name="negotiator",
    ufun=my_ufun,
    temperature=0.5,      # Lower = more deterministic
    max_tokens=2048,      # Response length limit
    llm_kwargs={          # Additional litellm parameters
        "top_p": 0.9,
    },
)
```

## Next Steps

- [Providers Guide](../guide/providers.md) - Learn about all supported LLM providers
- [Customization Guide](../guide/customization.md) - Create custom negotiators
- [API Reference](../api/negotiators.md) - Full API documentation

#!/usr/bin/env python3
"""Test script to demonstrate the verbose parameter in LLM negotiators."""

from negmas import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.sao import SAOMechanism

from negmas_llm import LLMAspirationNegotiator, OllamaNegotiator

# Create a simple negotiation scenario
issues = [
    make_issue(name="price", values=10),
    make_issue(name="quality", values=5),
]
outcome_space = make_os(issues)

# Create utility functions for both negotiators
ufun1 = LUFun.random(outcome_space=outcome_space, reserved_value=0.0)
ufun2 = LUFun.random(outcome_space=outcome_space, reserved_value=0.0)

# Test 1: LLMNegotiator with verbose=True
print("\n" + "=" * 80)
print("TEST 1: OllamaNegotiator with verbose=True")
print("=" * 80)

neg1 = OllamaNegotiator(
    model="qwen3:4b-instruct",
    verbose=True,  # Enable verbose mode
    name="VerboseNegotiator",
    ufun=ufun1,
)

neg2 = OllamaNegotiator(
    model="qwen3:4b-instruct",
    verbose=False,  # Disable verbose mode
    name="QuietNegotiator",
    ufun=ufun2,
)

mechanism = SAOMechanism(outcome_space=outcome_space, n_steps=3)
mechanism.add(neg1)
mechanism.add(neg2)

print("\nRunning negotiation with verbose negotiator...")
print("You should see LLM prompts and responses below:\n")

mechanism.run()

print("\n" + "=" * 80)
print("TEST 2: LLMAspirationNegotiator (meta) with verbose=True")
print("=" * 80)

# Test 2: LLMMetaNegotiator with verbose=True
meta_neg1 = LLMAspirationNegotiator(
    provider="ollama",
    model="qwen3:4b-instruct",
    verbose=True,  # Enable verbose mode
    name="VerboseMetaNegotiator",
    ufun=ufun1,
)

meta_neg2 = LLMAspirationNegotiator(
    provider="ollama",
    model="qwen3:4b-instruct",
    verbose=False,  # Disable verbose mode
    name="QuietMetaNegotiator",
    ufun=ufun2,
)

mechanism2 = SAOMechanism(outcome_space=outcome_space, n_steps=3)
mechanism2.add(meta_neg1)
mechanism2.add(meta_neg2)

print("\nRunning negotiation with verbose meta negotiator...")
print("You should see LLM prompts and responses below:\n")

mechanism2.run()

print("\n" + "=" * 80)
print("Tests completed!")
print("=" * 80)

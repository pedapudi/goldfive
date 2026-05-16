"""Testkit utilities for downstream consumers of goldfive.

This package collects deterministic stand-ins for the moving parts an
optimization or evaluation harness needs to drive against goldfive: a
catalog of adversarial agents that exercise specific drift behaviours
(:mod:`goldfive.testkit.adversarial`) and a canned-response LLM stub
(:class:`CannedCallLLM`) that replaces a live ``call_llm`` with a
pre-recorded transcript so a tuning loop can replay traces offline.

Determinism is a load-bearing property here. The adversarial agents
respect :func:`goldfive.runtime.set_seed`, and the testkit avoids
wall-clock timing decisions everywhere — two runs with the same seed
and the same input produce byte-identical event streams.
"""

from __future__ import annotations

from goldfive.testkit.adversarial import (
    CleanAgent,
    HallucinatingAgent,
    LoopingAgent,
    RefusingAgent,
    RunawayDelegationAgent,
    SlowAgent,
    WanderingAgent,
)
from goldfive.testkit.canned_call_llm import (
    CannedCallLLM,
    CannedCallLLMExhausted,
)

__all__ = [
    "CannedCallLLM",
    "CannedCallLLMExhausted",
    "CleanAgent",
    "HallucinatingAgent",
    "LoopingAgent",
    "RefusingAgent",
    "RunawayDelegationAgent",
    "SlowAgent",
    "WanderingAgent",
]

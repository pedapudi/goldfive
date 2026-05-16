"""Built-in judge factory registry.

This module is the public name for the built-in judges. Operators
reach them as ``goldfive.builtin_judges.reasoning_drift()`` etc.
The factory functions live in :mod:`goldfive.judges.builtins`; this
module re-exports them at the documented surface so the wire-up
matches the issue-time contract.
"""

from __future__ import annotations

from goldfive.judges.builtins import (
    BuiltinJudge,
    default_judges,
    goal_drift,
    looping_reasoning,
    looping_tool,
    reasoning_drift,
    refusal,
    stop_reason,
    tool_error,
)

__all__ = [
    "BuiltinJudge",
    "default_judges",
    "goal_drift",
    "looping_reasoning",
    "looping_tool",
    "reasoning_drift",
    "refusal",
    "stop_reason",
    "tool_error",
]

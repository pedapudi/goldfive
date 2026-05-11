"""Optional integrations between goldfive and external SDKs.

Each submodule here wires an external LLM SDK to goldfive's
``CallLLM = Callable[[str, str, str], Awaitable[str]]`` contract so it
can be passed to :func:`goldfive.wrap`, :class:`LLMPlanner`,
:class:`LLMGoalDeriver`, or judges.

These imports are intentionally not re-exported at package level:
each integration has its own optional dependency (e.g. ``claude-agent-sdk``
for :mod:`goldfive.integrations.claude_sdk`) which we don't want to
force on goldfive's core users.
"""

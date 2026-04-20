"""``adk web`` integration example — drop goldfive into an ADK web app.

Since Phase 2 (issue #77) ``goldfive.wrap(adk_agent)`` returns a
polymorphic :class:`~goldfive.adapters.adk_wrap.GoldfiveADKAgent` that
*is* a ``google.adk.agents.BaseAgent``. That means ``adk web`` can load
it as ``root_agent`` directly — every user turn flows through goldfive's
goal-derive → plan → execute pipeline and comes back as a short stream
of ``Event`` s rendered in the ADK UI.

Usage::

    uv pip install -e '.[adk]'
    adk web examples/adk_web_wrapped.py

This file is loaded by ``adk web`` rather than executed directly — it
must expose an ``App`` via the module-level ``app`` binding that
``adk web`` discovers. It does not run a turn unless driven by the UI.

Programmatic use of the same object still works — see the final block
for an optional one-shot run.
"""

from __future__ import annotations

import os

try:
    from google.adk.agents import Agent
    from google.adk.apps.app import App
except ImportError as _adk_err:  # pragma: no cover
    raise SystemExit(
        "install goldfive[adk] to run this example"
    ) from _adk_err

import goldfive

# ---------------------------------------------------------------------------
# The "real" ADK agent — the one you'd write without goldfive.
# ---------------------------------------------------------------------------
# Keep this tree as vanilla ADK so it's easy to see what goldfive adds. For
# a non-trivial example, see ``examples/adk_presentation/agent.py``.
_MODEL = os.getenv("GOLDFIVE_EXAMPLE_MODEL", "gpt-4o-mini")

real_agent = Agent(
    name="coordinator",
    model=_MODEL,
    description="A thin demo agent that answers user questions.",
    instruction=(
        "You are a helpful assistant. Keep answers to under three "
        "sentences unless asked otherwise."
    ),
)

# ---------------------------------------------------------------------------
# One-line goldfive wrap. Same call site that works programmatically now
# also satisfies adk web's root_agent BaseAgent contract.
# ---------------------------------------------------------------------------
root_agent = goldfive.wrap(real_agent)

# ADK web discovers `app` at module level.
app = App(name="goldfive_wrapped_demo", root_agent=root_agent)


# ---------------------------------------------------------------------------
# Optional: demonstrate the programmatic call site still works. Leaving
# this under the ``if __name__ == "__main__"`` gate keeps ``adk web``
# load-time side-effect free while letting ``python examples/…`` drive a
# one-shot turn.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        outcome = await root_agent.run(
            "summarise goldfive's value proposition in one line"
        )
        print("success=", outcome.success, "reason=", outcome.reason)

    asyncio.run(_main())

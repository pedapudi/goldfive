"""adk_agent — wrap a Google ADK agent with goldfive.

Gated on the ``adk`` extra. Install with::

    uv pip install -e '.[adk]'

The actual ADK adapter lives in ``goldfive.adapters.adk`` (delivered by
issue #13). This example documents the intended wiring; it import-fails
cleanly when either ADK itself or the goldfive ADK adapter is not yet
installed.
"""

from __future__ import annotations

try:
    import google.adk  # type: ignore  # noqa: F401
except ImportError as _adk_err:  # pragma: no cover
    raise SystemExit("install goldfive[adk] to run this example") from _adk_err

try:
    from goldfive.adapters.adk import ADKAdapter  # type: ignore[attr-defined]
except ImportError as _adk_adapter_err:  # pragma: no cover
    raise SystemExit(
        "goldfive.adapters.adk is not available yet; see issue #13"
    ) from _adk_adapter_err

import asyncio

from goldfive import (
    InMemorySink,
    PassthroughGoalDeriver,
    Plan,
    Runner,
    SequentialExecutor,
    StaticPlanner,
    Task,
)


def build_plan() -> Plan:
    return Plan(
        id="adk-demo",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="ADK task 1", assignee_agent_id="root"),
            Task(id="t2", title="ADK task 2", assignee_agent_id="root"),
        ],
        edges=[],
        summary="Two ADK tasks.",
    )


async def main() -> None:
    # Construct an ADK root agent however you normally would, e.g.:
    #   from google.adk.agents import Agent as ADKAgent
    #   root_agent = ADKAgent(name="root", model="gemini-2.0-flash")
    #   adapter = ADKAdapter(root_agent)
    adapter = ADKAdapter.from_scratch(model="gemini-2.0-flash")  # placeholder

    sink = InMemorySink()
    runner = Runner(
        agent=adapter,
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("Run two ADK tasks"),
        sinks=[sink],
    )
    outcome = await runner.run("go")
    await runner.close()
    print(f"success={outcome.success}")
    for e in sink.events:
        kind = e["kind"] if isinstance(e, dict) else getattr(e, "kind", "?")
        payload = e.get("payload") if isinstance(e, dict) else getattr(e, "payload", {})
        print(f"  {kind:<16}  {payload}")


if __name__ == "__main__":
    asyncio.run(main())

"""claude_agent — wrap a Claude Agent SDK agent with goldfive.

Gated on the ``claude`` extra. Install with::

    uv pip install -e '.[claude]'

The actual Claude adapter lives in ``goldfive.adapters.claude``
(delivered by issue #14). This example documents the intended wiring;
it import-fails cleanly when the SDK or the goldfive Claude adapter
are not yet installed.
"""

from __future__ import annotations

try:
    import anthropic  # type: ignore  # noqa: F401
except ImportError as _claude_err:  # pragma: no cover
    raise SystemExit("install goldfive[claude] to run this example") from _claude_err

try:
    from goldfive.adapters.claude import ClaudeAdapter  # type: ignore[attr-defined]
except ImportError as _claude_adapter_err:  # pragma: no cover
    raise SystemExit(
        "goldfive.adapters.claude is not available yet; see issue #14"
    ) from _claude_adapter_err

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
        id="claude-demo",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Claude task 1", assignee_agent_id="claude"),
            Task(id="t2", title="Claude task 2", assignee_agent_id="claude"),
        ],
        edges=[],
        summary="Two Claude tasks.",
    )


async def main() -> None:
    adapter = ClaudeAdapter(model="claude-sonnet-4-5")  # placeholder

    sink = InMemorySink()
    runner = Runner(
        agent=adapter,
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("Run two Claude tasks"),
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

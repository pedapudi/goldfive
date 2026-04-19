"""hello_callable — the simplest possible goldfive run.

Wires a :class:`CallableAdapter` to a :class:`StaticPlanner` with a
hand-built plan, runs it through a :class:`SequentialExecutor`, and
collects every event in an :class:`InMemorySink` so the full event log
can be printed at the end.

Run with::

    uv run python examples/hello_callable.py
"""

from __future__ import annotations

import asyncio

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)


def build_plan() -> Plan:
    return Plan(
        id="hello-plan",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="greet", title="Greet the user", assignee_agent_id="greeter"),
            Task(id="ask", title="Ask for a name", assignee_agent_id="greeter"),
            Task(id="thank", title="Thank the user", assignee_agent_id="greeter"),
        ],
        edges=[
            TaskEdge(from_task_id="greet", to_task_id="ask"),
            TaskEdge(from_task_id="ask", to_task_id="thank"),
        ],
        summary="Greet, ask, thank.",
    )


async def greeter_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """A toy agent that produces one canned reply per task."""
    _ = tools  # unused in this toy agent
    text = {
        "greet": "Hello!",
        "ask": "What's your name?",
        "thank": "Thanks for chatting!",
    }.get(task.id, "(no-op)")
    return InvocationResult(task_id=task.id, text=text)


async def main() -> None:
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(greeter_agent, available_agents=["greeter"]),
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("Say hello, ask for a name, thank them"),
        sinks=[sink],
    )

    outcome = await runner.run("run the greeter workflow")
    await runner.close()

    print(f"success={outcome.success}, reason={outcome.reason!r}")
    print(f"run_id={outcome.session.run_id}")
    print(f"goals={[g.summary for g in outcome.session.goals]}")
    print(f"{len(sink.events)} events:")
    for e in sink.events:
        if isinstance(e, dict):
            seq = e.get("sequence", "?")
            kind = e.get("kind", "?")
            payload = e.get("payload", {})
        else:
            seq = getattr(e, "sequence", "?")
            oneof = e.WhichOneof("payload") if hasattr(e, "WhichOneof") else None
            kind = (
                "".join(part.capitalize() for part in oneof.split("_"))
                if oneof
                else "?"
            )
            payload = getattr(e, oneof) if oneof else "{}"
        print(f"  seq={seq:>3}  {kind:<16}  {payload}")


if __name__ == "__main__":
    asyncio.run(main())

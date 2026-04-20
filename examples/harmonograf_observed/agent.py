"""harmonograf_observed — CallableAdapter agent wired to a HarmonografSink.

A zero-LLM, four-task :class:`StaticPlanner` demo that ships every event
to both an :class:`InMemorySink` and a :class:`HarmonografSink`, so a
newcomer can see goldfive events light up the harmonograf UI without
standing up an ADK or Claude SDK agent first. If ``harmonograf_client``
is not installed, the script falls back to ``InMemorySink`` +
``LoggingSink`` and prints a pointer to the observability guide.

Run with::

    uv run python examples/harmonograf_observed/agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os

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
from goldfive.sinks import LoggingSink  # None when the proto extra is missing

try:
    from harmonograf_client import Client, HarmonografSink
except ImportError:
    Client = None  # type: ignore[assignment,misc]
    HarmonografSink = None  # type: ignore[assignment,misc]


def build_plan() -> Plan:
    return Plan(
        id="observed-plan",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Gather notes", assignee_agent_id="worker"),
            Task(id="draft", title="Draft summary", assignee_agent_id="worker"),
            Task(id="review", title="Review draft", assignee_agent_id="worker"),
            Task(id="publish", title="Publish result", assignee_agent_id="worker"),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="draft"),
            TaskEdge(from_task_id="draft", to_task_id="review"),
            TaskEdge(from_task_id="review", to_task_id="publish"),
        ],
        summary="Research, draft, review, publish.",
    )


async def worker_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = session, tools
    text = {
        "research": "Collected three bullet points.",
        "draft": "Drafted a two-paragraph summary.",
        "review": "Reviewed — looks good.",
        "publish": "Published to the demo channel.",
    }.get(task.id, "(no-op)")
    return InvocationResult(task_id=task.id, text=text)


async def main() -> None:
    server_addr = os.environ.get("HARMONOGRAF_SERVER", "127.0.0.1:7531")
    memory_sink = InMemorySink()
    sinks: list = [memory_sink]
    client = None

    if Client is not None and HarmonografSink is not None:
        client = Client(name="harmonograf_observed", server_addr=server_addr)
        sinks.append(HarmonografSink(client))
        active = f"[InMemorySink, HarmonografSink -> {server_addr}]"
    else:
        print(
            "harmonograf_client not installed — run with [InMemorySink] only. "
            "See docs/guides/observability-with-harmonograf.md to add the console."
        )
        if LoggingSink is not None:
            sinks.append(LoggingSink())
            active = "[InMemorySink, LoggingSink]"
        else:
            active = "[InMemorySink]"

    runner = Runner(
        agent=CallableAdapter(worker_agent, available_agents=["worker"]),
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(max_task_invocations=8),
        goal_deriver=PassthroughGoalDeriver("Research, draft, review, publish"),
        sinks=sinks,
    )

    outcome = await runner.run("run the observed workflow")
    await runner.close()
    if client is not None:
        client.shutdown(flush_timeout=5.0)

    print(f"success={outcome.success}, reason={outcome.reason!r}")
    print(f"run_id={outcome.session.run_id}")
    print(f"sinks={active}")
    print(f"InMemorySink captured {len(memory_sink.events)} events.")
    if client is not None:
        print(
            f"Open the harmonograf UI (server at {server_addr}) to inspect the "
            "plan, task timeline, and drift markers for this run."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
    )
    asyncio.run(main())

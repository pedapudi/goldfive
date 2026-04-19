"""multi_sink_fanout — fan one event stream out to three sinks at once.

Wires :class:`InMemorySink`, :class:`LoggingSink`, and
:class:`JSONLPersistenceSink` to the same :class:`Runner` so a single run
populates an in-process buffer, a stdout log, and an on-disk JSONL file
simultaneously. Demonstrates the additive nature of the ``sinks=`` list.

Run with::

    uv run python examples/multi_sink_fanout.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

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
from goldfive.sinks import JSONLPersistenceSink, LoggingSink

if JSONLPersistenceSink is None or LoggingSink is None:  # type: ignore[truthy-function]
    raise SystemExit(
        "multi_sink_fanout requires the `proto` extra; install goldfive[proto] "
        "and run `make proto` to generate the proto stubs, then retry."
    )


def build_plan() -> Plan:
    return Plan(
        id="fanout-plan",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="collect", title="Collect inputs", assignee_agent_id="worker"),
            Task(id="process", title="Process them", assignee_agent_id="worker"),
            Task(id="emit", title="Emit summary", assignee_agent_id="worker"),
        ],
        edges=[
            TaskEdge(from_task_id="collect", to_task_id="process"),
            TaskEdge(from_task_id="process", to_task_id="emit"),
        ],
        summary="Three step pipeline emitted to three sinks.",
    )


async def worker_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools
    return InvocationResult(task_id=task.id, text=f"finished: {task.title}")


async def main() -> None:
    tmp = tempfile.NamedTemporaryFile(
        prefix="goldfive-fanout-", suffix=".jsonl", delete=False
    )
    tmp.close()
    jsonl_path = tmp.name

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="[LoggingSink] %(message)s",
    )
    logger = logging.getLogger("goldfive.examples.fanout")

    memory_sink = InMemorySink()
    logging_sink = LoggingSink(logger=logger)
    jsonl_sink = JSONLPersistenceSink(jsonl_path, mode="write")

    runner = Runner(
        agent=CallableAdapter(worker_agent, available_agents=["worker"]),
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("Run a 3-step pipeline"),
        sinks=[memory_sink, logging_sink, jsonl_sink],
    )

    outcome = await runner.run("fan out events to three sinks")
    await runner.close()

    with open(jsonl_path, encoding="utf-8") as f:
        jsonl_lines = sum(1 for line in f if line.strip())

    print()
    print(f"success={outcome.success}, reason={outcome.reason!r}")
    print(f"InMemorySink captured {len(memory_sink.events)} events")
    print(f"JSONLPersistenceSink wrote {jsonl_lines} lines to {jsonl_path}")
    print("LoggingSink emitted one [LoggingSink] line per event on stderr above")

    os.unlink(jsonl_path)


if __name__ == "__main__":
    asyncio.run(main())

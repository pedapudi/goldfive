"""persistence_and_recovery — JSONL persistence + crash recovery flow.

Runs a small plan with a :class:`JSONLPersistenceSink`, simulates a
crash partway through by raising inside the adapter callable, then
invokes :meth:`Runner.resume` to rebuild a :class:`Session` from the
persisted log.

The JSONL sink encodes proto ``Event`` messages, so this example
requires the ``proto`` extra. Install with::

    uv pip install -e '.[proto,dev]'
    make proto  # generate the proto stubs

Then run::

    uv run python examples/persistence_and_recovery.py

Until the ``proto`` extra is installed the example short-circuits with
a friendly :class:`SystemExit` pointing at the install instructions.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from goldfive import (
    CallableAdapter,
    InvocationResult,
    JSONLPersistenceSink,
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

if JSONLPersistenceSink is None:  # type: ignore[truthy-function]
    raise SystemExit(
        "goldfive.sinks.JSONLPersistenceSink requires the `proto` extra; "
        "install goldfive[proto] and run `make proto` to generate the "
        "proto stubs, then retry."
    )


class _SimulatedCrash(RuntimeError):
    """Marker — only raised to simulate a crash in the demo."""


def build_plan() -> Plan:
    return Plan(
        id="persist-demo",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="First", assignee_agent_id="worker"),
            Task(id="t2", title="Second", assignee_agent_id="worker"),
            Task(id="t3", title="Third (crashes)", assignee_agent_id="worker"),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t2", to_task_id="t3"),
        ],
        summary="Three steps; the third simulates a crash.",
    )


async def crashy_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    _ = tools
    if task.id == "t3":
        raise _SimulatedCrash("boom")
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


async def main() -> None:
    tmp = tempfile.NamedTemporaryFile(
        prefix="goldfive-demo-", suffix=".jsonl", delete=False
    )
    tmp.close()
    path = tmp.name
    print(f"persistence file: {path}")

    # First run: crashes on t3.
    sink = JSONLPersistenceSink(path)
    runner = Runner(
        agent=CallableAdapter(crashy_agent, available_agents=["worker"]),
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(fail_fast=True),
        goal_deriver=PassthroughGoalDeriver("Persist and recover"),
        sinks=[sink],
    )
    outcome = await runner.run("go")
    await runner.close()
    print(f"first run: success={outcome.success}, reason={outcome.reason!r}")

    # Now recover from the persistence log. We construct a fresh Runner
    # solely to call resume(); the components passed in are placeholders
    # because resume() does not re-invoke the executor in v0.1.
    placeholder = Runner(
        agent=CallableAdapter(crashy_agent, available_agents=["worker"]),
        planner=StaticPlanner(build_plan()),
        executor=SequentialExecutor(),
    )
    recovered = await placeholder.resume(path)
    print(
        f"recovered: success={recovered.success}, reason={recovered.reason!r}, "
        f"run_id={recovered.session.run_id}, "
        f"completed={list(recovered.session.completed_results.keys())}"
    )

    os.unlink(path)


if __name__ == "__main__":
    asyncio.run(main())

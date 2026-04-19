"""parallel_dag — diamond DAG executed concurrently.

Builds a 5-task diamond plan (A -> {B, C, D} -> E) and runs it through
:class:`ParallelDAGExecutor`. Each task records its start and finish
timestamp so the output makes the parallelism of B, C, D visible.

Run with::

    uv run python examples/parallel_dag.py
"""

from __future__ import annotations

import asyncio
import random
import time

from goldfive import (
    CallableAdapter,
    InMemorySink,
    InvocationResult,
    ParallelDAGExecutor,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)


def build_diamond_plan() -> Plan:
    return Plan(
        id="diamond",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="A", title="Prepare", assignee_agent_id="worker"),
            Task(id="B", title="Branch B", assignee_agent_id="worker"),
            Task(id="C", title="Branch C", assignee_agent_id="worker"),
            Task(id="D", title="Branch D", assignee_agent_id="worker"),
            Task(id="E", title="Join", assignee_agent_id="worker"),
        ],
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="A", to_task_id="C"),
            TaskEdge(from_task_id="A", to_task_id="D"),
            TaskEdge(from_task_id="B", to_task_id="E"),
            TaskEdge(from_task_id="C", to_task_id="E"),
            TaskEdge(from_task_id="D", to_task_id="E"),
        ],
        summary="Diamond DAG: A then {B,C,D} in parallel then E.",
    )


async def main() -> None:
    random.seed(42)
    timeline: list[tuple[str, str, float]] = []
    t0 = time.monotonic()

    async def worker_agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = tools
        timeline.append((task.id, "start", time.monotonic() - t0))
        await asyncio.sleep(random.uniform(0.1, 0.3))
        timeline.append((task.id, "finish", time.monotonic() - t0))
        return InvocationResult(task_id=task.id, text=f"done: {task.title}")

    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(worker_agent, available_agents=["worker"]),
        planner=StaticPlanner(build_diamond_plan()),
        executor=ParallelDAGExecutor(),
        goal_deriver=PassthroughGoalDeriver("Run a diamond DAG"),
        sinks=[sink],
    )

    outcome = await runner.run("execute diamond")
    await runner.close()

    print(f"success={outcome.success}, reason={outcome.reason!r}")
    print(f"events captured: {len(sink.events)}")
    print()
    print("Per-task timeline (seconds since run start):")
    print(f"  {'task':<6} {'event':<8} {'t (s)':>8}")
    for task_id, kind, t in timeline:
        print(f"  {task_id:<6} {kind:<8} {t:>8.3f}")

    starts = {tid: t for tid, kind, t in timeline if kind == "start"}
    finishes = {tid: t for tid, kind, t in timeline if kind == "finish"}
    parallel_window_start = max(starts[k] for k in ("B", "C", "D"))
    parallel_window_end = min(finishes[k] for k in ("B", "C", "D"))
    print()
    print(
        f"B, C, D all started by t={parallel_window_start:.3f}s and were all "
        f"still running until t={parallel_window_end:.3f}s -- "
        f"overlap of {max(0.0, parallel_window_end - parallel_window_start):.3f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())

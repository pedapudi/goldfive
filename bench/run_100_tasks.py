"""100-task performance baseline for goldfive.

Runs a synthetic 100-task linear plan through ``SequentialExecutor``
plus ``JSONLPersistenceSink`` with a no-op ``CallableAdapter``. The
goal is to measure goldfive's orchestration overhead in isolation —
no LLM call, no network, no real agent work — so future regressions
in the runner / executor / sink path are visible against a pinned
baseline.

Run with::

    uv run python bench/run_100_tasks.py

The script prints a single block of measurements to stdout and exits
with status 0. The temporary JSONL file is unlinked before exit.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import goldfive
from goldfive import (
    CallableAdapter,
    Goal,
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
from goldfive.sinks import JSONLPersistenceSink

NUM_TASKS = 100


def build_linear_plan(n: int) -> Plan:
    """Build a linear DAG of ``n`` tasks: t000 -> t001 -> ... -> t{n-1}."""
    tasks = [
        Task(
            id=f"t{i:03d}",
            title=f"Task {i}",
            description=f"Trivial benchmark task #{i}",
            assignee_agent_id="bench-agent",
        )
        for i in range(n)
    ]
    edges = [
        TaskEdge(from_task_id=f"t{i:03d}", to_task_id=f"t{i + 1:03d}")
        for i in range(n - 1)
    ]
    return Plan(
        id="bench-100",
        run_id="",
        goal_ids=["bench-goal"],
        tasks=tasks,
        edges=edges,
        summary=f"Linear {n}-task benchmark plan",
    )


async def noop_agent(
    task: Task,
    session: Session,
    tools: list[ReportingToolSpec],
) -> InvocationResult:
    """Return immediately with a tiny result — no work, no I/O."""
    _ = session, tools
    return InvocationResult(task_id=task.id, text="ok")


async def run_benchmark(jsonl_path: Path) -> tuple[float, int, bool]:
    """Run one benchmark pass. Returns (wall_seconds, peak_bytes, success)."""
    plan = build_linear_plan(NUM_TASKS)
    sink = JSONLPersistenceSink(jsonl_path, mode="write")
    runner = Runner(
        agent=CallableAdapter(noop_agent, available_agents=["bench-agent"]),
        planner=StaticPlanner(plan),
        executor=SequentialExecutor(
            max_task_invocations=NUM_TASKS + 1,
            fail_fast=True,
        ),
        goal_deriver=PassthroughGoalDeriver("100-task benchmark"),
        sinks=[sink],
        max_task_invocations=NUM_TASKS + 1,
    )

    tracemalloc.start()
    start = time.perf_counter()
    outcome = await runner.run([Goal(id="bench-goal", summary="run 100 tasks")])
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    await runner.close()
    return elapsed, peak, outcome.success


def main() -> int:
    tmp = tempfile.NamedTemporaryFile(
        prefix="goldfive-bench-", suffix=".jsonl", delete=False
    )
    tmp.close()
    jsonl_path = Path(tmp.name)
    try:
        elapsed, peak_bytes, success = asyncio.run(run_benchmark(jsonl_path))
        if not success:
            print("BENCHMARK FAILED: run did not complete successfully", file=sys.stderr)
            return 1
        jsonl_size_bytes = jsonl_path.stat().st_size
    finally:
        try:
            os.unlink(jsonl_path)
        except OSError:
            pass

    throughput = NUM_TASKS / elapsed if elapsed > 0 else float("inf")
    peak_mib = peak_bytes / (1024 * 1024)
    jsonl_kib = jsonl_size_bytes / 1024
    py = sys.version_info

    print(
        f"goldfive perf baseline -- {NUM_TASKS} tasks, "
        f"SequentialExecutor + JSONLPersistenceSink"
    )
    print("-" * 64)
    print(f"Wall time:        {elapsed:.3f} s")
    print(f"Throughput:       {throughput:.1f} tasks/s")
    print(f"Peak memory:      {peak_mib:.2f} MiB (tracemalloc)")
    print(f"JSONL file size:  {jsonl_kib:.2f} KiB")
    print(f"Python:           {py.major}.{py.minor}.{py.micro}")
    print(f"goldfive:         {goldfive.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

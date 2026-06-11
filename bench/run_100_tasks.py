"""goldfive bench entrypoints: perf baseline + AGENCY-PRESERVATION three-arm.

Three subcommands (``perf`` is the default, so the original invocation is
unchanged):

``perf`` (default)
    The 100-task performance baseline. Runs a synthetic 100-task linear
    plan through ``SequentialExecutor`` + ``JSONLPersistenceSink`` with a
    no-op ``CallableAdapter`` — no LLM, no network — so runner / executor /
    sink regressions are visible against a pinned baseline. Prints one
    block of measurements and exits 0.

        uv run python bench/run_100_tasks.py
        uv run python bench/run_100_tasks.py perf

``three-arm``
    The AGENCY-PRESERVATION.md PR 13 counterfactual harness: runs the same
    workload under arm A (judge-only baseline), arm B (new SIGNAL regime)
    and arm C (legacy ladder), printing the per-arm metric table sourced
    from captured artifacts + sink events. With a stub model this exercises
    the harness + telemetry plumbing; 13b plugs in a live model.

        uv run python bench/run_100_tasks.py three-arm --out-dir /tmp/bench

``shadow``
    Runs the three arms in §5.4 shadow mode (behavior arms forced
    ``observation_only`` so signals are dry-run) and diffs the legacy vs.
    new logs into the divergence report — the exit-criterion artifact for
    enabling a behavior PR.

        uv run python bench/run_100_tasks.py shadow --out-dir /tmp/bench

See :mod:`bench.harness` and :mod:`bench.shadow_diff` for the library API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

# Ensure the repo root is importable so ``from bench.harness import ...`` works
# both when this file is run as a script (``python bench/run_100_tasks.py``)
# and when imported as ``bench.run_100_tasks``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def perf_main() -> int:
    """The original 100-task perf baseline (the default subcommand)."""
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


def _print_arm_metrics_table(results: list) -> None:
    print("goldfive three-arm bench (AGENCY-PRESERVATION.md PR 13)")
    print("=" * 72)
    for m in results:
        reason = f"({m.goal_reason})" if m.goal_reason else ""
        tokens = m.tokens if m.tokens is not None else "n/a"
        signals = f"{m.signals_total} / {m.signals_real} / {m.signals_dry_run}"
        print(f"\narm {m.arm_name}  [{m.arm_kind}]")
        print("-" * 72)
        print(f"  goal_success:            {m.goal_success}  {reason}")
        print(f"  completed_outputs:       {m.completed_outputs}")
        print(f"  turns / tokens:          {m.turns} / {tokens}")
        print(f"  run aborted:             {m.aborted}  {m.abort_reason}")
        print(f"  drift_detected:          {m.drift_detected}")
        print(f"  signals (total/real/dry):{signals}")
        print(f"  intervention_count:      {m.intervention_count}")
        print(f"  by_channel:              {m.by_channel}")
        print(f"  outcomes:                {m.outcomes}")
        print(f"  self_correction_base:    {m.self_correction_base_rate}")
        print(f"  post_signal_refire_rate: {m.post_signal_refire_rate}")
        print(f"  applied flags:           {m.applied_flags}")
        if m.pending_flags:
            print(f"  PENDING flags (no-op):   {m.pending_flags}  (not consulted by this build)")
        print(f"  jsonl artifact:          {m.jsonl_path}")


def three_arm_main(args: argparse.Namespace) -> int:
    """Run the three arms over the deterministic stub-model workload."""
    from bench.harness import Scenario, default_arms, make_linear_run_driver, run_arms

    out_dir = Path(args.out_dir)
    scenario = Scenario(
        name="linear-stub",
        driver=make_linear_run_driver(num_tasks=args.tasks, inject_signals=True),
    )
    results = asyncio.run(run_arms(default_arms(), scenario, jsonl_dir=out_dir))
    if args.json:
        print(json.dumps([m.to_dict() for m in results], indent=2, sort_keys=True, default=str))
    else:
        _print_arm_metrics_table(results)
    # A run is "ok" if every arm completed and emitted telemetry (loud signal
    # the plumbing is wired); the comparative judgement is 13b's job.
    if any(m.signals_total == 0 for m in results if m.arm_kind != "baseline"):
        print("\nWARNING: a behavior arm emitted 0 signals — telemetry may be off", file=sys.stderr)
        return 1
    return 0


def shadow_main(args: argparse.Namespace) -> int:
    """Run the arms in shadow mode and emit the legacy-vs-new divergence report."""
    from bench.harness import Scenario, make_linear_run_driver, run_arms, shadow_arms
    from bench.shadow_diff import diff_two_logs, load_signals, render_two_log_text

    out_dir = Path(args.out_dir)
    arms = shadow_arms()
    scenario = Scenario(
        name="linear-stub-shadow",
        driver=make_linear_run_driver(num_tasks=args.tasks, inject_signals=True),
    )
    results = asyncio.run(run_arms(arms, scenario, jsonl_dir=out_dir))
    by_kind = {m.arm_kind: m for m in results}
    legacy_log = by_kind["legacy"].jsonl_path
    new_log = by_kind["signal"].jsonl_path

    legacy = load_signals(legacy_log)
    new = load_signals(new_log)
    report = diff_two_logs(legacy, new, legacy_path=legacy_log, new_path=new_log)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(render_two_log_text(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_100_tasks", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("perf", help="100-task perf baseline (default)")
    ta = sub.add_parser("three-arm", help="three-arm counterfactual bench")
    ta.add_argument("--out-dir", default="/tmp/g5-bench-artifacts", help="JSONL artifact dir")
    ta.add_argument("--tasks", type=int, default=3, help="tasks per arm (stub workload)")
    ta.add_argument("--json", action="store_true")
    sh = sub.add_parser("shadow", help="shadow-mode run + legacy-vs-new divergence report")
    sh.add_argument("--out-dir", default="/tmp/g5-bench-artifacts", help="JSONL artifact dir")
    sh.add_argument("--tasks", type=int, default=3, help="tasks per arm (stub workload)")
    sh.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd in (None, "perf"):
        return perf_main()
    if args.cmd == "three-arm":
        return three_arm_main(args)
    if args.cmd == "shadow":
        return shadow_main(args)
    parser.error(f"unknown command {args.cmd!r}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())

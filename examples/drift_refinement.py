"""drift_refinement — drift mid-plan triggers a planner refinement.

The middle task in a 3-task plan reports a recoverable failure via the
``report_task_failed`` reporting tool. That fires a WARNING drift through
:class:`DefaultSteerer`, which calls ``planner.refine`` and swaps in a
new plan that replaces the failed task with a fallback. Execution then
continues on the revised plan.

Run with::

    uv run python examples/drift_refinement.py
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from goldfive import (
    CallableAdapter,
    DefaultSteerer,
    DriftEvent,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    Task,
    TaskEdge,
)
from goldfive.types import Goal, TaskStatus


def build_initial_plan() -> Plan:
    return Plan(
        id="plan-initial",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="research", title="Gather sources", assignee_agent_id="worker"),
            Task(
                id="primary_api",
                title="Call primary data API (will fail)",
                assignee_agent_id="worker",
            ),
            Task(id="report", title="Write the report", assignee_agent_id="worker"),
        ],
        edges=[
            TaskEdge(from_task_id="research", to_task_id="primary_api"),
            TaskEdge(from_task_id="primary_api", to_task_id="report"),
        ],
        summary="Research, fetch data, write a report.",
    )


class _RefiningPlanner:
    """Returns ``build_initial_plan`` from generate; on refine swaps the
    failed task for a fallback that still satisfies the goal."""

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Mapping[str, Any] | None = None,
    ) -> Plan | None:
        plan = build_initial_plan()
        if context is not None:
            plan.run_id = str(context.get("run_id") or "")
        plan.goal_ids = [g.id for g in goals if g.id] or plan.goal_ids
        return plan

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append(drift)
        preserved = [
            Task(
                id=t.id,
                title=t.title,
                description=t.description,
                assignee_agent_id=t.assignee_agent_id,
                status=t.status,
            )
            for t in plan.tasks
            if t.status != TaskStatus.PENDING or t.id != drift.current_task_id
        ]
        fallback = Task(
            id="fallback_api",
            title="Call backup data API",
            assignee_agent_id="worker",
        )
        report = next(
            (t for t in preserved if t.id == "report"),
            None,
        )
        if report is None:
            preserved.append(
                Task(id="report", title="Write the report", assignee_agent_id="worker")
            )
        new_tasks = preserved + [fallback]
        new_edges = [
            TaskEdge(from_task_id="research", to_task_id="fallback_api"),
            TaskEdge(from_task_id="fallback_api", to_task_id="report"),
        ]
        return Plan(
            id=f"plan-refined-{uuid.uuid4().hex[:8]}",
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=new_tasks,
            edges=new_edges,
            summary="Refined: primary API failed, routed via backup.",
            revision_index=plan.revision_index + 1,
            revision_reason=drift.detail,
            revision_kind=str(drift.kind),
            revision_severity=str(drift.severity),
        )


def _print_plan(label: str, plan: Plan) -> None:
    print(f"  {label} (id={plan.id}, revision={plan.revision_index}):")
    for t in plan.tasks:
        print(f"    - {t.id:<14} status={t.status.value:<10} title={t.title!r}")
    for e in plan.edges:
        print(f"    edge {e.from_task_id} -> {e.to_task_id}")


async def main() -> None:
    steerer = DefaultSteerer()
    sink = InMemorySink()
    planner = _RefiningPlanner()

    initial_plan = build_initial_plan()

    async def worker_agent(
        task: Task,
        session: Session,
        tools: list[ReportingToolSpec],
    ) -> InvocationResult:
        _ = tools
        if task.id == "primary_api":
            await steerer.mark_task_failed(
                task.id,
                session=session,
                reason="primary API returned 503 (recoverable)",
                recoverable=True,
            )
            return InvocationResult(task_id=task.id, text="primary API unavailable")
        return InvocationResult(task_id=task.id, text=f"finished: {task.title}")

    runner = Runner(
        agent=CallableAdapter(worker_agent, available_agents=["worker"]),
        planner=planner,
        executor=SequentialExecutor(fail_fast=False, max_plan_reinvocations=6),
        goal_deriver=PassthroughGoalDeriver("Produce a report from API data"),
        steerer=steerer,
        sinks=[sink],
    )

    print("Initial plan:")
    _print_plan("initial", initial_plan)
    print()

    outcome = await runner.run("produce report")
    await runner.close()

    print(f"Run finished: success={outcome.success}, reason={outcome.reason!r}")
    print(f"planner.refine() was called {len(planner.refine_calls)} time(s)")
    if planner.refine_calls:
        d = planner.refine_calls[0]
        print(f"  drift kind={d.kind.value} severity={d.severity.value}")
        print(f"  drift detail={d.detail!r}")
    print()
    print("Final plan after refinement:")
    if outcome.session.plan is not None:
        _print_plan("revised", outcome.session.plan)
    print()
    print(f"Total events captured: {len(sink.events)}")


if __name__ == "__main__":
    asyncio.run(main())

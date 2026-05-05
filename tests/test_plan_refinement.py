"""Mid-run plan refinement semantics.

When ``planner.refine`` returns a revised plan while an executor is in
flight, the spec requires:

  * completed tasks are preserved (status COMPLETED, result reachable).
  * ``plan.revision_index`` is strictly monotonic.
  * a ``PlanRevised`` event is emitted to every bound sink.
"""

from __future__ import annotations

from typing import Any

import pytest

types = pytest.importorskip("goldfive.types")


def _mk_task(task_id: str, title: str = "") -> Any:
    return types.Task(id=task_id, title=title or task_id, assignee_agent_id="default")


def _mk_plan(
    *,
    run_id: str,
    tasks: list[Any],
    edges: list[Any] | None = None,
    revision_index: int = 0,
    revision_reason: str = "",
    revision_kind: str = "",
    revision_severity: str = "",
) -> Any:
    return types.Plan(
        id=f"plan-{revision_index}",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=tasks,
        edges=edges or [],
        revision_index=revision_index,
        revision_reason=revision_reason,
        revision_kind=revision_kind,
        revision_severity=revision_severity,
    )


def test_revision_index_is_strictly_monotonic() -> None:
    run_id = "refine-mono"
    p0 = _mk_plan(run_id=run_id, tasks=[_mk_task("t1")], revision_index=0)
    p1 = _mk_plan(run_id=run_id, tasks=[_mk_task("t1"), _mk_task("t2")], revision_index=1)
    p2 = _mk_plan(run_id=run_id, tasks=[_mk_task("t1"), _mk_task("t3")], revision_index=2)
    revisions = [p0.revision_index, p1.revision_index, p2.revision_index]
    assert revisions == sorted(revisions)
    assert len(set(revisions)) == len(revisions)


def test_refined_plan_preserves_completed_task_ids() -> None:
    """Regardless of executor, the contract is: a refined plan should
    include the already-completed task IDs so downstream executors and
    sinks don't double-dispatch them."""

    run_id = "refine-preserve"
    original = _mk_plan(run_id=run_id, tasks=[_mk_task("t1"), _mk_task("t2")])
    # goldfive#247: Plan is frozen — derive via with_task_status.
    original = types.with_task_status(original, "t1", types.TaskStatus.COMPLETED)

    refined = _mk_plan(
        run_id=run_id,
        tasks=[original.tasks[0], _mk_task("t3")],
        revision_index=1,
        revision_reason="drift: new_work_discovered",
    )

    refined_ids = {t.id for t in refined.tasks}
    assert "t1" in refined_ids
    completed = [t for t in refined.tasks if t.status == types.TaskStatus.COMPLETED]
    assert [t.id for t in completed] == ["t1"]


async def test_planner_refine_increments_revision_and_preserves_completed() -> None:
    """Contract check on any planner that implements ``refine``: the
    returned plan must bump ``revision_index`` by exactly one and must
    retain already-completed task IDs."""

    class _RefiningPlanner:
        def __init__(self) -> None:
            self.refines = 0

        async def generate(self, *, goals, available_agents, context=None):
            return _mk_plan(run_id="refine-int", tasks=[_mk_task("t1"), _mk_task("t2")])

        async def refine(self, *, plan, drift, goals):
            self.refines += 1
            return _mk_plan(
                run_id=plan.run_id,
                tasks=[plan.tasks[0], _mk_task("t3")],
                revision_index=plan.revision_index + 1,
                revision_reason=drift.detail or "refine",
            )

    planner = _RefiningPlanner()
    session = types.Session(run_id="refine-int")
    p0 = await planner.generate(
        goals=[types.Goal(id="g1", summary="do things")],
        available_agents=["default"],
    )
    # goldfive#247: install via the test helper so the channel-processor
    # gate is satisfied; flip t1 COMPLETED via with_task_status.
    p0 = types.with_task_status(p0, "t1", types.TaskStatus.COMPLETED)
    from tests._immutable_plan_helpers import force_plan

    force_plan(session, p0)

    drift = types.DriftEvent(
        kind=types.DriftKind.NEW_WORK_DISCOVERED,
        severity=types.DriftSeverity.INFO,
        detail="discovered t3",
        current_task_id="t1",
    )
    p1 = await planner.refine(plan=p0, drift=drift, goals=session.goals)
    assert p1 is not None
    assert p1.revision_index == p0.revision_index + 1
    assert {t.id for t in p1.tasks} >= {"t1", "t3"}
    preserved = next(t for t in p1.tasks if t.id == "t1")
    assert preserved.status == types.TaskStatus.COMPLETED
    assert planner.refines == 1


async def test_plan_revised_event_is_emitted_when_runner_refines() -> None:
    """If the Runner/Executor is implemented, a plan revision must emit
    a ``PlanRevised`` event. Skipped until those modules land."""

    events_mod = pytest.importorskip("goldfive.events")
    sinks_mod = pytest.importorskip("goldfive.sinks")
    InMemorySink = getattr(sinks_mod, "InMemorySink", None)
    emit_plan_revised = getattr(events_mod, "emit_plan_revised", None)
    if InMemorySink is None or emit_plan_revised is None:
        pytest.skip("emit_plan_revised helper or InMemorySink not yet implemented")

    sink = InMemorySink()
    session = types.Session(run_id="evt")
    plan = _mk_plan(run_id="evt", tasks=[_mk_task("t1")], revision_index=1)
    await emit_plan_revised([sink], session=session, plan=plan)
    # A single event should have been captured.
    recorded = getattr(sink, "events", [])
    assert len(recorded) == 1


async def test_refine_returning_none_is_a_noop() -> None:
    """A planner that returns None from ``refine`` signals "keep the
    current plan". Executors must not raise."""

    class _NoopPlanner:
        async def generate(self, *, goals, available_agents, context=None):
            return _mk_plan(run_id="noop", tasks=[_mk_task("t1")])

        async def refine(self, *, plan, drift, goals):
            return None

    planner = _NoopPlanner()
    plan = await planner.generate(goals=[], available_agents=["default"])
    drift = types.DriftEvent(kind=types.DriftKind.CUSTOM, severity=types.DriftSeverity.INFO)
    assert await planner.refine(plan=plan, drift=drift, goals=[]) is None


async def test_revision_reason_and_kind_propagate() -> None:
    """A refined plan carries drift kind/severity/reason metadata so
    UIs and auditors can explain "why did the plan change?"."""

    class _TaggingPlanner:
        async def generate(self, *, goals, available_agents, context=None):
            return _mk_plan(run_id="tag", tasks=[_mk_task("t1")])

        async def refine(self, *, plan, drift, goals):
            # goldfive#247: Plan is frozen — pass all metadata at
            # construction.
            return _mk_plan(
                run_id=plan.run_id,
                tasks=[*plan.tasks, _mk_task("t2")],
                revision_index=plan.revision_index + 1,
                revision_reason=drift.detail,
                revision_kind=drift.kind.value,
                revision_severity=drift.severity.value,
            )

    planner = _TaggingPlanner()
    plan = await planner.generate(goals=[], available_agents=["default"])
    drift = types.DriftEvent(
        kind=types.DriftKind.AGENT_REFUSAL,
        severity=types.DriftSeverity.WARNING,
        detail="agent refused tool call",
    )
    refined = await planner.refine(plan=plan, drift=drift, goals=[])
    assert refined is not None
    assert refined.revision_kind == types.DriftKind.AGENT_REFUSAL.value
    assert refined.revision_severity == types.DriftSeverity.WARNING.value
    assert "refused" in refined.revision_reason


def test_revision_kind_severity_default_empty_strings() -> None:
    """A fresh plan has no revision metadata — only refined plans do."""
    p = _mk_plan(run_id="fresh", tasks=[_mk_task("t1")])
    assert p.revision_index == 0
    assert p.revision_reason == ""
    assert p.revision_kind == ""
    assert p.revision_severity == ""

"""Unit tests for :class:`goldfive.executors.SequentialExecutor`.

The tests use lightweight stubs for :class:`AgentAdapter`, :class:`Steerer`,
and :class:`Planner` that are just rich enough to exercise the three required
scenarios:

1. A 3-task linear plan runs to completion.
2. Drift mid-run triggers ``planner.refine``, the executor applies the revised
   plan and keeps going.
3. ``fail_fast=True`` stops on the first task failure;
   ``fail_fast=False`` walks past failures and reports success=False at the end.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from goldfive.executors import SequentialExecutor, build_task_nudge
from goldfive.protocols import EventSink
from goldfive.results import InvocationResult
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class RecordingSink:
    """EventSink that accumulates everything emitted for later assertion."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True

    def payload_kinds(self) -> list[str]:
        kinds: list[str] = []
        for e in self.events:
            if hasattr(e, "WhichOneof"):
                kinds.append(e.WhichOneof("payload") or "")
            elif isinstance(e, dict):
                kinds.append(e.get("kind", ""))
            else:
                kinds.append(getattr(e, "payload_kind", ""))
        return kinds


class StubSteerer:
    """Minimal Steerer stub that just forwards reporting-tool calls into
    plan.Task.status mutations. Acts as the central authority over task state
    the way a real Steerer would.
    """

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:
        self.observed.append(event)

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        if session.plan is None:
            return
        for t in session.plan.tasks:
            if t.id == task_id:
                t.status = to
                return

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:
        return None


class StubPlanner:
    """Planner that records ``refine`` calls and returns whatever the caller
    supplied via ``set_refine_result``.
    """

    def __init__(self) -> None:
        self.refine_calls: list[tuple[Plan, DriftEvent]] = []
        self._refine_result: Plan | None = None

    def set_refine_result(self, plan: Plan | None) -> None:
        self._refine_result = plan

    async def generate(
        self,
        *,
        goals: list,
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list,
    ) -> Plan | None:
        self.refine_calls.append((plan, drift))
        return self._refine_result


class StubAdapter:
    """Adapter whose ``invoke`` calls a per-task callback (provided by the
    test). The callback receives ``(task, session, steerer, planner)`` and
    typically mutates task status directly — emulating the reporting-tool
    handler path.
    """

    def __init__(
        self,
        *,
        steerer: StubSteerer,
        planner: StubPlanner,
        on_invoke: Callable[
            [Task, Session, StubSteerer, StubPlanner],
            Awaitable[InvocationResult],
        ],
    ) -> None:
        self._steerer = steerer
        self._planner = planner
        self._on_invoke = on_invoke
        self.invocations: list[str] = []

    async def register_reporting_tools(self, tools: list[Any]) -> None:
        return None

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        self.invocations.append(task.id)
        return await self._on_invoke(task, session, self._steerer, self._planner)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_plan(n: int = 3, run_id: str = "run-1") -> Plan:
    """Return a linear plan t0 -> t1 -> ... -> t(n-1)."""
    tasks = [Task(id=f"t{i}", title=f"Task {i}") for i in range(n)]
    edges = [TaskEdge(from_task_id=f"t{i}", to_task_id=f"t{i+1}") for i in range(n - 1)]
    return Plan(id="p0", run_id=run_id, goal_ids=[], tasks=tasks, edges=edges)


def _fresh_session(run_id: str = "run-1") -> Session:
    return Session(run_id=run_id)


# ---------------------------------------------------------------------------
# Scenario 1: linear plan runs to completion.
# ---------------------------------------------------------------------------


async def test_linear_plan_runs_to_completion() -> None:
    plan = _linear_plan(3)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Simulate the reporting-tool handler path: transition the task to
        # COMPLETED through the steerer.
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        session.completed_results[task.id] = f"done:{task.id}"
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)

    # Default max_plan_reinvocations=3; allow exactly 3 tasks.
    executor = SequentialExecutor(max_plan_reinvocations=3)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True
    assert outcome.session is session
    assert adapter.invocations == ["t0", "t1", "t2"]
    for t in plan.tasks:
        assert t.status == TaskStatus.COMPLETED

    kinds = sink.payload_kinds()
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"


# ---------------------------------------------------------------------------
# Scenario 2: drift mid-run triggers planner.refine; executor applies the new
# plan and keeps going.
# ---------------------------------------------------------------------------


async def test_drift_mid_run_applies_refined_plan() -> None:
    plan = _linear_plan(2)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    # The refined plan replaces t1 with t1b (still depending on t0). When
    # the first invocation "drifts", it will swap session.plan to this one.
    refined_tasks = [
        Task(id="t0", title="Task 0", status=TaskStatus.COMPLETED),
        Task(id="t1b", title="Task 1 revised"),
    ]
    refined_edges = [TaskEdge(from_task_id="t0", to_task_id="t1b")]
    refined = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=[],
        tasks=refined_tasks,
        edges=refined_edges,
        revision_reason="new work discovered",
        revision_kind=DriftKind.NEW_WORK_DISCOVERED.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )
    planner.set_refine_result(refined)

    # Track how many times invoke has been called to script the drift on
    # the very first invocation.
    call_count = {"n": 0}

    async def _maybe_drift(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        call_count["n"] += 1
        # First call: complete t0, then trigger a drift-driven refine and
        # swap the plan onto the session (emulating what a real Steerer
        # would do on observing `report_plan_divergence`).
        if call_count["n"] == 1:
            await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
            drift = DriftEvent(
                kind=DriftKind.NEW_WORK_DISCOVERED,
                severity=DriftSeverity.WARNING,
                detail="found extra work",
                current_task_id=task.id,
            )
            revised = await planner.refine(
                plan=session.plan, drift=drift, goals=session.goals
            )
            if revised is not None:
                session.plan = revised
            return InvocationResult(task_id=task.id, text="done-with-drift")
        # Subsequent calls: just complete the task.
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_maybe_drift)

    executor = SequentialExecutor(max_plan_reinvocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True
    # Invocations: t0 (original plan) then t1b (from the refined plan).
    assert adapter.invocations == ["t0", "t1b"]
    # planner.refine was called exactly once.
    assert len(planner.refine_calls) == 1
    # A PlanRevised event landed in the sink between run_started and run_completed.
    kinds = sink.payload_kinds()
    assert "plan_revised" in kinds
    assert kinds.index("plan_revised") < kinds.index("run_completed")


# ---------------------------------------------------------------------------
# Scenario 3: fail_fast behavior.
# ---------------------------------------------------------------------------


async def test_fail_fast_stops_on_task_failure() -> None:
    plan = _linear_plan(3)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _fail_middle(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        if task.id == "t1":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_fail_middle)

    executor = SequentialExecutor(max_plan_reinvocations=5, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is False
    # Walked exactly t0 then t1; did not continue to t2.
    assert adapter.invocations == ["t0", "t1"]
    # Terminal event is RunAborted.
    assert sink.payload_kinds()[-1] == "run_aborted"
    assert plan.tasks[2].status == TaskStatus.PENDING


async def test_fail_fast_false_continues_past_failure() -> None:
    # With an independent plan (two roots) a non-fatal failure on one branch
    # should not block the other branch when fail_fast=False.
    tasks = [
        Task(id="a", title="A"),
        Task(id="b", title="B"),
        Task(id="c", title="C"),
    ]
    # No edges: all three are independent roots, so walker can still
    # progress past a failed one.
    plan = Plan(id="p0", run_id="run-1", goal_ids=[], tasks=tasks, edges=[])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _fail_b(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        if task.id == "b":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_fail_b)

    executor = SequentialExecutor(max_plan_reinvocations=5, fail_fast=False)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # All three tasks were attempted.
    assert set(adapter.invocations) == {"a", "b", "c"}
    # The run is not "successful" because one task failed, but we did walk
    # past the failure (that's the distinguishing behavior from fail_fast).
    assert outcome.success is False
    # Terminal event is RunAborted with a fail_fast=False reason.
    assert sink.payload_kinds()[-1] == "run_aborted"
    # Remaining tasks a and c are COMPLETED, b stayed FAILED.
    by_id = {t.id: t.status for t in plan.tasks}
    assert by_id == {
        "a": TaskStatus.COMPLETED,
        "b": TaskStatus.FAILED,
        "c": TaskStatus.COMPLETED,
    }


# ---------------------------------------------------------------------------
# Scenario 4 (bonus): the re-invocation budget terminates a stuck run.
# ---------------------------------------------------------------------------


async def test_budget_terminates_stuck_run() -> None:
    plan = _linear_plan(5)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _noop(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Deliberately do NOT transition the task. The executor should treat
        # post-invoke PENDING as a local failure so the walker can move on,
        # but it also will not progress past the failed task because the
        # next edge depends on it.
        return InvocationResult(task_id=task.id, text="")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_noop)

    executor = SequentialExecutor(max_plan_reinvocations=2, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # fail_fast=True + first task ends FAILED (because it was left PENDING
    # post-invoke and the executor marked it FAILED locally) -> aborts
    # after the first invocation.
    assert outcome.success is False
    assert sink.payload_kinds()[-1] == "run_aborted"
    assert adapter.invocations == ["t0"]


# ---------------------------------------------------------------------------
# build_task_nudge: canonical next-task nudge string.
# ---------------------------------------------------------------------------


def test_build_task_nudge_includes_id_title_and_description() -> None:
    t = Task(id="t42", title="Do the thing", description="Carefully and thoroughly.")
    nudge = build_task_nudge(t)
    assert "t42" in nudge
    assert "Do the thing" in nudge
    assert "Carefully and thoroughly." in nudge
    assert nudge.startswith("Continue executing the plan.")

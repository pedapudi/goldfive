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
    Goal,
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
        cancel_reason: str = "",
    ) -> None:
        if session.plan is None:
            return
        # goldfive#247: derive new plan + swap (frozen Plan).
        if not any(t.id == task_id for t in session.plan.tasks):
            return
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )
        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))

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
    edges = [TaskEdge(from_task_id=f"t{i}", to_task_id=f"t{i + 1}") for i in range(n - 1)]
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

    # Default max_task_invocations=3; allow exactly 3 tasks.
    executor = SequentialExecutor(max_task_invocations=3)
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
    # goldfive#247: read from session.plan (live).
    assert session.plan is not None
    for t in session.plan.tasks:
        assert t.status == TaskStatus.COMPLETED

    # The executor itself owns only the terminal RunCompleted (Runner
    # owns RunStarted / GoalDerived / PlanSubmitted). The StubSteerer
    # emits nothing, so the only event the sink sees is run_completed.
    kinds = sink.payload_kinds()
    assert kinds[-1] == "run_completed"
    assert "run_started" not in kinds


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
            revised = await planner.refine(plan=session.plan, drift=drift, goals=session.goals)
            if revised is not None:
                session.plan = revised
            return InvocationResult(task_id=task.id, text="done-with-drift")
        # Subsequent calls: just complete the task.
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_maybe_drift)

    executor = SequentialExecutor(max_task_invocations=5)
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


async def test_out_of_band_plan_revised_carries_trigger_event_id() -> None:
    """Out-of-band PlanRevised preserves revision_trigger_event_id (goldfive#199).

    When the steerer swaps ``session.plan`` mid-run, the sequential
    executor emits its own PlanRevised envelope to mark the boundary.
    That envelope must carry the ``trigger_event_id`` (read off
    ``plan.revision_trigger_event_id``) so harmonograf's intervention
    aggregator can strict-id-join it to the source annotation / drift —
    otherwise a slow refine strands the plan row and leaks a duplicate
    card (harmonograf#95 rescope).
    """
    plan = _linear_plan(2)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    # Simulate the post-steerer state: a revised plan stamped with the
    # trigger_event_id (what DefaultSteerer._apply_revision would produce
    # on a USER_STEER from an annotation-backed ControlMessage).
    refined = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Task 0", status=TaskStatus.COMPLETED),
            Task(id="t1b", title="Task 1 revised"),
        ],
        edges=[TaskEdge(from_task_id="t0", to_task_id="t1b")],
        revision_reason="by alice: refocus",
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
        revision_trigger_event_id="ann_seq_mid_run",
    )
    planner.set_refine_result(refined)

    call_count = {"n": 0}

    async def _swap_plan(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
            # Mimic the steerer's USER_STEER handoff: swap session.plan
            # without emitting the steerer-side PlanRevised, so the
            # executor's out-of-band detector is what fires.
            session.plan = refined
            return InvocationResult(task_id=task.id, text="steered")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_swap_plan)
    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True
    revised_events = [
        e
        for e in sink.events
        if getattr(e, "WhichOneof", lambda *_: None)("payload") == "plan_revised"
    ]
    assert revised_events, "executor should emit PlanRevised on out-of-band plan swap"
    evt = revised_events[0]
    assert evt.plan_revised.trigger_event_id == "ann_seq_mid_run"
    assert evt.plan_revised.plan.revision_trigger_event_id == "ann_seq_mid_run"


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

    executor = SequentialExecutor(max_task_invocations=5, fail_fast=True)
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

    executor = SequentialExecutor(max_task_invocations=5, fail_fast=False)
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
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
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

    async def _stuck(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Return an invocation-level error without transitioning: the
        # executor auto-fails the task on its behalf (matching the
        # "InvocationResult.error is set" branch of the auto-transition
        # logic) so fail_fast aborts after the first invocation.
        return InvocationResult(task_id=task.id, text="", error=RuntimeError("stuck"))

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_stuck)

    executor = SequentialExecutor(max_task_invocations=2, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # fail_fast=True + first task ends FAILED (via the executor's auto
    # transition on the non-None InvocationResult.error) -> aborts after
    # the first invocation.
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


# ---------------------------------------------------------------------------
# Per-task retry-lineage cap (TASK-LIFECYCLE.md §7.7).
#
# The executor bounds how many times any one "task lineage" may be sent to
# the adapter inside a single run. A lineage is identified by stripping
# chained ``retry_`` / ``retry<N>_`` prefixes from the task id, so
# ``t0``, ``retry_t0``, ``retry2_retry_t0`` all share the same root ``t0``
# and share one invocation budget. This caps blast radius when a
# misbehaving planner keeps regenerating ``retry_<task>`` clones.
# ---------------------------------------------------------------------------


def test_retry_lineage_cap_default_is_3() -> None:
    """Sanity: the default lineage cap is 3 (see TASK-LIFECYCLE.md §7.7)."""
    executor = SequentialExecutor()
    assert executor.max_retries_per_task_lineage == 3


async def test_retry_lineage_cap_fails_task_after_N_retries() -> None:
    """With cap=2, the 3rd invocation on the same lineage is skipped.

    We simulate a refine loop that keeps producing ``retry_<prev>``
    versions of a failing task. After 2 invocations on the lineage the
    executor should refuse to invoke the 3rd clone and transition it
    to FAILED in place. We run with ``fail_fast=False`` to keep the
    run going long enough to observe the 3rd clone being refused
    (fail_fast=True would abort after the first task failure).
    """
    # Start with a single task t0. After each failed invocation, the
    # adapter simulates a refine by swapping session.plan to a new plan
    # whose only pending task is a ``retry_<prev>`` clone.
    initial_plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[Task(id="t0", title="Task 0")],
        edges=[],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _fail_and_spawn_retry(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Fail the current task, then swap in a new plan whose only
        # pending task is a retry clone. This mirrors what a misbehaving
        # planner.refine would do.
        await steerer.transition(task.id, TaskStatus.FAILED, session=session)
        new_id = f"retry_{task.id}"
        # Keep prior tasks (all FAILED) so the plan history stays
        # consistent; append the fresh PENDING retry clone.
        new_tasks = [
            Task(id=t.id, title=t.title, status=TaskStatus.FAILED)
            for t in (session.plan.tasks if session.plan else [])
        ]
        new_tasks.append(Task(id=new_id, title=f"retry of {task.title}"))
        session.plan = Plan(
            id="p0",
            run_id=session.run_id,
            goal_ids=[],
            tasks=new_tasks,
            edges=[],
            revision_index=(session.plan.revision_index if session.plan else 0) + 1,
        )
        return InvocationResult(task_id=task.id, text="", stop_reason="failed")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_fail_and_spawn_retry)

    # cap=2 -> allow 2 invocations on lineage root "t0"; the 3rd clone
    # is refused before the adapter is called.
    executor = SequentialExecutor(
        max_task_invocations=32,
        max_retries_per_task_lineage=2,
        fail_fast=False,
    )
    outcome = await executor.run(
        plan=initial_plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # Adapter saw t0 and retry_t0 only; retry_retry_t0 was refused.
    assert adapter.invocations == ["t0", "retry_t0"]
    assert outcome.success is False
    # The skipped lineage task must be marked FAILED in the session's
    # current plan without ever hitting the adapter.
    final = session.plan
    assert final is not None
    by_id = {t.id: t.status for t in final.tasks}
    assert by_id.get("retry_retry_t0") == TaskStatus.FAILED


async def test_retry_lineage_cap_is_per_task_not_global() -> None:
    """Two independent lineages each get their own budget.

    Plan ``a`` and ``b`` are independent roots. With cap=2 each lineage
    may burn 2 invocations; a cross-lineage invocation must NOT count
    against a sibling lineage's budget.
    """
    tasks = [
        Task(id="a", title="A"),
        Task(id="b", title="B"),
    ]
    plan = Plan(id="p0", run_id="run-1", goal_ids=[], tasks=tasks, edges=[])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    # Per-lineage state held outside the closure: each lineage fails
    # exactly once, then the retry succeeds. Both lineages therefore
    # spend exactly 2 invocations each and both complete.
    fail_once: dict[str, bool] = {"a": False, "b": False}

    async def _fail_first_then_succeed(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Derive the lineage root by stripping a single "retry_" prefix;
        # good enough for this 2-deep test.
        root = task.id[len("retry_") :] if task.id.startswith("retry_") else task.id
        if not fail_once[root]:
            fail_once[root] = True
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            # Add a retry clone so the plan still has work to do.
            new_tasks = list(session.plan.tasks) if session.plan else []
            new_tasks.append(Task(id=f"retry_{task.id}", title=f"retry of {task.title}"))
            session.plan = Plan(
                id="p0",
                run_id=session.run_id,
                goal_ids=[],
                tasks=new_tasks,
                edges=list(session.plan.edges) if session.plan else [],
                revision_index=(session.plan.revision_index if session.plan else 0) + 1,
            )
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_fail_first_then_succeed)

    executor = SequentialExecutor(
        max_task_invocations=32,
        max_retries_per_task_lineage=2,
        fail_fast=False,
    )
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # Each lineage got its own budget: we saw exactly four invocations
    # (original + one retry for each of the two lineages). Neither
    # lineage was throttled by the other's spending. The original
    # ``a`` / ``b`` tasks stay FAILED (that's how the retry was
    # spawned) so ``outcome.success`` is False, but the point of the
    # test is the invocation count and retry completion below — we
    # specifically did NOT let one lineage exhaust the cap because the
    # other's invocations don't count against it.
    assert adapter.invocations == ["a", "b", "retry_a", "retry_b"]
    # Both retry clones were invoked and completed — no lineage hit
    # the cap.
    final = session.plan
    assert final is not None
    by_id = {t.id: t.status for t in final.tasks}
    assert by_id["retry_a"] == TaskStatus.COMPLETED
    assert by_id["retry_b"] == TaskStatus.COMPLETED
    # Nothing was lineage-capped: had the cap been global at 2,
    # ``retry_b`` would have been refused (4 > 2) and ended FAILED.
    _ = outcome  # explicitly acknowledge we don't gate on success


async def test_retry_lineage_cap_passes_on_successful_retry() -> None:
    """If a task fails once and its retry succeeds, the cap must not impede.

    With cap=3, a lineage that consumes only 2 invocations (original +
    one successful retry) should run cleanly to completion — the cap
    only bites when a lineage exhausts its budget.
    """
    initial_plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[Task(id="t0", title="Task 0")],
        edges=[],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    attempts = {"n": 0}

    async def _fail_then_retry_succeeds(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        attempts["n"] += 1
        if task.id == "t0" and attempts["n"] == 1:
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            # Spawn a retry clone.
            session.plan = Plan(
                id="p0",
                run_id=session.run_id,
                goal_ids=[],
                tasks=[
                    Task(id="t0", title="Task 0", status=TaskStatus.FAILED),
                    Task(id="retry_t0", title="retry of Task 0"),
                ],
                edges=[],
                revision_index=1,
            )
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        # retry_t0 (second invocation in the lineage) succeeds.
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_fail_then_retry_succeeds)

    executor = SequentialExecutor(
        max_task_invocations=32,
        max_retries_per_task_lineage=3,
        fail_fast=False,
    )
    outcome = await executor.run(
        plan=initial_plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # Exactly two invocations: the original failing call plus the
    # successful retry. The cap (3) is never exhausted, so the retry
    # was allowed through and completed.
    assert adapter.invocations == ["t0", "retry_t0"]
    final = session.plan
    assert final is not None
    by_id = {t.id: t.status for t in final.tasks}
    assert by_id["retry_t0"] == TaskStatus.COMPLETED
    # The original ``t0`` is FAILED (that was how the retry got
    # spawned) so ``outcome.success`` is False under fail_fast=False;
    # the behavior under test is that the retry survived the cap, not
    # that the run summary is a clean pass.
    _ = outcome


async def test_retry_lineage_cap_collapses_nested_retry_prefixes() -> None:
    """``retry_retry_t0`` and ``retry2_t0`` share lineage root ``t0``.

    Explicit coverage for the prefix-stripping convention so a future
    change to the regex doesn't silently let nested retry ids escape
    the lineage cap.
    """
    # Construct a plan where all three tasks share the same lineage
    # root ``t0``. With cap=2 exactly two of them may be invoked; the
    # third must be skipped.
    tasks = [
        Task(id="t0", title="Task 0"),
        Task(id="retry_t0", title="retry of Task 0"),
        Task(id="retry_retry_t0", title="retry of retry of Task 0"),
    ]
    # Chain them so the executor sees them in order: t0 -> retry_t0 -> retry_retry_t0.
    edges = [
        TaskEdge(from_task_id="t0", to_task_id="retry_t0"),
        TaskEdge(from_task_id="retry_t0", to_task_id="retry_retry_t0"),
    ]
    plan = Plan(id="p0", run_id="run-1", goal_ids=[], tasks=tasks, edges=edges)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_task(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_task)

    executor = SequentialExecutor(
        max_task_invocations=32,
        max_retries_per_task_lineage=2,
        fail_fast=False,
    )
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # Only t0 and retry_t0 are invoked; retry_retry_t0 is refused by the
    # lineage cap and marked FAILED without hitting the adapter.
    assert adapter.invocations == ["t0", "retry_t0"]
    assert outcome.success is False  # one task ended FAILED (retry_retry_t0)
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["t0"] == TaskStatus.COMPLETED
    assert by_id["retry_t0"] == TaskStatus.COMPLETED
    assert by_id["retry_retry_t0"] == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Reachability audit: orphaned PENDING tasks at loop exit.
#
# Belt-and-suspenders for the cancellation cascade (TASK-LIFECYCLE.md §6.1,
# §7.8). If _pick_next_task returns None but PENDING tasks remain, the
# executor must NOT report success=True — it must cancel the orphans,
# emit a CRITICAL PLAN_DIVERGENCE drift, and return success=False.
# ---------------------------------------------------------------------------


async def test_executor_run_with_orphaned_pending_reports_failure() -> None:
    """A cancelled task with a downstream PENDING dep ends the run as failure.

    Reproduces the 'waffles' regression at the executor boundary: the
    first task is cancelled (simulating USER_STEER whose refine failed
    to produce a new plan). The downstream PENDING task has a
    CANCELLED predecessor, so _pick_next_task refuses to pick it. The
    executor must not reach run_completed with success=True; it must
    cascade-cancel the orphan and emit RunAborted.
    """
    plan = _linear_plan(2)  # t0 -> t1
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _cancel_t0(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        # Simulate a STEER-cancel on the current task without any
        # cascade and without a refine that replaces the plan. The
        # StubSteerer.transition flips status directly; this mirrors
        # what the real system used to do before the cascade fix.
        await steerer.transition(task.id, TaskStatus.CANCELLED, session=session)
        return InvocationResult(task_id=task.id, text="cancelled")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_cancel_t0)

    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # Only t0 was invoked (t1 is blocked by CANCELLED predecessor).
    assert adapter.invocations == ["t0"]
    # Reachability audit flipped t1 from PENDING to CANCELLED and
    # ended the run as failure. Without the audit, outcome.success
    # would be True and t1 would stay PENDING forever.
    assert outcome.success is False
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["t0"] == TaskStatus.CANCELLED
    assert by_id["t1"] == TaskStatus.CANCELLED
    # Terminal event is RunAborted (not RunCompleted).
    assert sink.payload_kinds()[-1] == "run_aborted"


async def test_executor_emits_plan_divergence_when_pending_orphaned() -> None:
    """Audit emits a CRITICAL PLAN_DIVERGENCE drift before RunAborted."""
    plan = _linear_plan(3)  # t0 -> t1 -> t2
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _cancel_t0(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.CANCELLED, session=session)
        return InvocationResult(task_id=task.id, text="cancelled")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_cancel_t0)

    executor = SequentialExecutor(max_task_invocations=5)
    await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # The sink saw a DriftDetected(PLAN_DIVERGENCE, CRITICAL) before
    # RunAborted. Match by poking the drift_detected payload fields.
    drift_events = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    assert drift_events, "expected at least one DriftDetected event"
    from goldfive.pb.goldfive.v1 import types_pb2

    audit_drifts = [
        e for e in drift_events if e.drift_detected.kind == types_pb2.DRIFT_KIND_PLAN_DIVERGENCE
    ]
    assert audit_drifts, "expected a PLAN_DIVERGENCE drift from the audit"
    assert audit_drifts[-1].drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "orphaned" in audit_drifts[-1].drift_detected.detail.lower()


async def test_executor_skips_audit_when_all_tasks_terminal() -> None:
    """Audit is a no-op when no PENDING tasks remain — success stays True."""
    plan = _linear_plan(2)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)

    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True
    # No PLAN_DIVERGENCE drift was emitted (no orphans).
    drift_payloads = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert not any(
        e.drift_detected.kind == types_pb2.DRIFT_KIND_PLAN_DIVERGENCE for e in drift_payloads
    )
    assert sink.payload_kinds()[-1] == "run_completed"


# ---------------------------------------------------------------------------
# Goal success predicates (PLAN-LIFECYCLE.md §6.1, third clause).
#
# A run is only successful when every ``Goal.success_predicate`` on the
# session returns True (or is None, vacuously met). If all tasks finish
# but a goal predicate reports the semantic outcome as unmet, the run
# must end with ``success=False`` and a reason that names the goal.
# ---------------------------------------------------------------------------


async def _run_happy_executor(
    *,
    goals: list[Goal],
) -> tuple[Any, Session, RecordingSink]:
    """Helper: run a 2-task linear plan that completes cleanly.

    All goal-predicate tests share the same "all tasks succeed" setup;
    what varies is ``session.goals``. Returns the outcome, the session,
    and the recording sink so tests can assert on both the Outcome and
    the terminal event.
    """
    plan = _linear_plan(2)
    session = _fresh_session()
    session.goals = list(goals)
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)
    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )
    return outcome, session, sink


async def test_run_with_unmet_goal_reports_failure() -> None:
    """Tasks all succeed but a goal predicate returns False -> run fails.

    The happy-path "every task terminal + no orphans" gate passes, but
    the goal predicate tells us the semantic outcome wasn't reached.
    The executor must fail the run and name the goal in the reason.
    """
    goals = [
        Goal(id="g1", summary="deliver the thing", success_predicate=lambda _s: False),
    ]
    outcome, _session, sink = await _run_happy_executor(goals=goals)

    assert outcome.success is False
    # Reason names the goal by its summary so operators can triage.
    assert "deliver the thing" in outcome.reason
    assert "unmet" in outcome.reason.lower()
    # Terminal event is RunAborted (not RunCompleted).
    assert sink.payload_kinds()[-1] == "run_aborted"


async def test_run_with_none_predicate_treats_as_met() -> None:
    """``success_predicate=None`` is vacuously true — run reports success."""
    goals = [
        Goal(id="g1", summary="vacuous goal", success_predicate=None),
    ]
    outcome, _session, sink = await _run_happy_executor(goals=goals)

    assert outcome.success is True
    assert outcome.reason == ""
    assert sink.payload_kinds()[-1] == "run_completed"


async def test_run_with_raising_predicate_treats_as_unmet() -> None:
    """A predicate that raises is logged and treated as unmet."""

    def _boom(_session: Session) -> bool:
        raise RuntimeError("evaluation exploded")

    goals = [
        Goal(id="g1", summary="raising goal", success_predicate=_boom),
    ]
    outcome, _session, sink = await _run_happy_executor(goals=goals)

    assert outcome.success is False
    # Reason carries both the goal name and the exception message.
    assert "raising goal" in outcome.reason
    assert "raised" in outcome.reason.lower()
    assert "evaluation exploded" in outcome.reason
    assert sink.payload_kinds()[-1] == "run_aborted"


async def test_all_goals_met_returns_success() -> None:
    """Standard happy path: every predicate returns True -> success."""
    calls: list[str] = []

    def _met_a(session: Session) -> bool:
        calls.append("a")
        # Predicate sees the actual session (tasks terminal by now).
        assert session.plan is not None
        return True

    def _met_b(session: Session) -> bool:
        calls.append("b")
        return True

    goals = [
        Goal(id="g1", summary="goal A", success_predicate=_met_a),
        Goal(id="g2", summary="goal B", success_predicate=_met_b),
    ]
    outcome, _session, sink = await _run_happy_executor(goals=goals)

    assert outcome.success is True
    assert outcome.reason == ""
    # Both predicates were evaluated.
    assert calls == ["a", "b"]
    assert sink.payload_kinds()[-1] == "run_completed"


async def test_run_with_empty_goals_returns_success() -> None:
    """No goals at all: vacuously met -> success."""
    outcome, _session, sink = await _run_happy_executor(goals=[])
    assert outcome.success is True
    assert outcome.reason == ""
    assert sink.payload_kinds()[-1] == "run_completed"


async def test_unmet_goal_short_circuits_on_first_false() -> None:
    """Goals are evaluated in order; the first unmet goal short-circuits."""
    calls: list[str] = []

    def _first_fails(_s: Session) -> bool:
        calls.append("first")
        return False

    def _second(_s: Session) -> bool:
        calls.append("second")
        return True

    goals = [
        Goal(id="g1", summary="first goal", success_predicate=_first_fails),
        Goal(id="g2", summary="second goal", success_predicate=_second),
    ]
    outcome, _session, _sink = await _run_happy_executor(goals=goals)

    assert outcome.success is False
    # Only the first predicate ran; we short-circuited on its False.
    assert calls == ["first"]
    # Reason names the first goal, not the second.
    assert "first goal" in outcome.reason
    assert "second goal" not in outcome.reason


# ---------------------------------------------------------------------------
# max_task_invocations: rename + unbounded default + deprecation shim.
# ---------------------------------------------------------------------------


async def test_max_task_invocations_unbounded_default_completes_large_plan() -> None:
    """With the new default (``None`` == unbounded), a plan with more
    tasks than the old default cap (32) still runs to completion.

    The executor must not abort with "exhausted max_task_invocations=..."
    when no cap was configured; only the natural plan completion
    terminates the run.
    """
    n = 40  # comfortably above the old default of 32.
    plan = _linear_plan(n)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)

    # Default: no cap passed. Must still complete all 40 tasks.
    executor = SequentialExecutor(max_retries_per_task_lineage=n + 1)
    assert executor.max_task_invocations is None

    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True
    assert len(adapter.invocations) == n
    # goldfive#247: read from session.plan (live) — local ``plan`` is
    # the pre-mutation snapshot.
    assert outcome.session.plan is not None
    for t in outcome.session.plan.tasks:
        assert t.status == TaskStatus.COMPLETED
    assert sink.payload_kinds()[-1] == "run_completed"


async def test_max_task_invocations_explicit_cap_honored() -> None:
    """An explicit ``max_task_invocations=3`` aborts the run after
    exactly 3 adapter invocations when work remains, with a reason
    that names the new parameter.
    """
    plan = _linear_plan(5)
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)

    executor = SequentialExecutor(max_task_invocations=3)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is False
    assert len(adapter.invocations) == 3
    assert "max_task_invocations=3" in outcome.reason
    assert sink.payload_kinds()[-1] == "run_aborted"


async def test_observer_exception_emits_pipeline_failure_drift() -> None:
    """``steerer.observe`` raising must emit an INFO ``CUSTOM`` drift.

    goldfive#134: a bug in the drift pipeline used to be swallowed at
    ``log.debug`` level. The run now surfaces an INFO ``CUSTOM`` drift
    with a ``drift_pipeline_failed:`` detail prefix so sinks see the
    plumbing failure instead of silently accepting the task's output
    as benign.
    """
    plan = _linear_plan(1)
    session = _fresh_session()
    planner = StubPlanner()
    sink = RecordingSink()

    class RaisingSteerer(StubSteerer):
        async def observe(self, event: Any, session: Session) -> None:
            raise RuntimeError("classifier exploded")

    steerer = RaisingSteerer()

    async def _complete_current(
        task: Task, session: Session, steerer: StubSteerer, planner: StubPlanner
    ) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(steerer=steerer, planner=planner, on_invoke=_complete_current)
    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    # The run completed (the plumbing failure is record-only), but a
    # DriftDetected(kind=CUSTOM, severity=INFO) was emitted.
    assert outcome.success is True
    from goldfive.pb.goldfive.v1 import types_pb2

    drift_events = [
        e
        for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected"
    ]
    pipeline_failures = [
        e
        for e in drift_events
        if e.drift_detected.kind == types_pb2.DRIFT_KIND_CUSTOM
        and "drift_pipeline_failed" in e.drift_detected.detail
    ]
    assert pipeline_failures, (
        "expected an INFO CUSTOM 'drift_pipeline_failed' drift after "
        "steerer.observe raised"
    )
    assert (
        pipeline_failures[0].drift_detected.severity
        == types_pb2.DRIFT_SEVERITY_INFO
    )
    assert "classifier exploded" in pipeline_failures[0].drift_detected.detail


def test_deprecation_warning_fires_for_old_kwarg() -> None:
    """Passing ``max_plan_reinvocations=`` emits a :class:`DeprecationWarning`
    and maps to the new attribute name.
    """
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        executor = SequentialExecutor(max_plan_reinvocations=5)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning for max_plan_reinvocations"
    assert "max_task_invocations" in str(deprecations[0].message)
    # The legacy value is mapped onto the new attribute.
    assert executor.max_task_invocations == 5
    # The old attribute is no longer exposed on the instance.
    assert not hasattr(executor, "max_plan_reinvocations")

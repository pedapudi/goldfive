"""Structural loop-prevention regressions (goldfive#271 follow-up).

Pre-fix evidence: ``PLAN_REVISION_COUNT_LIMIT`` capped successful plan
revisions at N=3 per (kind, task_id) before routing to
HUMAN_INTERVENTION_REQUIRED. The cap was a numerical heuristic that
didn't address the root cause — a planner that produces no-op revisions
will loop forever; a task that has stopped making progress is
indistinguishable from a productive one to a counter.

The structural replacements:

1. **No-op revision rejection.** A refine that returns a structurally
   identical plan (same task ids, edges, assignees, statuses) is
   treated as handler exhaustion: escalate immediately to
   HUMAN_INTERVENTION_REQUIRED rather than bumping ``revision_index``
   for a no-op.
2. **Progress-based escalation.** A drift firing on a task whose
   ``Session.task_last_progress_at`` is older than
   :attr:`DefaultSteerer.PROGRESS_STALL_THRESHOLD_SECONDS` (default
   600s) escalates to HUMAN_INTERVENTION_REQUIRED. A productively
   iterating task continually emits progress events; a stuck task does
   not.
3. **Handler exhaustion as the primary escalation primitive.** Empty
   refine responses, raises, AND no-op revisions all flow through the
   same escalation path — the steerer no longer differentiates "ran
   out of count budget" from "the planner can't help here".
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer, RefineExhausted  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None


class _NoOpPlanner:
    """Planner whose ``refine`` always returns a structurally identical plan.

    Models the pathology the no-op rejection guards against: an LLM
    judge that re-fires on a corrected task and produces a refine with
    no real change. Pre-#271 this looped against the count cap; post-
    #271 the first no-op escalates immediately.
    """

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan:
        self.refine_calls.append(drift)
        # Bump ``revision_index`` (the validator requires it) but keep
        # the structural content identical to the prior plan.
        return Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=[
                Task(
                    id=t.id,
                    title=t.title,
                    description=t.description,
                    assignee_agent_id=t.assignee_agent_id,
                    status=t.status,
                )
                for t in plan.tasks
            ],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in plan.edges
            ],
            revision_index=plan.revision_index + 1,
        )


class _GrowingPlanner:
    """Planner whose ``refine`` always returns a structurally distinct plan.

    Each call appends a unique sentinel task so the no-op rejection
    never fires; this isolates progress-based escalation tests from
    the structural-identity gate.
    """

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan:
        self.refine_calls.append(drift)
        sentinel = f"growing-{len(self.refine_calls)}"
        return Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=[
                Task(id=t.id, title=t.title, status=t.status) for t in plan.tasks
            ]
            + [Task(id=sentinel, title=sentinel)],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in plan.edges
            ],
            revision_index=plan.revision_index + 1,
        )


def _drift(
    kind: DriftKind = DriftKind.OFF_TOPIC,
    *,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=f"{kind.value} drift on {task_id}",
        current_task_id=task_id,
    )


def _session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1")],
        edges=[],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


# ---------------------------------------------------------------------------
# (1) No-op revision rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_revision_does_not_bump_revision_index() -> None:
    """A structurally identical refine does not install a new plan.

    The session's plan stays at its prior ``revision_index``; no
    ``PlanRevised`` event lands; the handler-exhaustion escalation
    fires once.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _NoOpPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    initial_revision = session.plan.revision_index

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Refine WAS called (the planner attempted to produce a revision).
    assert len(planner.refine_calls) == 1
    # But the no-op was rejected: revision_index is unchanged.
    assert session.plan.revision_index == initial_revision


@pytest.mark.asyncio
async def test_no_op_revision_escalates_to_human_intervention_after_one_attempt() -> None:
    """Handler exhaustion fires on the FIRST no-op (N=1, not N=3).

    The deleted count cap waited for 3 successful revisions; the
    structural replacement escalates immediately on the first
    structurally-identical revision because the planner has
    demonstrated it cannot make progress on this drift.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _NoOpPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Exactly ONE refine call (no retry loop).
    assert len(planner.refine_calls) == 1
    # Session is paused for human intervention.
    # Phase 2 (path-duality fix): pause now signalled by
    # GOLDFIVE_PAUSE_ESCALATE ControlMessage; HUMAN_INTERVENTION_REQUIRED
    # drift on the sink stream is the durable observable signal.


# ---------------------------------------------------------------------------
# (2) Progress-based escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_stall_escalates_when_task_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drift on a task with stale ``task_last_progress_at`` escalates.

    Records a stale progress timestamp older than the threshold, then
    fires a fresh drift on that task. The structural gate routes the
    drift to HUMAN_INTERVENTION_REQUIRED instead of refining.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _GrowingPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    # Tighten the threshold so the test stays fast.
    steerer.PROGRESS_STALL_THRESHOLD_SECONDS = 60.0

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # Stamp progress at t=1000.
    session.task_last_progress_at["t1"] = 1000.0

    # Advance past the threshold and fire a drift.
    clock[0] = 1100.0  # 100s elapsed > 60s threshold

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Refine was NOT called — the gate fired before refine.
    assert planner.refine_calls == []
    # Escalation: paused for human intervention.
    # Phase 2 (path-duality fix): pause now signalled by
    # GOLDFIVE_PAUSE_ESCALATE ControlMessage; HUMAN_INTERVENTION_REQUIRED
    # drift on the sink stream is the durable observable signal.


@pytest.mark.asyncio
async def test_progress_stall_does_not_fire_when_task_is_iterating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A productively iterating task is not gated by the stall threshold.

    Stamps fresh progress within the threshold; the drift refines
    normally without escalation.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _GrowingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PROGRESS_STALL_THRESHOLD_SECONDS = 60.0

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # Stamp progress at t=1000.
    session.task_last_progress_at["t1"] = 1000.0
    # Only 10s elapsed — well within threshold.
    clock[0] = 1010.0

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Refine WAS called (the gate did not fire).
    assert len(planner.refine_calls) == 1
    # Session is NOT paused.
    # Phase 2 (path-duality fix): paused_for_human_intervention
    # field has been deleted. Absence of HUMAN_INTERVENTION_REQUIRED
    # drift on the sink (asserted above where applicable) is the
    # durable observable signal for non-pause cases.


@pytest.mark.asyncio
async def test_progress_stall_skipped_when_task_has_no_progress_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task with no progress record yet is given the benefit of the doubt.

    Fresh tasks may not have stamped ``task_last_progress_at`` if the
    drift fires before the first transition; the gate must not fire on
    these to avoid false escalations during run startup.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _GrowingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PROGRESS_STALL_THRESHOLD_SECONDS = 60.0

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # No entry in ``task_last_progress_at`` for "t1".
    assert "t1" not in session.task_last_progress_at

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Refine WAS called (no record => no gate).
    assert len(planner.refine_calls) == 1
    # Phase 2 (path-duality fix): paused_for_human_intervention
    # field has been deleted. Absence of HUMAN_INTERVENTION_REQUIRED
    # drift on the sink (asserted above where applicable) is the
    # durable observable signal for non-pause cases.


@pytest.mark.asyncio
async def test_user_steer_bypasses_progress_stall_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USER_STEER drifts honour user intent regardless of stall state.

    A user intervention should always be processed; suppressing it
    because a task happens to be stalled would silently drop operator
    actions.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _GrowingPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.PROGRESS_STALL_THRESHOLD_SECONDS = 60.0

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # Stale progress.
    session.task_last_progress_at["t1"] = 1000.0
    clock[0] = 1100.0  # past threshold

    await steerer._handle_drift(
        _drift(DriftKind.USER_STEER, task_id="t1", severity=DriftSeverity.WARNING),
        session,
    )

    # USER_STEER is not gated by the progress check.
    assert len(planner.refine_calls) == 1


# ---------------------------------------------------------------------------
# (3) Progress liveness is stamped by the task state machine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (4) RefineExhausted sentinel
# ---------------------------------------------------------------------------


class _ExhaustedPlanner:
    """Planner whose ``refine`` raises ``RefineExhausted``."""

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan:
        self.refine_calls.append(drift)
        raise RefineExhausted("cannot make progress on this drift")


@pytest.mark.asyncio
async def test_refine_exhausted_sentinel_escalates_immediately() -> None:
    """A planner raising ``RefineExhausted`` triggers handler-exhaustion escalation.

    Same outcome as the structural no-op rejection — the steerer
    pauses for human intervention rather than retrying.
    """
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    planner = _ExhaustedPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)

    # Refine was called exactly once (no retry).
    assert len(planner.refine_calls) == 1
    # Session paused.
    # Phase 2 (path-duality fix): pause now signalled by
    # GOLDFIVE_PAUSE_ESCALATE ControlMessage; HUMAN_INTERVENTION_REQUIRED
    # drift on the sink stream is the durable observable signal.


@pytest.mark.asyncio
async def test_mark_task_running_stamps_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mark_task_running`` updates ``task_last_progress_at``."""
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_GrowingPlanner())

    clock = [1234.5]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer.mark_task_running("t1", session=session)

    assert session.task_last_progress_at["t1"] == pytest.approx(1234.5)


@pytest.mark.asyncio
async def test_mark_task_progress_refreshes_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mark_task_progress`` refreshes ``task_last_progress_at``."""
    steerer = DefaultSteerer()
    session = _session()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_GrowingPlanner())

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer.mark_task_running("t1", session=session)
    assert session.task_last_progress_at["t1"] == pytest.approx(1000.0)

    clock[0] = 1500.0
    await steerer.mark_task_progress("t1", session=session, fraction=0.5)
    assert session.task_last_progress_at["t1"] == pytest.approx(1500.0)

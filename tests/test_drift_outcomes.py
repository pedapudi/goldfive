"""Tests for the goldfive#215 iter-8 P2 refine-outcome state machine.

The P2 refactor replaces the four-gate refine protection chain (numerical
``refine_failure_counts`` cap + ``KEY_ACTIVE_DRIFTS`` lifecycle gate +
``plan_revision_cooldown_seconds`` time gate + progress-stall gate) with
a single per-(kind, task) :class:`RefineOutcome` table on
:class:`Session` plus the orthogonal progress-stall gate.

These tests pin the new contract end-to-end:

1. First emit of a (kind, task) drift attempts refine and records the
   resulting outcome.
2. A subsequent same-(kind, task) drift on the same turn that finds a
   prior ``"succeeded"`` outcome short-circuits without calling refine.
3. A failed refine increments ``fail_count``; consecutive failures
   accumulate on the same (kind, task) key.
4. When ``fail_count`` crosses ``REFINE_FAILURE_THRESHOLD`` the steerer
   marks the task FAILED (non-recoverable) and emits a CRITICAL
   ``REPEATED_FAILURE`` drift directly via ``_emit_drift_detected``
   (NOT through ``_handle_drift`` — the synthesised drift keys on a
   different (kind, task) tuple than the source so the gate doesn't
   loop).
5. ``USER_STEER`` / ``USER_CANCEL`` / ``GOAL_DRIFT`` bypass the outcome
   gate — operator intent / trajectory drifts must always reach the
   refine path.
6. ``DefaultSteerer.reset_for_turn`` clears the outcome table at the
   ``run_started`` boundary so a fresh turn starts with empty state.
7. Regression for the v30 cross-agent case: same (kind, task) on
   different ``current_agent_id`` values still increments the SAME
   counter (the outcome key is (kind, task), not (kind, task, agent)).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    RefineOutcome,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """Capture all emitted events (proto + dict envelopes)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    @property
    def proto_events(self) -> list[Any]:
        return [e for e in self.events if hasattr(e, "WhichOneof")]


class StubPlanner:
    """Configurable planner stub.

    ``revised`` may be ``None`` (default — refine returns None) or a
    template plan (the stub appends a unique sentinel task per call so
    successive refines don't trip the structural no-op guard). Set
    ``raise_exc`` to a configured exception to make refine raise instead.
    """

    def __init__(
        self,
        *,
        revised: Plan | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.revised = revised
        self.raise_exc = raise_exc
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **_: Any) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.revised is None:
            return None
        sentinel_id = f"stub-refine-{len(self.refine_calls)}"
        return Plan(
            id=self.revised.id,
            run_id=self.revised.run_id,
            goal_ids=list(self.revised.goal_ids),
            tasks=[
                Task(id=t.id, title=t.title, status=t.status)
                for t in self.revised.tasks
            ]
            + [Task(id=sentinel_id, title=sentinel_id)],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in self.revised.edges
            ],
            revision_index=self.revised.revision_index + len(self.refine_calls),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="T2"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=_make_plan(),
    )


def _drift(
    kind: DriftKind = DriftKind.TOOL_ERROR,
    *,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    agent_id: str = "",
    detail: str = "drift",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail,
        current_task_id=task_id,
        current_agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# 1. First emit attempts refine and records the outcome.
# ---------------------------------------------------------------------------


async def test_first_drift_attempts_refine() -> None:
    """No prior outcome -> refine runs -> outcome=succeeded."""
    revised = _make_plan()
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")
    assert session.refine_outcomes.get(key) is None

    await steerer._handle_drift(_drift(), session)

    assert len(planner.refine_calls) == 1
    outcome = session.refine_outcomes[key]
    assert outcome.state == "succeeded"
    assert outcome.fail_count == 0


# ---------------------------------------------------------------------------
# 2. Pre-seeded "succeeded" outcome short-circuits a same-(kind, task) drift.
# ---------------------------------------------------------------------------


async def test_second_drift_after_success_skipped() -> None:
    """A drift on a (kind, task) with prior ``"succeeded"`` outcome
    skips refine entirely — the prior refine already addressed the
    condition; re-running it would be a same-turn no-op replay.
    """
    planner = StubPlanner(revised=_make_plan())
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # Pre-seed the outcome as if a prior refine on this turn had landed.
    key = (DriftKind.TOOL_ERROR.value, "t1")
    session.refine_outcomes[key] = RefineOutcome(state="succeeded", fail_count=0)

    await steerer._handle_drift(_drift(), session)

    # Refine NOT called: the gate short-circuited at the top.
    assert planner.refine_calls == []
    # The outcome is unchanged (no overwrite happened).
    assert session.refine_outcomes[key].state == "succeeded"


# ---------------------------------------------------------------------------
# 3. Failed refine increments fail_count; failures accumulate on the same key.
# ---------------------------------------------------------------------------


async def test_failure_increments_count() -> None:
    """Two consecutive same-(kind, task) drifts with a raising planner
    accumulate ``fail_count`` on the SAME outcome entry, not separate ones.
    """
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")

    # First emit -> refine raises -> outcome=(failed, 1).
    await steerer._handle_drift(_drift(), session)
    assert session.refine_outcomes[key].state == "failed"
    assert session.refine_outcomes[key].fail_count == 1

    # Second emit on the SAME (kind, task) -> outcome.fail_count below
    # threshold so the gate lets refine run again -> raises -> outcome
    # transitions to (failed, 2). The dict has ONE entry (same key
    # overwritten), not two.
    await steerer._handle_drift(_drift(), session)
    assert session.refine_outcomes[key].state == "failed"
    assert session.refine_outcomes[key].fail_count == 2
    # The ONLY non-cascade entries on the outcome table are
    # (TOOL_ERROR, t1) and the cascaded (TASK_FAILED_FATAL, t1) — NOT
    # two distinct TOOL_ERROR rows.
    tool_error_keys = [k for k in session.refine_outcomes if k[0] == DriftKind.TOOL_ERROR.value]
    assert tool_error_keys == [key]


# ---------------------------------------------------------------------------
# 4. Threshold trip: mark_task_failed + REPEATED_FAILURE drift emitted.
# ---------------------------------------------------------------------------


async def test_threshold_reached_marks_task_failed_and_emits_repeated_failure() -> None:
    """At ``fail_count == REFINE_FAILURE_THRESHOLD`` (== 2) the steerer:

    * Calls :meth:`mark_task_failed` (recoverable=False) on the
      drift's task — observable as a ``TaskFailed`` proto event +
      a cascaded ``TaskCancelled`` for downstream tasks +
      a ``TASK_FAILED_FATAL`` ``DriftDetected``.
    * Emits a CRITICAL ``REPEATED_FAILURE`` ``DriftDetected``
      directly (NOT through ``_handle_drift``).
    """
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # Two consecutive same-key failures cross the threshold.
    await steerer._handle_drift(_drift(), session)
    await steerer._handle_drift(_drift(), session)

    from goldfive.pb.goldfive.v1 import types_pb2

    # TaskFailed for t1 (mark_task_failed cascade).
    task_failed_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "task_failed"
    ]
    assert len(task_failed_events) == 1
    assert task_failed_events[0].task_failed.task_id == "t1"
    assert task_failed_events[0].task_failed.recoverable is False

    # TaskCancelled for downstream t2 (cascade_cancel_downstream).
    cancelled_events = [
        e for e in sink.proto_events if e.WhichOneof("payload") == "task_cancelled"
    ]
    assert any(e.task_cancelled.task_id == "t2" for e in cancelled_events)

    # TASK_FAILED_FATAL drift fired by mark_task_failed.
    drift_kinds = [
        e.drift_detected.kind
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
    ]
    assert types_pb2.DRIFT_KIND_TASK_FAILED_FATAL in drift_kinds

    # REPEATED_FAILURE drift emitted directly with CRITICAL severity.
    repeated = [
        e
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_REPEATED_FAILURE
    ]
    assert len(repeated) == 1
    assert repeated[0].drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL


# ---------------------------------------------------------------------------
# 5. USER_STEER bypasses the outcome gate.
# ---------------------------------------------------------------------------


async def test_user_steer_bypasses_outcome_check() -> None:
    """A pre-seeded ``"succeeded"`` outcome on USER_STEER does NOT
    short-circuit a fresh USER_STEER drift. ``_record_refine_outcome``
    also doesn't write anything for USER_STEER, so the outcome is
    unchanged — operator intent is always honoured.
    """
    revised = _make_plan()
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # Pre-seed an outcome on USER_STEER. The outcome gate would
    # short-circuit here for an autonomous drift, but USER_STEER is
    # exempt.
    key = (DriftKind.USER_STEER.value, "t1")
    session.refine_outcomes[key] = RefineOutcome(state="succeeded", fail_count=0)

    user_drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please pivot",
        current_task_id="t1",
    )
    await steerer._handle_drift(user_drift, session)

    # Refine ran (USER_STEER bypassed the gate).
    assert len(planner.refine_calls) == 1
    # Outcome unchanged (USER_STEER doesn't write outcome bookkeeping).
    assert session.refine_outcomes[key].state == "succeeded"


# ---------------------------------------------------------------------------
# 6. reset_for_turn clears the outcome table.
# ---------------------------------------------------------------------------


def test_run_started_resets_outcomes() -> None:
    """``DefaultSteerer.reset_for_turn`` clears the entire
    ``session.refine_outcomes`` dict — wired from
    ``Runner.run`` immediately after the ``run_started`` emit, so each
    turn starts with empty per-(kind, task) bookkeeping.
    """
    steerer = DefaultSteerer()
    session = Session(run_id="r1")
    session.refine_outcomes[("tool_error", "t1")] = RefineOutcome(
        state="failed", fail_count=1
    )
    session.refine_outcomes[("plan_divergence", "t2")] = RefineOutcome(
        state="succeeded", fail_count=0
    )
    assert len(session.refine_outcomes) == 2

    steerer.reset_for_turn(session)

    assert session.refine_outcomes == {}


# ---------------------------------------------------------------------------
# 7. Regression — same (kind, task) on different agent_id increments same key.
# ---------------------------------------------------------------------------


async def test_v30_repro() -> None:
    """v30 cross-agent regression: ``current_agent_id`` is NOT part of
    the outcome key.

    Before the P2 refactor, the lifecycle gate keyed on
    ``(kind, task, agent, turn)`` so callers had to vary ``agent_id``
    across emits to bypass same-condition de-duplication, while the
    failure counter (then named ``refine_failure_counts``) keyed only on
    ``(kind, task)``. The new outcome state machine inherits the
    counter's (kind, task) key — so two emits on the same (kind, task)
    but different ``current_agent_id`` values legitimately accumulate
    on the SAME outcome, and the threshold trips on the second emit.
    """
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")

    await steerer._handle_drift(_drift(agent_id="agent-a"), session)
    assert session.refine_outcomes[key].fail_count == 1

    await steerer._handle_drift(_drift(agent_id="agent-b"), session)
    # Threshold tripped on the SECOND emit despite distinct agent_ids.
    assert session.refine_outcomes[key].fail_count == 2
    assert _task_status(session, "t1") is TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_status(session: Session, task_id: str) -> TaskStatus:
    assert session.plan is not None
    for t in session.plan.tasks:
        if t.id == task_id:
            return t.status
    raise AssertionError(f"task {task_id!r} missing from plan")

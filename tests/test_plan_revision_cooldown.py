"""Unit tests for the drift-triggered plan-revision cooldown.

The cooldown lives in :class:`goldfive.steerer.DefaultSteerer` and
gates ``planner.refine`` calls on the ``(task_id, drift_kind)`` key.
These tests drive ``_handle_drift`` directly so each fire is isolated
from :meth:`DefaultSteerer.observe`'s drift classifier.

See goldfive feedback-loop fix (CRITICAL drift spam producing back-to-
back ``plan_revised`` rows within seconds).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import ControlKind, ControlMessage  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
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


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class StubPlanner:
    """Planner stub whose ``refine`` returns a fresh plan every call."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append({"drift": drift})
        # Return a minimally-valid revised plan. Re-use the incoming
        # plan's tasks/edges so ``Plan.validate(for_revision=True,
        # prior=plan)`` passes -- the steerer runs that after refine.
        revised = Plan(
            id=plan.id,
            run_id=plan.run_id,
            goal_ids=list(plan.goal_ids),
            tasks=[Task(id=t.id, title=t.title, status=t.status) for t in plan.tasks],
            edges=[
                TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
                for e in plan.edges
            ],
            revision_index=plan.revision_index + 1,
        )
        return revised


def _make_plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1"), Task(id="t2", title="T2")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _make_session() -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=_make_plan(),
    )


def _drift(
    kind: DriftKind,
    *,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    detail: str = "drift",
) -> DriftEvent:
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=detail,
        current_task_id=task_id,
    )


def _plan_revised_count(sink: ListSink) -> int:
    return sum(1 for e in sink.events if e.WhichOneof("payload") == "plan_revised")


def _build(
    cooldown: float = 30.0,
) -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    steerer = DefaultSteerer(plan_revision_cooldown_seconds=cooldown)
    sink = ListSink()
    planner = StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)  # type: ignore[arg-type]
    session = _make_session()
    return steerer, session, sink, planner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_two_off_topic_drifts_within_window_emit_one_plan_revised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive OFF_TOPIC drifts within 5s -> one plan_revised."""
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)
    clock[0] += 5.0
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    assert _plan_revised_count(sink) == 1
    assert len(planner.refine_calls) == 1


async def test_two_off_topic_drifts_past_cooldown_emit_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two OFF_TOPIC drifts 35s apart -> both plan_revised emissions fire."""
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)
    clock[0] += 35.0
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    assert _plan_revised_count(sink) == 2
    assert len(planner.refine_calls) == 2


async def test_different_drift_kinds_do_not_share_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF_TOPIC then LOOPING_REASONING within 5s -> both fire.

    The cooldown is scoped per ``(task_id, drift_kind)`` so genuinely
    different problems replan independently.
    """
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)
    clock[0] += 5.0
    await steerer._handle_drift(_drift(DriftKind.LOOPING_REASONING), session)

    assert _plan_revised_count(sink) == 2
    assert len(planner.refine_calls) == 2


async def test_different_task_ids_do_not_share_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift on task_A then task_B within 5s -> both fire."""
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)
    clock[0] += 5.0
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t2"), session)

    assert _plan_revised_count(sink) == 2
    assert len(planner.refine_calls) == 2


async def test_user_steer_drift_bypasses_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USER_STEER drift fires immediately regardless of a recent revision.

    A user steer is an explicit intervention -- we always honour it,
    even if goldfive just replanned the same task for an autonomous
    drift.
    """
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # Prime the cooldown with an autonomous drift on t1.
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)
    clock[0] += 2.0

    # Build a USER_STEER drift the way _drift_from_control does: with a
    # ControlMessage carrying an annotation_id in ``raw``.
    ctrl = ControlMessage(
        kind=ControlKind.STEER,
        payload={"annotation_id": "ann_123", "note": "please pivot to X"},
    )
    user_drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please pivot to X",
        current_task_id="t1",
        raw=ctrl,
    )
    await steerer._handle_drift(user_drift, session)

    # Two plan_revised rows: one for the autonomous OFF_TOPIC, one for
    # the user steer that arrived inside the cooldown window.
    assert _plan_revised_count(sink) == 2
    # Both refine calls fired (USER_STEER wasn't suppressed).
    assert len(planner.refine_calls) == 2


async def test_goal_drift_does_not_consume_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GOAL_DRIFT path is unaffected by this cooldown (rate-limited elsewhere).

    GOAL_DRIFT maps to PAUSE_ESCALATE in the intervention ladder, so it
    does NOT hit the refine path and therefore does not stamp the
    cooldown table. The second OFF_TOPIC drift should see an empty
    cooldown entry and fire normally.
    """
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    # Fire a GOAL_DRIFT first; it routes to PAUSE_ESCALATE and does not
    # invoke planner.refine.
    await steerer._handle_drift(
        _drift(DriftKind.GOAL_DRIFT, task_id="t1", severity=DriftSeverity.CRITICAL),
        session,
    )
    assert len(planner.refine_calls) == 0
    assert _plan_revised_count(sink) == 0

    # Reset pause flag so the next handle_drift isn't gated by the
    # pause bookkeeping; the cooldown under test is orthogonal to pause.
    session.paused_for_human_intervention = False

    clock[0] += 5.0
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC, task_id="t1"), session)
    assert _plan_revised_count(sink) == 1
    assert len(planner.refine_calls) == 1


async def test_cooldown_zero_disables_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """``plan_revision_cooldown_seconds=0.0`` disables the cooldown entirely.

    Two back-to-back identical drifts both replan.
    """
    steerer, session, sink, planner = _build(cooldown=0.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)
    # Same tick; with cooldown==0 the gate no-ops.
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    assert _plan_revised_count(sink) == 2
    assert len(planner.refine_calls) == 2


async def test_suppressed_revision_does_not_emit_extra_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suppressed revision emits DriftDetected but no PlanRevised / refine.

    Operators still see the drift on the wire -- only the replan is
    throttled.
    """
    steerer, session, sink, planner = _build(cooldown=30.0)

    clock = [1000.0]
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: clock[0])

    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)
    clock[0] += 1.0
    await steerer._handle_drift(_drift(DriftKind.OFF_TOPIC), session)

    kinds = [e.WhichOneof("payload") for e in sink.events]
    # First handle: drift_detected + plan_revised.
    # Second handle: drift_detected only (suppressed).
    assert kinds == ["drift_detected", "plan_revised", "drift_detected"]
    assert len(planner.refine_calls) == 1


# ---------------------------------------------------------------------------
# PlanRevised refine-context observability (judge-observability event)
# ---------------------------------------------------------------------------


def _plan_revised_events(sink: ListSink) -> list[Any]:
    return [e for e in sink.events if e.WhichOneof("payload") == "plan_revised"]


async def test_plan_revised_event_populates_refine_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PlanRevised carries refine_input_summary + refine_output_summary."""
    steerer, session, sink, _planner = _build(cooldown=0.0)
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: 1000.0)

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="agent drifted to raccoons",
        current_task_id="t1",
        current_agent_id="researcher",
    )
    await steerer._handle_drift(drift, session)

    events = _plan_revised_events(sink)
    assert len(events) == 1
    pr = events[0].plan_revised
    # Input summary names the drift kind + severity + detail + prior plan shape.
    # ``DriftKind.value`` is lowercase (e.g. "off_topic"), so that's what
    # gets rendered into the human-readable summary.
    assert "off_topic" in pr.refine_input_summary
    assert "warning" in pr.refine_input_summary
    assert "raccoons" in pr.refine_input_summary
    assert "task=t1" in pr.refine_input_summary
    assert "prior_plan=rev0" in pr.refine_input_summary
    # Output summary carries the revised index + task count + titles.
    assert "revision_index=1" in pr.refine_output_summary
    assert "tasks=2" in pr.refine_output_summary


async def test_plan_revised_event_stamps_target_agent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """target_agent_id mirrors DriftEvent.current_agent_id for agent-scoped refines."""
    steerer, session, sink, _planner = _build(cooldown=0.0)
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: 1000.0)

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="off topic",
        current_task_id="t1",
        current_agent_id="researcher",
    )
    await steerer._handle_drift(drift, session)

    pr = _plan_revised_events(sink)[0].plan_revised
    assert pr.target_agent_id == "researcher"


async def test_plan_revised_event_target_agent_id_empty_for_trajectory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drift with no current_agent_id → target_agent_id == ""."""
    steerer, session, sink, _planner = _build(cooldown=0.0)
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: 1000.0)

    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail="unscoped",
        current_task_id="t1",
        current_agent_id="",  # no agent bound — trajectory-level
    )
    await steerer._handle_drift(drift, session)

    pr = _plan_revised_events(sink)[0].plan_revised
    assert pr.target_agent_id == ""


async def test_plan_revised_refine_input_summary_truncates_when_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pathological drift.detail does not blow the refine_input_summary field."""
    steerer, session, sink, _planner = _build(cooldown=0.0)
    monkeypatch.setattr("goldfive.steerer.time.monotonic", lambda: 1000.0)

    huge_detail = "x" * 5000
    drift = DriftEvent(
        kind=DriftKind.OFF_TOPIC,
        severity=DriftSeverity.WARNING,
        detail=huge_detail,
        current_task_id="t1",
        current_agent_id="researcher",
    )
    await steerer._handle_drift(drift, session)

    pr = _plan_revised_events(sink)[0].plan_revised
    assert pr.refine_input_summary.endswith(" … [truncated]")
    # 2048 + suffix length.
    assert len(pr.refine_input_summary) == 2048 + len(" … [truncated]")

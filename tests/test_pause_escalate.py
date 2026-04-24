"""Tests for Level 4 (PAUSE_ESCALATE) of the intervention ladder.

Checks that:

* A drift routing to Level 4 sets ``session.paused_for_human_intervention``.
* A ``HUMAN_INTERVENTION_REQUIRED`` drift is emitted at CRITICAL.
* A subsequent ``CONTROL_RESUME`` or ``CONTROL_STEER`` clears the flag.
* The executor's pre-task loop blocks while the flag is set (integration
  test using a real ``ControlChannel``).

See goldfive#142 for the ladder spec.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.control import (  # noqa: E402
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.executors._control import dispatch_control  # noqa: E402
from goldfive.steerer import DefaultSteerer, InterventionLevel  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """In-memory sink that records every emitted proto ``Event``."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:  # pragma: no cover - not used
        return


class StubPlanner:
    """Planner stub that records refine calls; returns no revised plan."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:  # pragma: no cover
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


def _fresh() -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    # goldfive-steer-unification: pause-escalate tests exercise the
    # LEGACY ladder semantics (INTENT_DIVERGENCE CRITICAL -> pause,
    # HUMAN_INTERVENTION_REQUIRED -> pause) so we explicitly disable
    # the new drift-to-steer promotion. The promotion path itself is
    # covered in tests/test_steer_unification.py.
    steerer = DefaultSteerer(goldfive_steer_threshold="off")
    session = Session(run_id="pause-test", current_task_id="t1")
    session.plan = Plan(
        id="p1",
        run_id="pause-test",
        goal_ids=[],
        tasks=[Task(id="t1", title="work", status=TaskStatus.RUNNING)],
        edges=[],
    )
    sink = ListSink()
    planner = StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink, planner


def _drift_kind_pb(name: str) -> Any:
    from goldfive.pb.goldfive.v1 import types_pb2

    return getattr(types_pb2, f"DRIFT_KIND_{name}")


# ---------------------------------------------------------------------------
# Level 4 pause triggering
# ---------------------------------------------------------------------------


async def test_pause_escalate_sets_session_flag_and_emits_drift() -> None:
    steerer, session, sink, planner = _fresh()
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="critical intent drift",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)

    assert session.paused_for_human_intervention is True
    # Sink got the original drift + the HUMAN_INTERVENTION_REQUIRED escalation.
    kinds = [
        evt.drift_detected.kind
        for evt in sink.events
        if evt.WhichOneof("payload") == "drift_detected"
    ]
    assert _drift_kind_pb("INTENT_DIVERGENCE") in kinds
    assert _drift_kind_pb("HUMAN_INTERVENTION_REQUIRED") in kinds
    # Level 4 does not call refine.
    assert planner.refine_calls == []


async def test_pause_escalate_does_not_double_emit_human_intervention() -> None:
    """A HUMAN_INTERVENTION_REQUIRED drift entering the ladder should
    NOT be re-escalated into a second HUMAN_INTERVENTION_REQUIRED
    drift -- the original emission at the top of _handle_drift already
    carried the signal."""
    steerer, session, sink, _planner = _fresh()
    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="stuck",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)

    assert session.paused_for_human_intervention is True
    human_intervention_events = [
        e
        for e in sink.events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == _drift_kind_pb("HUMAN_INTERVENTION_REQUIRED")
    ]
    assert len(human_intervention_events) == 1


async def test_refine_validation_failed_pauses_without_refining() -> None:
    steerer, session, sink, planner = _fresh()
    drift = DriftEvent(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        severity=DriftSeverity.CRITICAL,
        detail="planner retry budget spent",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)
    assert session.paused_for_human_intervention is True
    assert planner.refine_calls == []
    # HUMAN_INTERVENTION_REQUIRED escalation is on the sink.
    assert any(
        e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == _drift_kind_pb("HUMAN_INTERVENTION_REQUIRED")
        for e in sink.events
    )


async def test_ladder_level_for_pause_escalate_on_repeat() -> None:
    """Threshold-crossing occurrence drives CRITICAL to Level 4.

    Uses PLAN_DIVERGENCE: first CRITICAL = CANCEL_REINVOKE, repeat =
    PAUSE_ESCALATE.
    """
    steerer = DefaultSteerer()
    assert (
        steerer._ladder_level_for(DriftKind.PLAN_DIVERGENCE, DriftSeverity.CRITICAL, 0)
        is InterventionLevel.CANCEL_REINVOKE
    )
    # Threshold is 2 in DefaultSteerer.REFINE_FAILURE_THRESHOLD.
    assert (
        steerer._ladder_level_for(
            DriftKind.PLAN_DIVERGENCE,
            DriftSeverity.CRITICAL,
            DefaultSteerer.REFINE_FAILURE_THRESHOLD,
        )
        is InterventionLevel.PAUSE_ESCALATE
    )


# ---------------------------------------------------------------------------
# Control-side resume / steer clears the flag
# ---------------------------------------------------------------------------


async def test_resume_clears_paused_flag() -> None:
    _steerer, session, sink, _planner = _fresh()
    session.paused_for_human_intervention = True
    msg = ControlMessage(kind=ControlKind.RESUME)

    class _NullSteerer:
        async def observe(self, *args: Any, **kwargs: Any) -> None:
            return

    outcome = await dispatch_control(msg, session=session, steerer=_NullSteerer(), sinks=[sink])
    assert outcome.request_resume is True
    assert session.paused_for_human_intervention is False


async def test_steer_clears_paused_flag() -> None:
    _steerer, session, sink, _planner = _fresh()
    session.paused_for_human_intervention = True
    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "try a different approach"},
    )

    class _NullSteerer:
        async def observe(self, *args: Any, **kwargs: Any) -> None:
            return

    outcome = await dispatch_control(msg, session=session, steerer=_NullSteerer(), sinks=[sink])
    assert outcome.steer_message is not None
    assert session.paused_for_human_intervention is False


# ---------------------------------------------------------------------------
# Executor-integration: pre-task loop blocks on session flag
# ---------------------------------------------------------------------------


async def test_executor_pre_task_blocks_on_session_pause_flag() -> None:
    """The sequential executor's ``_apply_pre_task_controls`` should
    block when ``session.paused_for_human_intervention`` is set, and
    unblock when a RESUME arrives on the control channel.
    """
    from goldfive.executors.sequential import SequentialExecutor

    _steerer, session, sink, _planner = _fresh()
    session.paused_for_human_intervention = True
    channel = ControlChannel()

    executor = SequentialExecutor()

    # Start the pre-task wait; it should NOT return immediately.
    wait_task = asyncio.create_task(
        executor._apply_pre_task_controls(
            control=channel,
            session=session,
            steerer=_steerer,
            sinks=[sink],
        )
    )
    # Yield once so the wait task has a chance to enter its paused loop.
    await asyncio.sleep(0)
    assert not wait_task.done(), "executor should block while pause flag is set"

    # Send RESUME; the wait task should complete.
    await channel.send(ControlMessage(kind=ControlKind.RESUME))
    result = await asyncio.wait_for(wait_task, timeout=1.0)
    stop, steer_msg = result
    assert stop is False
    assert steer_msg is None
    assert session.paused_for_human_intervention is False


async def test_executor_pre_task_no_control_channel_clears_flag() -> None:
    """Without a control channel the executor cannot wait for a user
    action. Preserve liveness by clearing the flag -- the
    HUMAN_INTERVENTION_REQUIRED drift on the sink stream is the
    durable signal."""
    from goldfive.executors.sequential import SequentialExecutor

    _steerer, session, sink, _planner = _fresh()
    session.paused_for_human_intervention = True

    executor = SequentialExecutor()
    result = await executor._apply_pre_task_controls(
        control=None,
        session=session,
        steerer=_steerer,
        sinks=[sink],
    )
    stop, steer_msg = result
    assert stop is False
    assert steer_msg is None
    assert session.paused_for_human_intervention is False


# ---------------------------------------------------------------------------
# Level 2 / 3 handoff slots
# ---------------------------------------------------------------------------


async def test_nudge_queues_message_on_session() -> None:
    """Level 2 dispatch queues a nudge the Runner's overlay loop
    (goldfive#141) picks up after the current invocation ends."""
    steerer, session, _sink, _planner = _fresh()
    # CONFUSION at WARNING -> Level 2 (NUDGE) in the issue table. My
    # implementation maps this to ABSORB to preserve existing refine-
    # on-WARNING semantics; force the level via a direct dispatch.
    drift = DriftEvent(
        kind=DriftKind.CONFUSION,
        severity=DriftSeverity.WARNING,
        detail="agent uncertain",
        current_task_id="t1",
    )
    await steerer._dispatch_nudge(drift, session)
    assert len(session.pending_nudges) == 1
    assert "t1" in session.pending_nudges[0]


async def test_cancel_reinvoke_queues_corrective_message() -> None:
    """Level 3 handoff stashes a composed corrective message on the
    session for the Runner's overlay loop to pick up. A refine that
    returns a revised plan is a prerequisite; with a stub planner that
    returns None the Level 3 slot stays clear."""
    steerer, session, _sink, _planner = _fresh()

    # Force the level-3 codepath by using a planner that returns a
    # real revised plan (we already have a bound StubPlanner that
    # returns None, which makes refine "fail" and prevents the Level 3
    # slot from being written). Install a functioning planner here.
    class GoodPlanner:
        async def generate(self, **_kw: Any) -> Plan | None:  # pragma: no cover
            return None

        async def refine(self, **_kw: Any) -> Plan | None:
            return Plan(
                id="p-revised",
                run_id="pause-test",
                goal_ids=[],
                tasks=[
                    Task(
                        id="t1",
                        title="Try a different approach",
                        status=TaskStatus.PENDING,
                    )
                ],
                edges=[],
                revision_index=1,
            )

    steerer.bind(sinks=[], planner=GoodPlanner())
    # Use TOOL_ERROR for the Level 3 first-occurrence check:
    # LOOPING_REASONING's CRITICAL-first tier routes to NUDGE
    # (Level 2) after goldfive#204, so we pick a drift kind whose
    # CRITICAL-first tier still routes to CANCEL_REINVOKE AND whose
    # corrective-template interpolates the current_task_id.
    drift = DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.CRITICAL,
        detail="tool error",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)
    assert session.pending_corrective_message is not None
    assert "t1" in session.pending_corrective_message
    assert "Try a different approach" in session.pending_corrective_message

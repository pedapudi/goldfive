"""Tests for Level 4 (PAUSE_ESCALATE) of the intervention ladder.

Phase 2 of the path-duality fix re-routed Level 4 through a
``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage on the bound channel,
replacing the deleted ``Session.paused_for_human_intervention`` flag.
These tests now check that:

* A drift routing to Level 4 dispatches a ``GOLDFIVE_PAUSE_ESCALATE``
  ControlMessage on the bound channel.
* A ``HUMAN_INTERVENTION_REQUIRED`` drift is emitted at CRITICAL.
* A subsequent ``CONTROL_RESUME`` or ``CONTROL_STEER`` unblocks the
  executor's pre-task loop (the channel-state semantics are unchanged).
* The executor's pre-task loop blocks until that resume/steer arrives.

See goldfive#142 for the original ladder spec; the path-duality fix
brief in ``docs/design/DRIFT.md`` for the Phase 2 routing change.
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


def _fresh() -> tuple[DefaultSteerer, Session, ListSink, StubPlanner, ControlChannel]:
    # goldfive-steer-unification: pause-escalate tests exercise the
    # LEGACY ladder semantics (INTENT_DIVERGENCE CRITICAL -> pause,
    # HUMAN_INTERVENTION_REQUIRED -> pause) so we explicitly disable
    # the new drift-to-steer promotion. The promotion path itself is
    # covered in tests/test_steer_unification.py.
    #
    # Phase 2 of the path-duality fix: bind a real ControlChannel so
    # the steerer's ``GOLDFIVE_PAUSE_ESCALATE`` dispatch lands somewhere
    # observable to the assertions.
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
    channel = ControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)
    return steerer, session, sink, planner, channel


def _drain_goldfive_pause(channel: ControlChannel) -> list[ControlMessage]:
    """Pop every queued ``GOLDFIVE_PAUSE_ESCALATE`` from the channel."""
    drained: list[ControlMessage] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        msg = inbox.get_nowait()
        if msg.kind is ControlKind.GOLDFIVE_PAUSE_ESCALATE:
            drained.append(msg)
    return drained


def _drain_goldfive_steer(channel: ControlChannel) -> list[ControlMessage]:
    """Pop every queued ``GOLDFIVE_STEER`` from the channel."""
    drained: list[ControlMessage] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        msg = inbox.get_nowait()
        if msg.kind is ControlKind.GOLDFIVE_STEER:
            drained.append(msg)
    return drained


def _drift_kind_pb(name: str) -> Any:
    from goldfive.pb.goldfive.v1 import types_pb2

    return getattr(types_pb2, f"DRIFT_KIND_{name}")


# ---------------------------------------------------------------------------
# Level 4 pause triggering
# ---------------------------------------------------------------------------


async def test_pause_escalate_dispatches_control_and_emits_drift() -> None:
    steerer, session, sink, planner, channel = _fresh()
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="critical intent drift",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)

    pause_msgs = _drain_goldfive_pause(channel)
    assert len(pause_msgs) == 1
    assert pause_msgs[0].payload["drift_kind"] == DriftKind.INTENT_DIVERGENCE.value
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
    steerer, session, sink, _planner, channel = _fresh()
    drift = DriftEvent(
        kind=DriftKind.HUMAN_INTERVENTION_REQUIRED,
        severity=DriftSeverity.CRITICAL,
        detail="stuck",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)

    assert _drain_goldfive_pause(channel)  # at least one dispatched
    human_intervention_events = [
        e
        for e in sink.events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == _drift_kind_pb("HUMAN_INTERVENTION_REQUIRED")
    ]
    assert len(human_intervention_events) == 1


async def test_refine_validation_failed_pauses_without_refining() -> None:
    steerer, session, sink, planner, channel = _fresh()
    drift = DriftEvent(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        severity=DriftSeverity.CRITICAL,
        detail="planner retry budget spent",
        current_task_id="t1",
    )
    await steerer._handle_drift(drift, session)
    assert _drain_goldfive_pause(channel)
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


async def test_resume_unblocks_pause() -> None:
    """RESUME on the channel emits ``request_resume=True``; the executor's
    pre-task pause loop unwinds when it sees this. Phase 2 of the
    path-duality fix collapsed the goldfive-side pause flag onto the
    same channel state, so a single RESUME unblocks both
    user-initiated PAUSE and goldfive-initiated PAUSE_ESCALATE."""
    _steerer, _session, sink, _planner, _channel = _fresh()
    msg = ControlMessage(kind=ControlKind.RESUME)

    class _NullSteerer:
        async def observe(self, *args: Any, **kwargs: Any) -> None:
            return

    outcome = await dispatch_control(
        msg, session=_session, steerer=_NullSteerer(), sinks=[sink]
    )
    assert outcome.request_resume is True


async def test_steer_acts_as_resume() -> None:
    """A STEER ControlMessage carries the implicit RESUME semantics:
    when the executor's pre-task loop sees a steer_message it breaks
    out of the paused wait. Phase 2 of the path-duality fix relies on
    this: a STEER from the operator unblocks any goldfive-initiated
    pause so the corrective intent can land."""
    _steerer, _session, sink, _planner, _channel = _fresh()
    msg = ControlMessage(
        kind=ControlKind.STEER,
        payload={"note": "try a different approach"},
    )

    class _NullSteerer:
        async def observe(self, *args: Any, **kwargs: Any) -> None:
            return

    outcome = await dispatch_control(
        msg, session=_session, steerer=_NullSteerer(), sinks=[sink]
    )
    assert outcome.steer_message is not None


# ---------------------------------------------------------------------------
# Executor-integration: pre-task loop blocks on session flag
# ---------------------------------------------------------------------------


async def test_executor_pre_task_blocks_on_goldfive_pause_control() -> None:
    """The sequential executor's ``_apply_pre_task_controls`` should
    block when the steerer queues a ``GOLDFIVE_PAUSE_ESCALATE`` on the
    channel, and unblock when a RESUME arrives. This is the channel-
    routed replacement for the deleted
    ``session.paused_for_human_intervention`` flag.
    """
    from goldfive.executors.sequential import SequentialExecutor

    _steerer, session, sink, _planner, channel = _fresh()
    # Pre-queue a goldfive pause on the channel so the drain picks it
    # up and request_pause flips on.
    await channel.send(
        ControlMessage(
            kind=ControlKind.GOLDFIVE_PAUSE_ESCALATE,
            payload={"reason": "test pause"},
        )
    )

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
    # Yield so the wait task drains the queued pause and enters its
    # paused loop.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not wait_task.done(), "executor should block on goldfive pause"

    # Send RESUME; the wait task should complete.
    await channel.send(ControlMessage(kind=ControlKind.RESUME))
    result = await asyncio.wait_for(wait_task, timeout=1.0)
    stop, steer_msg = result
    assert stop is False
    assert steer_msg is None


async def test_executor_pre_task_no_control_channel_does_not_wedge() -> None:
    """Without a control channel the steerer's
    ``GOLDFIVE_PAUSE_ESCALATE`` dispatch is a best-effort no-op;
    the executor's pre-task loop must not wedge waiting for a
    pause that has nowhere to land. The
    ``HUMAN_INTERVENTION_REQUIRED`` drift on the sink stream is the
    durable signal."""
    from goldfive.executors.sequential import SequentialExecutor

    _steerer, session, sink, _planner, _channel = _fresh()

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


# ---------------------------------------------------------------------------
# Level 2 / 3 handoff slots
# ---------------------------------------------------------------------------


async def test_nudge_queues_message_on_session() -> None:
    """Level 2 dispatch queues a nudge the Runner's overlay loop
    (goldfive#141) picks up after the current invocation ends."""
    steerer, session, _sink, _planner, _channel = _fresh()
    # Direct ``_dispatch_nudge`` exercise: any drift kind that maps
    # to ABSORB at WARNING in the ladder works as input here -- the
    # test pins the queueing behaviour, not the kind-specific routing.
    drift = DriftEvent(
        kind=DriftKind.SELF_REPORTED_STUCK,
        severity=DriftSeverity.WARNING,
        detail="agent reported no progress",
        current_task_id="t1",
    )
    await steerer._dispatch_nudge(drift, session)
    assert len(session.pending_nudges) == 1
    assert "t1" in session.pending_nudges[0]


async def test_cancel_reinvoke_dispatches_goldfive_steer_control() -> None:
    """Level 3 handoff dispatches a ``GOLDFIVE_STEER`` ControlMessage
    on the bound channel so the executor cancels the in-flight invoke
    and restarts with a corrective body. A refine that returns a
    revised plan is a prerequisite; with a stub planner that returns
    None the dispatch stays empty.

    Phase 2 of the path-duality fix: replaces the deleted
    ``session.pending_corrective_message`` write."""
    steerer, session, _sink, _planner, channel = _fresh()

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
    steer_msgs = _drain_goldfive_steer(channel)
    assert len(steer_msgs) == 1
    payload = steer_msgs[0].payload
    assert payload["drift_kind"] == DriftKind.TOOL_ERROR.value
    assert "t1" in payload["body"]
    assert "Try a different approach" in payload["body"]

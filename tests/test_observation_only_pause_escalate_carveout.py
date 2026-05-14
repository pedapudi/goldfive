"""observation_only carve-out for GOLDFIVE_PAUSE_ESCALATE dispatch (goldfive#264).

The steerer's intervention ladder routes refine-handler exhaustion, no-op
plan revisions, ``planner.refine`` returning ``None``, validator-rejected
revisions, and progress-stall escalations all through
:meth:`~goldfive.steerer.DefaultSteerer._dispatch_goldfive_pause_control`.
That helper mints a ``GOLDFIVE_PAUSE_ESCALATE`` ControlMessage on the
bound channel; the executor's overlay loop reads the message, cancels
the in-flight ``invoke_passthrough`` task, and ends the overlay turn
(see ``sequential.py`` line 1530+: ``goldfive_pause_message`` branch of
the control dispatcher).

Under :class:`~goldfive.config.SteeringConfig.observation_only` this
cancel-and-end-overlay sequence violates the "passive — observe, don't
enforce" contract: goldfive is supposed to detect, run the planner
(dry-run), emit ``PlanRevised(dry_run=True)``, and otherwise NOT touch
the live invocation. #260 fixed the executor's ``run_aborted`` path but
left this one; #264 closes the gap.

The carve-out gates at TWO layers:

1. **Primary gate** in :meth:`_dispatch_goldfive_pause_control`: under
   ``observation_only`` the method logs the would-be payload at INFO
   and returns ``False`` WITHOUT calling ``channel.send``. The
   originating ``HUMAN_INTERVENTION_REQUIRED`` ``DriftDetected`` emit
   in the caller (e.g. ``_emit_handler_exhausted_escalation``) is
   OUTSIDE the dispatch and continues to fire — sinks still see the
   escalation, the operator can still react, but the live invocation
   is not cancelled.

2. **Defense-in-depth** in
   :class:`~goldfive.executors.sequential.SequentialExecutor`: the
   ``goldfive_pause_message`` branch in
   ``_invoke_passthrough_with_control`` and the ``goldfive_pause`` arm
   of the overlay loop both check ``steerer._observation_only``. A
   custom steerer or future code path that bypasses the primary gate
   would otherwise still drive ``_cancel_invoke_task`` here; the
   defense-in-depth drops the message and lets the loop continue.

Live reproduction (2026-05-11, session
``4538863f-0dea-4fe8-97b4-5f660ee2cb7f``): an OFF_TOPIC drift under
``observation_only=True`` reached refine handler exhaustion via the
#271 no-op-revision detector, which dispatched
``GOLDFIVE_PAUSE_ESCALATE`` on the control channel, which the
SequentialExecutor handled by cancelling the in-flight invoke and
ending the overlay turn. That bypassed observation_only's contract.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.control import ControlKind  # noqa: E402
from goldfive.executors.sequential import SequentialExecutor  # noqa: E402
from goldfive.protocols import EventSink  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
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
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """Minimal sink that records emitted events for assertions."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None

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


class RecordingControlChannel:
    """Minimal control-channel stub: records every ``send`` for assertions."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, msg: Any) -> None:
        self.sent.append(msg)


class NoOpPlanner:
    """Planner whose ``refine`` / ``refine_steer`` return a structurally
    identical plan so the #271 no-op-revision detector fires and routes
    through ``_emit_handler_exhausted_escalation`` →
    ``_dispatch_goldfive_pause_control``. Models the live reproduction's
    refine exhaustion path.
    """

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []
        self.refine_steer_calls: list[DriftEvent] = []

    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],  # noqa: ARG002
    ) -> Plan:
        self.refine_calls.append(drift)
        return _structurally_identical_plan(plan)

    async def refine_steer(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],  # noqa: ARG002
        available_agents: Any = None,  # noqa: ARG002
    ) -> Plan:
        self.refine_steer_calls.append(drift)
        return _structurally_identical_plan(plan)


class NoneRefinePlanner:
    """Planner whose ``refine`` / ``refine_steer`` always return ``None``.

    Drives the second handler-exhaustion code path
    (``_handle_drift`` line 3479+, ``_promote_drift_to_steer`` line
    5045+: ``if revised is None``).
    """

    def __init__(self) -> None:
        self.refine_calls: list[DriftEvent] = []
        self.refine_steer_calls: list[DriftEvent] = []

    async def generate(self, **kwargs: Any) -> Plan | None:  # noqa: ARG002
        return None

    async def refine(
        self,
        *,
        plan: Plan,  # noqa: ARG002
        drift: DriftEvent,
        goals: list[Goal],  # noqa: ARG002
    ) -> Plan | None:
        self.refine_calls.append(drift)
        return None

    async def refine_steer(
        self,
        *,
        plan: Plan,  # noqa: ARG002
        drift: DriftEvent,
        goals: list[Goal],  # noqa: ARG002
        available_agents: Any = None,  # noqa: ARG002
    ) -> Plan | None:
        self.refine_steer_calls.append(drift)
        return None


class OverlayStubAdapter:
    """Overlay-mode adapter that runs a configurable passthrough effect.

    Mirrors ``OverlayStubAdapter`` from
    ``tests/test_observation_only_abort_carveout.py`` but adds a hook
    so the passthrough effect can synthesize a drift mid-invocation
    (simulating the live reproduction's "OFF_TOPIC drift fires while
    the coordinator is reasoning" timing).
    """

    def __init__(
        self,
        *,
        passthrough_effect: Callable[
            [str, Session, Any], Awaitable[InvocationResult | None]
        ],
    ) -> None:
        self._passthrough_effect = passthrough_effect
        self.passthrough_calls: list[str] = []

    @property
    def available_agents(self) -> list[str]:
        return ["agent"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    async def invoke(self, task: Task, session: Session) -> InvocationResult:  # noqa: ARG002
        return InvocationResult(task_id=task.id, text="")

    async def invoke_passthrough(
        self,
        user_message: str,
        *,
        session: Session,
        reconciler: Any = None,
        ctx: Any = None,  # noqa: ARG002
    ) -> InvocationResult:
        self.passthrough_calls.append(user_message)
        result = await self._passthrough_effect(user_message, session, reconciler)
        if result is not None:
            return result
        return InvocationResult(task_id="", text="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structurally_identical_plan(prior: Plan) -> Plan:
    """Return a plan with the same structural shape but a bumped
    ``revision_index`` (the validator requires it).

    Triggers ``_plans_structurally_identical`` (steerer.py:7155+) in
    the refine handler so the run takes the no-op-revision path and
    calls ``_emit_handler_exhausted_escalation`` →
    ``_dispatch_goldfive_pause_control``.
    """
    return Plan(
        id=prior.id,
        run_id=prior.run_id,
        goal_ids=list(prior.goal_ids),
        tasks=[
            Task(
                id=t.id,
                title=t.title,
                description=t.description,
                assignee_agent_id=t.assignee_agent_id,
                status=t.status,
            )
            for t in prior.tasks
        ],
        edges=[
            TaskEdge(from_task_id=e.from_task_id, to_task_id=e.to_task_id)
            for e in prior.edges
        ],
        revision_index=prior.revision_index + 1,
    )


def _session(run_id: str = "r1", task_id: str = "t1") -> Session:
    task = Task(
        id=task_id,
        title="Research solar panels",
        description="Find specs",
        assignee_agent_id="agent",
    )
    plan = Plan(
        id="p1",
        run_id=run_id,
        goal_ids=["g1"],
        tasks=[task],
        edges=[],
    )
    return Session(
        run_id=run_id,
        goals=[Goal(id="g1", summary="Publish a memo on solar panels")],
        plan=plan,
        current_task_id=task_id,
    )


def _drift(
    kind: DriftKind = DriftKind.OFF_TOPIC,
    *,
    task_id: str = "t1",
    severity: DriftSeverity = DriftSeverity.WARNING,
    authored_by: str = "goldfive",
) -> DriftEvent:
    """Build a drift event matching the live reproduction's shape.

    Default ``OFF_TOPIC`` + ``WARNING`` + ``authored_by="goldfive"``
    routes through ``_promote_drift_to_steer`` → ``refine_steer`` →
    handler exhaustion (matches session ``4538863f`` exactly). Tests
    can pass other kinds to exercise alternate ladder levels.
    """
    return DriftEvent(
        kind=kind,
        severity=severity,
        detail=f"{kind.value} drift on {task_id}",
        current_task_id=task_id,
        current_agent_id="agent",
        authored_by=authored_by,
    )


def _pause_escalate_messages(channel: RecordingControlChannel) -> list[Any]:
    return [
        m
        for m in channel.sent
        if getattr(m, "kind", None) is ControlKind.GOLDFIVE_PAUSE_ESCALATE
    ]


def _human_intervention_drift_events(sink: ListSink) -> list[Any]:
    """Filter recorded events for HUMAN_INTERVENTION_REQUIRED DriftDetected."""
    from goldfive.pb.goldfive.v1 import types_pb2

    out: list[Any] = []
    for e in sink.events:
        if not hasattr(e, "WhichOneof"):
            continue
        if e.WhichOneof("payload") != "drift_detected":
            continue
        if e.drift_detected.kind == types_pb2.DRIFT_KIND_HUMAN_INTERVENTION_REQUIRED:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# (1) Direct unit test on ``_dispatch_goldfive_pause_control``
# ---------------------------------------------------------------------------


async def test_dispatch_pause_control_observation_only_skips_channel_send(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Under ``observation_only=True``, the dispatch method MUST NOT call
    ``channel.send``. Returns ``False`` (the documented "no bound
    channel / send failure" return), logs the would-be payload at INFO.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _session()
    channel = RecordingControlChannel()
    steerer.bind_control_channel(channel)

    drift = _drift(kind=DriftKind.OFF_TOPIC)
    with caplog.at_level(logging.INFO, logger="goldfive"):
        landed = await steerer._dispatch_goldfive_pause_control(
            drift, session, reason="test exhaustion"
        )

    assert landed is False, (
        "observation_only must short-circuit before channel.send so the "
        "caller can distinguish a skipped dispatch from a successful one "
        "(matches the documented return contract)"
    )
    assert channel.sent == [], (
        "observation_only must NOT enqueue GOLDFIVE_PAUSE_ESCALATE on the "
        f"bound channel; got {channel.sent!r}"
    )
    # Operator-visible INFO log confirms the would-have-dispatched
    # payload made it to logs (so a sysadmin can see WHAT the steerer
    # would have done without the gate).
    assert any(
        "SKIPPING GOLDFIVE_PAUSE_ESCALATE" in rec.message
        and "off_topic" in rec.message.lower()
        for rec in caplog.records
    ), (
        f"observation_only skip must log the would-have-dispatched payload "
        f"at INFO; got records={[r.message for r in caplog.records]!r}"
    )


async def test_dispatch_pause_control_active_steering_sends_channel_message() -> None:
    """Positive control: with ``observation_only=False``, the dispatch
    method enqueues the ControlMessage on the bound channel and
    returns ``True``. Confirms the gate is the only behavioural
    difference — the active-steering path is preserved bit-for-bit.
    """
    cfg = SteeringConfig(observation_only=False)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _session()
    channel = RecordingControlChannel()
    steerer.bind_control_channel(channel)

    drift = _drift(kind=DriftKind.OFF_TOPIC)
    landed = await steerer._dispatch_goldfive_pause_control(
        drift, session, reason="test exhaustion"
    )

    assert landed is True
    pause_msgs = _pause_escalate_messages(channel)
    assert len(pause_msgs) == 1, (
        f"active steering must enqueue exactly one GOLDFIVE_PAUSE_ESCALATE; "
        f"got {channel.sent!r}"
    )
    payload = pause_msgs[0].payload
    assert payload.get("reason") == "test exhaustion"
    assert payload.get("drift_kind") == "off_topic"


# ---------------------------------------------------------------------------
# (2) End-to-end through ``_handle_drift`` / ``_promote_drift_to_steer``
# ---------------------------------------------------------------------------


async def test_observation_only_no_op_refine_does_not_dispatch_pause_but_emits_drift() -> (
    None
):
    """The live reproduction scenario, scoped to the steerer.

    A WARNING OFF_TOPIC drift under ``observation_only=True`` reaches
    ``_promote_drift_to_steer`` → ``refine_steer`` → no-op revision
    detector → ``_emit_handler_exhausted_escalation``. The escalation
    helper:

    * dispatches ``GOLDFIVE_PAUSE_ESCALATE`` via
      ``_dispatch_goldfive_pause_control`` (which our gate now drops);
    * emits a ``HUMAN_INTERVENTION_REQUIRED`` ``DriftDetected`` event
      to sinks (outside the dispatch — operator observability MUST be
      preserved).

    Asserts: NO ControlMessage on the channel,
    HUMAN_INTERVENTION_REQUIRED DriftDetected DOES land on the sink.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _session()
    sink = ListSink()
    planner = NoOpPlanner()
    channel = RecordingControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)

    drift = _drift(kind=DriftKind.OFF_TOPIC)
    await steerer._handle_drift(drift, session)

    # Refine ran (detection + planner unaffected by the gate). OFF_TOPIC
    # + WARNING is on the goldfive-steer promotion path so it lands on
    # ``refine_steer`` rather than ``refine``.
    refine_total = len(planner.refine_calls) + len(planner.refine_steer_calls)
    assert refine_total == 1, (
        f"observation_only must NOT skip the planner refine; got refine="
        f"{len(planner.refine_calls)} refine_steer="
        f"{len(planner.refine_steer_calls)}"
    )

    # Primary assertion: NO GOLDFIVE_PAUSE_ESCALATE on the channel.
    pause_msgs = _pause_escalate_messages(channel)
    assert pause_msgs == [], (
        f"observation_only must NOT enqueue GOLDFIVE_PAUSE_ESCALATE "
        f"(would cancel the in-flight invoke at the executor); "
        f"got {[m.payload for m in pause_msgs]!r}"
    )

    # Observability preserved: HUMAN_INTERVENTION_REQUIRED drift DOES
    # land on the sink.
    hir_events = _human_intervention_drift_events(sink)
    assert len(hir_events) >= 1, (
        "HUMAN_INTERVENTION_REQUIRED DriftDetected must continue to fire "
        "on the sink stream even under observation_only — operators MUST "
        "be able to see the escalation; got sink kinds="
        f"{sink.payload_kinds()!r}"
    )


async def test_observation_only_refine_returns_none_does_not_dispatch_pause() -> None:
    """Second handler-exhaustion path: ``refine`` / ``refine_steer``
    returning ``None``. Same gate, same expected behaviour.
    """
    cfg = SteeringConfig(observation_only=True)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _session()
    sink = ListSink()
    planner = NoneRefinePlanner()
    channel = RecordingControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)

    drift = _drift(kind=DriftKind.OFF_TOPIC)
    await steerer._handle_drift(drift, session)

    pause_msgs = _pause_escalate_messages(channel)
    assert pause_msgs == [], (
        f"observation_only must NOT dispatch GOLDFIVE_PAUSE_ESCALATE on the "
        f"refine-returns-None path; got {[m.payload for m in pause_msgs]!r}"
    )
    hir_events = _human_intervention_drift_events(sink)
    assert len(hir_events) >= 1, (
        "HUMAN_INTERVENTION_REQUIRED must still emit on the refine-None "
        "path; got sink kinds=" + repr(sink.payload_kinds())
    )


async def test_active_steering_no_op_refine_dispatches_pause_escalate() -> None:
    """Regression guard: with ``observation_only=False``, the same
    no-op-refine scenario DOES dispatch GOLDFIVE_PAUSE_ESCALATE.
    Confirms the steering-enabled path is untouched by the carve-out.
    """
    cfg = SteeringConfig(observation_only=False)
    steerer = DefaultSteerer(steering_config=cfg)
    session = _session()
    sink = ListSink()
    planner = NoOpPlanner()
    channel = RecordingControlChannel()
    steerer.bind(sinks=[sink], planner=planner)
    steerer.bind_control_channel(channel)

    drift = _drift(kind=DriftKind.OFF_TOPIC)
    await steerer._handle_drift(drift, session)

    pause_msgs = _pause_escalate_messages(channel)
    assert len(pause_msgs) == 1, (
        f"active steering must enqueue GOLDFIVE_PAUSE_ESCALATE for the "
        f"no-op-refine path; got {channel.sent!r}"
    )
    # The HUMAN_INTERVENTION_REQUIRED drift is ALSO emitted under
    # active steering (it's outside the dispatch in both modes).
    hir_events = _human_intervention_drift_events(sink)
    assert len(hir_events) >= 1


# ---------------------------------------------------------------------------
# (3) Overlay-level: executor must NOT end the overlay turn under
#     observation_only even if a GOLDFIVE_PAUSE_ESCALATE leaks through
#     (defense-in-depth).
# ---------------------------------------------------------------------------


class _StubSteererForOverlay:
    """Bare-bones Steerer for overlay tests; carries the
    ``_observation_only`` attribute the executor's defense-in-depth gate
    reads. Avoids the full DefaultSteerer to keep these overlay tests
    fast and avoid wiring planner / refine plumbing — the executor's
    gate is read structurally from ``steerer._observation_only``.
    """

    def __init__(self, *, observation_only: bool = False) -> None:
        self._observation_only = observation_only
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:  # noqa: ARG002
        self.observed.append(event)

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:  # noqa: ARG002
        self.observed.append(drift)

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",  # noqa: ARG002
        session: Session,
        cancel_reason: str = "",  # noqa: ARG002
    ) -> None:
        if session.plan is None:
            return
        if not any(t.id == task_id for t in session.plan.tasks):
            return
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )

        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:  # noqa: ARG002
        return None


class _LeakyControlChannel:
    """Control channel stub that injects ONE pre-canned GOLDFIVE_PAUSE_ESCALATE
    message during the overlay invocation, then blocks forever on
    ``receive()``. Simulates a future code path that bypasses the
    steerer's primary gate so the executor's defense-in-depth gate
    fires.
    """

    def __init__(self, *, payload_reason: str = "leaked") -> None:
        from goldfive.control import ControlKind, ControlMessage

        self._msg = ControlMessage(
            kind=ControlKind.GOLDFIVE_PAUSE_ESCALATE,
            payload={"reason": payload_reason, "drift_kind": "off_topic"},
        )
        self._delivered = False
        self._sends: list[Any] = []
        self._block = asyncio.Event()
        # Latch: only deliver the leak once so the executor moves on.

    async def send(self, msg: Any) -> None:
        self._sends.append(msg)

    async def receive(self) -> Any:
        if not self._delivered:
            self._delivered = True
            return self._msg
        # Block until the test ends — the executor sees no further
        # control messages and falls through on its own.
        await self._block.wait()
        return None

    async def ack(self, ack: Any) -> None:
        return None


async def test_overlay_observation_only_drops_leaked_pause_escalate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defense-in-depth: if a ``GOLDFIVE_PAUSE_ESCALATE`` ever leaks onto
    the channel under ``observation_only`` (e.g. a custom steerer
    subclass bypassing the primary gate), the executor MUST NOT cancel
    the in-flight invoke or end the overlay turn.

    Asserts:

    * the passthrough invocation completes normally (success=True);
    * the in-flight passthrough was NOT cancelled mid-flight;
    * the overlay logs the "would have paused" carve-out at INFO.
    """
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="t1", assignee_agent_id="agent")],
        edges=[],
    )
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="g")], plan=plan)
    steerer = _StubSteererForOverlay(observation_only=True)
    sink = ListSink()
    channel = _LeakyControlChannel()

    cancelled_mid_flight = {"v": False}

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Give the executor's control-recv task a chance to pick up the
        # leaked ControlMessage. Under the carve-out, the executor must
        # NOT cancel us mid-await — we should complete normally.
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled_mid_flight["v"] = True
            raise
        # Complete the only task so the executor reaps it and ends the
        # overlay turn naturally.
        await steerer.transition("t1", TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id="", text="ok")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=True)

    with caplog.at_level(logging.INFO, logger="goldfive"):
        outcome = await executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=None,
            sinks=[sink],
            user_input="do it",
            control=channel,  # type: ignore[arg-type]
        )

    assert not cancelled_mid_flight["v"], (
        "observation_only must NOT cancel the in-flight passthrough when a "
        "GOLDFIVE_PAUSE_ESCALATE arrives on the channel"
    )
    assert outcome.success is True, (
        f"overlay under observation_only must complete normally when a "
        f"GOLDFIVE_PAUSE_ESCALATE is dropped; got success={outcome.success} "
        f"reason={outcome.reason!r}"
    )
    # The defense-in-depth log line must fire so an operator can grep
    # for the dropped escalation.
    assert any(
        "observation_only=True" in rec.message
        and "GOLDFIVE_PAUSE_ESCALATE" in rec.message
        for rec in caplog.records
    ), (
        f"executor must log the dropped GOLDFIVE_PAUSE_ESCALATE at INFO; "
        f"got records={[r.message for r in caplog.records]!r}"
    )


async def test_overlay_active_steering_honours_leaked_pause_escalate() -> None:
    """Regression guard: with ``observation_only=False``, a
    GOLDFIVE_PAUSE_ESCALATE arriving on the channel DOES cancel the
    in-flight invoke and ends the overlay turn (preserves Phase-2 of
    the path-duality fix).
    """
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="t1", assignee_agent_id="agent")],
        edges=[],
    )
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="g")], plan=plan)
    steerer = _StubSteererForOverlay(observation_only=False)
    sink = ListSink()
    channel = _LeakyControlChannel(payload_reason="active-pause")

    cancelled_mid_flight = {"v": False}

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        try:
            # Long sleep so the executor's control-recv definitely
            # wins the race and the cancel path fires.
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled_mid_flight["v"] = True
            raise
        return InvocationResult(task_id="", text="ok")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=True)

    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=None,
        sinks=[sink],
        user_input="do it",
        control=channel,  # type: ignore[arg-type]
    )

    assert cancelled_mid_flight["v"], (
        "active steering must cancel the in-flight invoke when "
        "GOLDFIVE_PAUSE_ESCALATE arrives on the channel"
    )
    # The overlay returns success=True (the pause is the "operator
    # decides next" state — the overlay turn ended cleanly).
    assert outcome.success is True
    assert outcome.reason and "goldfive_pause_escalate" in outcome.reason, (
        f"overlay must report goldfive_pause_escalate as the end-of-turn "
        f"reason under active steering; got reason={outcome.reason!r}"
    )

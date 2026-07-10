"""Tests for the pause-escalation deadline (escalation-ladder terminus).

The intervention ladder's top used to be an infinite hang in unattended
deployments: the executors' pre-task / pre-stage pause loops awaited
``ControlChannel.receive()`` with no bound, and Level 5 (TERMINATE)
silently degraded to another PAUSE_ESCALATE. These tests pin the fix:

* ``SteeringConfig.pause_escalate_deadline_s`` bounds the pause wait;
  expiry aborts the run (non-terminal tasks CANCELLED, ``RunAborted``
  carrying the escalation lineage) — sequential AND parallel executors.
* ``None`` (the default) preserves the block-forever behaviour.
* TERMINATE dispatches a pause that ALWAYS carries a deadline (the
  configured value, or ``DEFAULT_TERMINATE_PAUSE_DEADLINE_S``).
* Pause escalation stays gated under ``observation_only=True``.

Every blocking assertion is wrapped in an asyncio timeout so a
regression fails fast instead of hanging CI.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.control import (  # noqa: E402
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.drift_observer import DEFAULT_TERMINATE_PAUSE_DEADLINE_S  # noqa: E402
from goldfive.executors import ParallelDAGExecutor, SequentialExecutor  # noqa: E402
from goldfive.executors._control import _ControlCancelled, pause_deadline_s  # noqa: E402
from goldfive.results import InvocationResult  # noqa: E402
from goldfive.steerer import DefaultSteerer, InterventionLevel  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    RefineOutcome,
    Session,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs (kept local; the suite avoids shared mutable state across files)
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return

    def payload_kinds(self) -> list[str]:
        return [e.WhichOneof("payload") or "" for e in self.events if hasattr(e, "WhichOneof")]

    def run_aborted_reasons(self) -> list[str]:
        return [
            e.run_aborted.reason
            for e in self.events
            if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "run_aborted"
        ]


class StubSteerer:
    """Minimal Steerer: applies transitions in place and records them."""

    def __init__(self) -> None:
        self.transitions: list[tuple[str, TaskStatus, str]] = []

    class _Drift:
        async def observe(self, event: Any, session: Session) -> None:
            return

        def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:
            return None

    drift = _Drift()

    def bind(self, *, sinks: list[Any], planner: Any) -> None:
        return

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
        cancel_reason: str = "",
    ) -> None:
        self.transitions.append((task_id, to, cancel_reason or detail))
        if session.plan is None or not any(t.id == task_id for t in session.plan.tasks):
            return
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )

        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))


class StubPlanner:
    async def generate(self, **_kw: Any) -> Plan | None:
        return None

    async def refine(self, **_kw: Any) -> Plan | None:
        return None


class StubAdapter:
    def __init__(
        self, on_invoke: Callable[[Task, Session], Awaitable[InvocationResult]] | None = None
    ) -> None:
        self._on_invoke = on_invoke
        self.invocations: list[str] = []

    async def register_reporting_tools(self, tools: list[Any]) -> None:
        return None

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        self.invocations.append(task.id)
        if self._on_invoke is not None:
            return await self._on_invoke(task, session)
        return InvocationResult(task_id=task.id, text="done")


def _plan(ids: list[str], run_id: str = "run-dl") -> Plan:
    return Plan(
        id="p0",
        run_id=run_id,
        goal_ids=[],
        tasks=[Task(id=t, title=f"Task {t}") for t in ids],
        edges=[],
    )


def _pause_message(deadline_s: float | None, **extra: Any) -> ControlMessage:
    payload: dict[str, Any] = {
        "reason": "test escalation",
        "drift_kind": DriftKind.INTENT_DIVERGENCE.value,
        "ladder_level": "pause_escalate",
        **extra,
    }
    if deadline_s is not None:
        payload["deadline_s"] = deadline_s
    return ControlMessage(kind=ControlKind.GOLDFIVE_PAUSE_ESCALATE, payload=payload)


# ---------------------------------------------------------------------------
# Sequential executor: deadline expiry aborts the run
# ---------------------------------------------------------------------------


async def test_sequential_pause_deadline_expiry_aborts_run() -> None:
    plan = _plan(["t0", "t1"])
    session = Session(run_id="run-dl")
    steerer = StubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()
    await channel.send(_pause_message(0.1, ladder_level="terminate"))

    executor = SequentialExecutor(max_task_invocations=5)
    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=StubAdapter(),
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
        ),
        timeout=5.0,
    )

    assert outcome.success is False
    assert "pause escalation deadline expired" in (outcome.reason or "")
    # Escalation lineage travels on the RunAborted reason.
    reasons = sink.run_aborted_reasons()
    assert len(reasons) == 1
    assert f"drift_kind={DriftKind.INTENT_DIVERGENCE.value}" in reasons[0]
    assert "ladder_level=terminate" in reasons[0]
    assert sink.payload_kinds()[-1] == "run_aborted"
    # Every task was cancelled; none ever ran.
    assert session.plan is not None
    assert all(t.status is TaskStatus.CANCELLED for t in session.plan.tasks)
    assert all(
        reason.startswith("run_aborted:pause_escalate_deadline:")
        for _tid, to, reason in steerer.transitions
        if to is TaskStatus.CANCELLED
    )


async def test_sequential_pause_without_deadline_still_blocks() -> None:
    """Default (no ``deadline_s`` payload) preserves the blocking wait."""
    plan = _plan(["t0"])
    session = Session(run_id="run-dl")
    steerer = StubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()
    await channel.send(_pause_message(None))

    executor = SequentialExecutor(max_task_invocations=5)
    run_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=StubAdapter(),
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
        )
    )
    await asyncio.sleep(0.2)
    assert not run_task.done(), "no-deadline pause must keep blocking"

    # Bounded by a test-side control message, not a deadline.
    await channel.send(ControlMessage(kind=ControlKind.RESUME))
    outcome = await asyncio.wait_for(run_task, timeout=5.0)
    assert outcome.success is True
    assert steerer.transitions == [] or all(
        to is not TaskStatus.CANCELLED for _tid, to, _r in steerer.transitions
    )


async def test_sequential_deadline_adopted_while_already_paused() -> None:
    """A deadline-carrying escalation arriving mid-pause bounds the wait.

    This is how the ladder's TERMINATE row lands on an executor already
    parked on an unbounded Level-4 pause.
    """
    session = Session(run_id="run-dl")
    session.plan = _plan(["t0"], run_id="run-dl")
    steerer = StubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()
    await channel.send(_pause_message(None))

    executor = SequentialExecutor()
    wait_task = asyncio.create_task(
        executor._apply_pre_task_controls(
            control=channel,
            session=session,
            steerer=steerer,
            sinks=[sink],
        )
    )
    await asyncio.sleep(0.05)
    assert not wait_task.done(), "executor should be parked in the pause loop"

    await channel.send(_pause_message(0.1, ladder_level="terminate"))
    with pytest.raises(_ControlCancelled, match="deadline expired"):
        await asyncio.wait_for(wait_task, timeout=5.0)
    assert session.plan.tasks[0].status is TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Parallel executor: same deadline semantics on the pre-stage wait
# ---------------------------------------------------------------------------


async def test_parallel_pause_deadline_expiry_aborts_run() -> None:
    plan = _plan(["t0", "t1"])
    session = Session(run_id="run-dl")
    steerer = StubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()
    await channel.send(_pause_message(0.1))

    executor = ParallelDAGExecutor()
    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=StubAdapter(),
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
        ),
        timeout=5.0,
    )

    assert outcome.success is False
    assert "pause escalation deadline expired" in (outcome.reason or "")
    assert sink.payload_kinds()[-1] == "run_aborted"
    assert session.plan is not None
    assert all(t.status is TaskStatus.CANCELLED for t in session.plan.tasks)


# ---------------------------------------------------------------------------
# Steerer: TERMINATE dispatches pause-with-deadline
# ---------------------------------------------------------------------------


def _bound_steerer(
    steering_config: SteeringConfig | None = None,
) -> tuple[DefaultSteerer, Session, RecordingSink, ControlChannel]:
    # Explicit active mode unless the caller supplies its own config:
    # the pause dispatch under test is suppressed under the shipped
    # observation-only default.
    if steering_config is None:
        steering_config = SteeringConfig(observation_only=False)
    steerer = DefaultSteerer(goldfive_steer_threshold="off", steering_config=steering_config)
    session = Session(run_id="terminate-test", current_task_id="t1")
    session.plan = Plan(
        id="p1",
        run_id="terminate-test",
        goal_ids=[],
        tasks=[Task(id="t1", title="work", status=TaskStatus.RUNNING)],
        edges=[],
    )
    sink = RecordingSink()
    channel = ControlChannel()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    steerer.bind_control_channel(channel)
    return steerer, session, sink, channel


def _drain_goldfive_pause(channel: ControlChannel) -> list[ControlMessage]:
    drained: list[ControlMessage] = []
    inbox = channel._inbox  # noqa: SLF001 — test inspection
    while not inbox.empty():
        msg = inbox.get_nowait()
        if msg.kind is ControlKind.GOLDFIVE_PAUSE_ESCALATE:
            drained.append(msg)
    return drained


def _terminate_drift(session: Session) -> DriftEvent:
    """A HUMAN_INTERVENTION_REQUIRED repeat: the ladder's TERMINATE row."""
    kind = DriftKind.HUMAN_INTERVENTION_REQUIRED
    session.refine_outcomes[(kind.value, "t1")] = RefineOutcome(
        state="failed", fail_count=DefaultSteerer.REFINE_FAILURE_THRESHOLD
    )
    return DriftEvent(
        kind=kind,
        severity=DriftSeverity.CRITICAL,
        detail="operator never came",
        current_task_id="t1",
    )


async def test_terminate_row_dispatches_pause_with_builtin_deadline() -> None:
    steerer, session, _sink, channel = _bound_steerer()
    drift = _terminate_drift(session)
    assert (
        steerer.drift._ladder_level_for(
            drift.kind, drift.severity, DefaultSteerer.REFINE_FAILURE_THRESHOLD
        )
        is InterventionLevel.TERMINATE
    )
    await steerer.drift.handle_drift(drift, session)

    msgs = _drain_goldfive_pause(channel)
    assert len(msgs) == 1
    assert msgs[0].payload["ladder_level"] == "terminate"
    assert pause_deadline_s(msgs[0]) == DEFAULT_TERMINATE_PAUSE_DEADLINE_S


async def test_terminate_row_respects_configured_deadline() -> None:
    steerer, session, _sink, channel = _bound_steerer(
        SteeringConfig(observation_only=False, pause_escalate_deadline_s=1.5)
    )
    await steerer.drift.handle_drift(_terminate_drift(session), session)

    msgs = _drain_goldfive_pause(channel)
    assert len(msgs) == 1
    assert msgs[0].payload["ladder_level"] == "terminate"
    assert pause_deadline_s(msgs[0]) == 1.5


async def test_pause_escalate_row_carries_configured_deadline() -> None:
    """Level 4 attaches the config deadline when set — and only then."""
    steerer, session, _sink, channel = _bound_steerer(
        SteeringConfig(observation_only=False, pause_escalate_deadline_s=2.0)
    )
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="critical intent drift",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    msgs = _drain_goldfive_pause(channel)
    assert len(msgs) == 1
    assert msgs[0].payload["ladder_level"] == "pause_escalate"
    assert pause_deadline_s(msgs[0]) == 2.0


async def test_pause_escalate_row_has_no_deadline_by_default() -> None:
    """Default (deadline unset) is behaviour-preserving: no ``deadline_s``."""
    steerer, session, _sink, channel = _bound_steerer()
    drift = DriftEvent(
        kind=DriftKind.INTENT_DIVERGENCE,
        severity=DriftSeverity.CRITICAL,
        detail="critical intent drift",
        current_task_id="t1",
    )
    await steerer.drift.handle_drift(drift, session)
    msgs = _drain_goldfive_pause(channel)
    assert len(msgs) == 1
    assert "deadline_s" not in msgs[0].payload
    assert pause_deadline_s(msgs[0]) is None


async def test_terminate_row_stays_gated_under_observation_only() -> None:
    """Regression: pause escalation (TERMINATE included) never dispatches
    a control message under ``observation_only=True`` — the production
    default. The HUMAN_INTERVENTION_REQUIRED drift on the sink stream
    remains the durable signal."""
    steerer, session, sink, channel = _bound_steerer(
        SteeringConfig(observation_only=True, pause_escalate_deadline_s=1.0)
    )
    await steerer.drift.handle_drift(_terminate_drift(session), session)

    assert _drain_goldfive_pause(channel) == []
    assert "drift_detected" in sink.payload_kinds()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_pause_escalate_deadline_default_is_none() -> None:
    assert SteeringConfig().pause_escalate_deadline_s is None


def test_pause_escalate_deadline_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S", "12.5")
    assert SteeringConfig.from_env().pause_escalate_deadline_s == 12.5


def test_pause_escalate_deadline_env_rejects_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S", "soon")
    assert SteeringConfig.from_env().pause_escalate_deadline_s is None
    monkeypatch.setenv("GOLDFIVE_STEER_PAUSE_ESCALATE_DEADLINE_S", "-3")
    assert SteeringConfig.from_env().pause_escalate_deadline_s is None

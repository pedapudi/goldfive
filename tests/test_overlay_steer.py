"""Overlay-mode STEER handling (goldfive#149).

The #141 overlay refactor wired ``_run_overlay`` to terminate on
``cancelled`` / ``adapter_error`` but silently dropped ``kind == "steer"``
returns from :meth:`_invoke_passthrough_with_control`. STEER control
messages reached the executor, cancelled the in-flight passthrough,
and then the overlay fell straight through to the missed-task
follow-up loop — driving the ORIGINAL pre-steer tasks instead of
feeding the STEER through the steerer so USER_STEER drift could
cascade-cancel + refine.

These tests exercise the control-channel-driven STEER path end-to-end
through the overlay loop, asserting:

* ``steerer.observe`` is called with the STEER ``ControlMessage``.
* The reconciler's task-claim state is reset for the revised plan.
* ``invoke_passthrough`` is re-invoked with the steer body as user
  input.
* Successive STEERs each trigger their own refine + re-invoke.
* CANCEL remains terminal (no restart).
* :meth:`PlanReconciler.reset_for_new_plan` clears plan-scoped claim
  state while preserving historical observation lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from goldfive.control import (
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.executors.sequential import SequentialExecutor
from goldfive.protocols import EventSink
from goldfive.reconciler import PlanReconciler
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
# Stubs mirroring tests/test_sequential_executor_overlay.py shapes.
# ---------------------------------------------------------------------------


class RecordingSink:
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


class SteerAwareStubSteerer:
    """Steerer stub that reacts to STEER ControlMessage observations.

    Captures the full observe stream, and when a STEER arrives: records
    the fact, cascade-cancels every non-terminal task in the current
    plan (mirroring DefaultSteerer's USER_STEER handling), bumps
    ``drift_events`` with a synthetic USER_STEER drift, and swaps
    ``session.plan`` with a pre-configured ``refined_plan`` if one was
    provided. ``planner_refine_calls`` captures the (plan, drift)
    tuples the fake refine saw so tests can assert refine was invoked.
    """

    def __init__(self, *, refined_plans: list[Plan] | None = None) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []
        self.drift_events: list[DriftEvent] = []
        self.cascade_cancelled: list[str] = []
        self.planner_refine_calls: list[tuple[Plan, DriftEvent]] = []
        # One refined plan per STEER received, consumed in order.
        self._refined_plans: list[Plan] = list(refined_plans or [])

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:
        self.observed.append(event)
        kind_val = str(
            getattr(getattr(event, "kind", None), "value", getattr(event, "kind", ""))
        ).upper()
        if kind_val != "STEER":
            return
        # Simulate USER_STEER drift + cascade-cancel + refine.
        drift = DriftEvent(
            kind=DriftKind.USER_STEER,
            severity=DriftSeverity.WARNING,
            detail=str((event.payload or {}).get("note", "")),
            current_task_id=session.current_task_id,
        )
        self.drift_events.append(drift)
        if session.plan is not None:
            for t in session.plan.tasks:
                if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    t.status = TaskStatus.CANCELLED
                    self.cascade_cancelled.append(t.id)
        # Emulate planner.refine being called by the intervention ladder.
        if session.plan is not None:
            self.planner_refine_calls.append((session.plan, drift))
        if self._refined_plans:
            session.plan = self._refined_plans.pop(0)

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:  # noqa: ARG002
        self.drift_events.append(drift)

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
        for t in session.plan.tasks:
            if t.id == task_id:
                t.status = to
                return

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:  # noqa: ARG002
        return None


class StubPlanner:
    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class OverlayStubAdapter:
    """Adapter mirroring the shape in test_sequential_executor_overlay.py."""

    def __init__(
        self,
        *,
        passthrough_effect: Callable[[str, Session, Any], Awaitable[InvocationResult | None]]
        | None = None,
        follow_up_effect: Callable[[Task, Session], Awaitable[InvocationResult | None]]
        | None = None,
    ) -> None:
        self._passthrough_effect = passthrough_effect
        self._follow_up_effect = follow_up_effect
        self.passthrough_calls: list[str] = []
        self.follow_up_calls: list[str] = []

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
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
        if self._passthrough_effect is not None:
            result = await self._passthrough_effect(user_message, session, reconciler)
            if result is not None:
                return result
        return InvocationResult(task_id="", text="")

    async def invoke_follow_up(self, task: Task, session: Session) -> InvocationResult:
        self.follow_up_calls.append(task.id)
        if self._follow_up_effect is not None:
            result = await self._follow_up_effect(task, session)
            if result is not None:
                return result
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")


# ---------------------------------------------------------------------------
# Plan helpers.
# ---------------------------------------------------------------------------


def _three_task_plan(ids: tuple[str, str, str] = ("t0", "t1", "t2")) -> Plan:
    a, b, c = ids
    return Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id=a, title=f"task {a}", assignee_agent_id=f"agent_{a}"),
            Task(id=b, title=f"task {b}", assignee_agent_id=f"agent_{b}"),
            Task(id=c, title=f"task {c}", assignee_agent_id=f"agent_{c}"),
        ],
        edges=[
            TaskEdge(from_task_id=a, to_task_id=b),
            TaskEdge(from_task_id=b, to_task_id=c),
        ],
    )


def _revised_plan(run_id: str = "r1", revision_index: int = 1) -> Plan:
    return Plan(
        id=f"p{revision_index}",
        run_id=run_id,
        goal_ids=[],
        tasks=[
            Task(id="r0", title="revised 0", assignee_agent_id="agent_r0"),
            Task(id="r1", title="revised 1", assignee_agent_id="agent_r1"),
        ],
        edges=[TaskEdge(from_task_id="r0", to_task_id="r1")],
        revision_reason="user steer",
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=revision_index,
    )


# ---------------------------------------------------------------------------
# 1. STEER fires USER_STEER drift + cascade-cancel + refine + plan-swap.
# ---------------------------------------------------------------------------


async def test_overlay_steer_triggers_user_steer_drift_and_refine() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    refined = _revised_plan(run_id="r1")
    steerer = SteerAwareStubSteerer(refined_plans=[refined])
    sink = RecordingSink()
    channel = ControlChannel()

    first_passthrough_started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        first_passthrough_started.set()
        # On the first call, block until cancelled so the STEER can
        # fire mid-invocation. On subsequent calls (after the refine),
        # complete immediately so the run can progress.
        if len(adapter.passthrough_calls) == 1:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        # Second invocation: simulate tree completing every revised
        # task via the reconciler.
        if session.plan is not None:
            for t in list(session.plan.tasks):
                await reconciler.on_before_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
                await reconciler.on_after_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
        return InvocationResult(task_id="", text="revised plan done")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="original goal",
        )
    )

    await first_passthrough_started.wait()
    await channel.send(
        ControlMessage(
            kind=ControlKind.STEER,
            payload={"note": "actually focus on batteries instead"},
        )
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    # Steerer observed the STEER ControlMessage.
    steer_observations = [
        o
        for o in steerer.observed
        if str(getattr(getattr(o, "kind", None), "value", "")).upper() == "STEER"
    ]
    assert len(steer_observations) == 1, (
        f"expected exactly 1 STEER observation, got {len(steer_observations)}"
    )

    # USER_STEER drift fired.
    assert any(
        d.kind is DriftKind.USER_STEER for d in steerer.drift_events
    ), "USER_STEER drift did not fire"

    # Cascade-cancel happened on the pre-steer plan's live tasks.
    assert set(steerer.cascade_cancelled) >= {"t0"}, (
        "expected cascade-cancel to touch t0 (at minimum)"
    )

    # planner.refine was invoked (via the steerer's fake refine hook).
    assert len(steerer.planner_refine_calls) == 1
    assert steerer.planner_refine_calls[0][1].kind is DriftKind.USER_STEER

    # session.plan is now the revised plan.
    assert session.plan is refined

    # The run completed successfully on the revised plan.
    assert outcome.success is True
    # Revised tasks are terminal.
    for t in refined.tasks:
        assert t.status in (
            TaskStatus.COMPLETED,
            TaskStatus.NOT_NEEDED,
        ), f"revised task {t.id} ended in {t.status}"


# ---------------------------------------------------------------------------
# 2. STEER restarts the invocation with the steer body as user input.
# ---------------------------------------------------------------------------


async def test_overlay_steer_restarts_invocation_with_steer_body() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    refined = _revised_plan()
    steerer = SteerAwareStubSteerer(refined_plans=[refined])
    sink = RecordingSink()
    channel = ControlChannel()

    first_started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        if len(adapter.passthrough_calls) == 1:
            first_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        # Second call: complete whatever's pending in the revised plan.
        if session.plan is not None:
            for t in list(session.plan.tasks):
                await reconciler.on_before_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
                await reconciler.on_after_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
        return InvocationResult(task_id="", text="ok")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    steer_body = "switch to reviewing the draft instead"

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="original user input",
        )
    )
    await first_started.wait()
    await channel.send(
        ControlMessage(kind=ControlKind.STEER, payload={"note": steer_body})
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is True
    # Two invocations total: pre-steer (cancelled) and post-steer.
    assert len(adapter.passthrough_calls) == 2, (
        f"expected 2 invoke_passthrough calls, got {adapter.passthrough_calls}"
    )
    assert adapter.passthrough_calls[0] == "original user input"
    # goldfive#152: the post-steer user input is now wrapped in a
    # goldfive-authored override header instead of handed raw. The
    # steer body still appears verbatim inside the framed message.
    second = adapter.passthrough_calls[1]
    assert second.startswith("[USER STEERING CONTROL"), second
    assert steer_body in second


# ---------------------------------------------------------------------------
# 3. STEER body fallback: missing/empty body reuses the previous input.
# ---------------------------------------------------------------------------


async def test_overlay_steer_empty_body_falls_back_to_previous_input() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    refined = _revised_plan()
    steerer = SteerAwareStubSteerer(refined_plans=[refined])
    sink = RecordingSink()
    channel = ControlChannel()

    first_started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        if len(adapter.passthrough_calls) == 1:
            first_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        if session.plan is not None:
            for t in list(session.plan.tasks):
                await reconciler.on_before_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
                await reconciler.on_after_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
        return InvocationResult(task_id="", text="ok")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="the original",
        )
    )
    await first_started.wait()
    # STEER with empty body — should fall back to previous user input.
    await channel.send(ControlMessage(kind=ControlKind.STEER, payload={"note": ""}))

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is True
    # goldfive#152: empty-body steer still wraps the fallback in the
    # override header so the LLM sees the override semantics even
    # when the operator's note is empty.
    assert adapter.passthrough_calls[0] == "the original"
    second = adapter.passthrough_calls[1]
    assert second.startswith("[USER STEERING CONTROL"), second
    assert "the original" in second


# ---------------------------------------------------------------------------
# 4. CANCEL remains terminal (no restart).
# ---------------------------------------------------------------------------


async def test_overlay_cancel_distinct_from_steer() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = SteerAwareStubSteerer()
    sink = RecordingSink()
    channel = ControlChannel()

    started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        return InvocationResult(task_id="", text="unreachable")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="go",
        )
    )
    await started.wait()
    await channel.send(
        ControlMessage(kind=ControlKind.CANCEL, payload={"reason": "user aborted"})
    )

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is False
    assert "aborted" in (outcome.reason or "")
    # No restart — only the original invocation.
    assert len(adapter.passthrough_calls) == 1
    # RunAborted was the last event.
    assert sink.payload_kinds()[-1] == "run_aborted"
    # Steerer did NOT observe a STEER (CANCEL is a different kind).
    assert not any(
        str(getattr(getattr(o, "kind", None), "value", "")).upper() == "STEER"
        for o in steerer.observed
    )


# ---------------------------------------------------------------------------
# 5. Two consecutive STEERs each trigger their own refine + re-invoke.
# ---------------------------------------------------------------------------


async def test_overlay_steer_after_steer() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    refined1 = _revised_plan(revision_index=1)
    # Second revision is a further-reshaped plan.
    refined2 = Plan(
        id="p2",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="rr0", title="refined-again 0", assignee_agent_id="agent_rr0"),
        ],
        edges=[],
        revision_reason="user steer v2",
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=2,
    )
    steerer = SteerAwareStubSteerer(refined_plans=[refined1, refined2])
    sink = RecordingSink()
    channel = ControlChannel()

    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult | None:
        call_idx = len(adapter.passthrough_calls)
        if call_idx == 1:
            first_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        if call_idx == 2:
            second_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        # Third invocation: complete the final revised plan.
        if session.plan is not None:
            for t in list(session.plan.tasks):
                await reconciler.on_before_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
                await reconciler.on_after_agent(
                    agent_name=t.assignee_agent_id, invocation_id=f"inv_{t.id}"
                )
        return InvocationResult(task_id="", text="final")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=StubPlanner(),
            sinks=[sink],
            control=channel,
            user_input="v0",
        )
    )
    await first_started.wait()
    await channel.send(
        ControlMessage(kind=ControlKind.STEER, payload={"note": "pivot to v1"})
    )
    await second_started.wait()
    await channel.send(
        ControlMessage(kind=ControlKind.STEER, payload={"note": "now pivot to v2"})
    )
    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is True
    # Three invocations: v0 (steered), v1 (steered), v2 (completed).
    # goldfive#152: post-steer inputs wrapped in the override header.
    assert adapter.passthrough_calls[0] == "v0"
    assert adapter.passthrough_calls[1].startswith("[USER STEERING CONTROL")
    assert "pivot to v1" in adapter.passthrough_calls[1]
    assert adapter.passthrough_calls[2].startswith("[USER STEERING CONTROL")
    assert "now pivot to v2" in adapter.passthrough_calls[2]
    # Two USER_STEER drift events, two refine calls.
    user_steer_drifts = [
        d for d in steerer.drift_events if d.kind is DriftKind.USER_STEER
    ]
    assert len(user_steer_drifts) == 2
    assert len(steerer.planner_refine_calls) == 2
    # Final plan is the v2 revision.
    assert session.plan is refined2


# ---------------------------------------------------------------------------
# 6. PlanReconciler.reset_for_new_plan clears claim state.
# ---------------------------------------------------------------------------


async def test_reconciler_reset_for_new_plan_clears_plan_scoped_state() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1", plan=plan)

    class _NoopSteerer:
        async def transition(
            self,
            task_id: str,  # noqa: ARG002
            to: TaskStatus,  # noqa: ARG002
            *,
            detail: str = "",  # noqa: ARG002
            session: Session,  # noqa: ARG002
            cancel_reason: str = "",  # noqa: ARG002
        ) -> None:
            return None

        async def observe(self, event: Any, session: Session) -> None:  # noqa: ARG002
            return None

        async def _handle_drift(
            self, drift: DriftEvent, session: Session  # noqa: ARG002
        ) -> None:
            return None

        def detect_drift(
            self, event: Any, session: Session  # noqa: ARG002
        ) -> DriftEvent | None:
            return None

        def bind(self, **kw: Any) -> None:
            return None

    reconciler = PlanReconciler(
        session=session,
        steerer=_NoopSteerer(),
        host_agent_name="",
    )

    # Simulate an observation pass on the original plan.
    await reconciler.on_before_agent(agent_name="agent_t0", invocation_id="inv_t0")
    # A before without a matching plan assignee should populate
    # off_plan_seen; exercise that too.
    await reconciler.on_before_agent(
        agent_name="not_on_plan", invocation_id="inv_off"
    )

    # Historical lists have entries, plan-scoped state is non-empty.
    assert reconciler.observed_agents, "observed_agents should accumulate"
    assert "t0" in reconciler._observed_task_ids
    assert "agent_t0" in reconciler._running_by_agent
    assert "not_on_plan" in reconciler._off_plan_seen
    # divergence_events populated by the off-plan observation.
    assert reconciler.divergence_events, "divergence_events should be recorded"

    historical_agents = list(reconciler.observed_agents)
    historical_divergence = list(reconciler.divergence_events)

    revised = _revised_plan(run_id="r1")
    session.plan = revised
    reconciler.reset_for_new_plan(revised)

    # Plan-scoped state cleared.
    assert reconciler._observed_task_ids == set()
    assert reconciler._running_by_agent == {}
    assert reconciler._off_plan_seen == set()
    # Historical observation lists preserved (see goldfive#144-style replay).
    assert reconciler.observed_agents == historical_agents
    assert reconciler.divergence_events == historical_divergence

    # After reset, missed-task accounting operates on the new plan shape.
    missed = reconciler.get_missed_tasks()
    missed_ids = {t.id for t in missed}
    assert missed_ids == {"r0", "r1"}, (
        "reset should make the revised plan's PENDING tasks appear missed again"
    )

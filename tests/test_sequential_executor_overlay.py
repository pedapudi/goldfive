"""Unit tests for :class:`SequentialExecutor`'s overlay-mode branch.

The overlay path (goldfive#141, refined in goldfive#163) flips
execution from "loop over plan tasks calling adapter.invoke(task)" to:

1. ONE call to ``adapter.invoke_passthrough(user_input, reconciler=...)``.
2. When the invocation ends, mark any PENDING tasks as ``NOT_NEEDED``.

goldfive#163 specifically **removed** the old "fire
``adapter.invoke_follow_up(task)`` for each missed task" loop — it
was amplifying slow flow-prompted coordinators into 4-5x rework
loops. The tests below encode the new contract: no follow-up
dispatch, PENDING → NOT_NEEDED at invocation end, STEER and CANCEL
paths preserved.

These tests use stub adapters — no ADK, no LLM, no network.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from goldfive.executors.sequential import SequentialExecutor
from goldfive.protocols import EventSink
from goldfive.results import InvocationResult
from goldfive.types import (
    DriftEvent,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs (mirror tests/test_sequential_executor.py shapes plus overlay hooks).
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


class StubSteerer:
    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []
        self.transitions: list[tuple[str, TaskStatus]] = []

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
        self.transitions.append((task_id, to))
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
    """Adapter that supports invoke_passthrough + invoke_follow_up.

    ``passthrough_effect`` is called on ``invoke_passthrough`` with
    the (user_input, session, reconciler) and may transition tasks
    through the reconciler to simulate the agent tree's natural
    activity.

    The adapter also exposes ``invoke_follow_up`` so the tests can
    **assert it is never called** under the goldfive#163 contract.
    If the overlay ever re-introduces follow-up dispatch, these
    tests catch it via ``follow_up_calls``.
    """

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
        # Legacy path — not exercised in overlay mode by default but
        # kept as a fallback.
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


def _three_task_plan() -> Plan:
    return Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
            Task(id="t1", title="b", assignee_agent_id="agent_b"),
            Task(id="t2", title="c", assignee_agent_id="agent_c"),
        ],
        edges=[
            TaskEdge(from_task_id="t0", to_task_id="t1"),
            TaskEdge(from_task_id="t1", to_task_id="t2"),
        ],
    )


# ---------------------------------------------------------------------------
# Overlay ON: single passthrough, no follow-ups under any circumstance.
# ---------------------------------------------------------------------------


async def test_overlay_mode_single_passthrough_no_missed_tasks() -> None:
    """If the passthrough transitions every task COMPLETED via the
    reconciler, the overlay exits cleanly and no follow-ups fire.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str, session: Session, reconciler: Any
    ) -> InvocationResult:
        # Simulate tree activity: every sub-agent fires.
        for agent_name, tid in (("agent_a", "t0"), ("agent_b", "t1"), ("agent_c", "t2")):
            await reconciler.on_before_agent(agent_name=agent_name, invocation_id=f"inv_{tid}")
            await reconciler.on_after_agent(agent_name=agent_name, invocation_id=f"inv_{tid}")
        return InvocationResult(task_id="", text="all done")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="make it",
    )

    assert outcome.success is True
    assert adapter.passthrough_calls == ["make it"]
    assert adapter.follow_up_calls == [], "no follow-ups should ever fire under #163"
    for t in plan.tasks:
        assert t.status is TaskStatus.COMPLETED
    assert sink.payload_kinds()[-1] == "run_completed"


async def test_overlay_does_not_dispatch_follow_ups() -> None:
    """goldfive#163: even when the reconciler reports missed PENDING
    tasks, the overlay MUST NOT dispatch ``invoke_follow_up``.

    goldfive#208 (revised contract): missed tasks no longer become
    NOT_NEEDED at end-of-invocation. They remain PENDING and are
    carried forward to the next turn (Conversation.stash_plan /
    prior_plan_for) where the next user message can drive them. Only
    structurally UNREACHABLE PENDING tasks (predecessors broken) are
    cancelled at overlay end.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Only t0 is exercised by the tree. t1 and t2 stay PENDING.
        await reconciler.on_before_agent(agent_name="agent_a", invocation_id="inv_t0")
        await reconciler.on_after_agent(agent_name="agent_a", invocation_id="inv_t0")
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="do it",
    )

    assert outcome.success is True
    # Passthrough ran exactly once.
    assert adapter.passthrough_calls == ["do it"]
    # CRITICAL: no follow-up dispatch under any circumstance.
    assert adapter.follow_up_calls == [], (
        f"overlay must not dispatch follow-ups (goldfive#163); got {adapter.follow_up_calls!r}"
    )
    # t0 completed via the reconciler. t1 and t2 stay PENDING — t1's
    # predecessor t0 just COMPLETED so t1 is reachable next turn; t2's
    # predecessor t1 is still PENDING (not broken). The overlay leaves
    # both alone for the next turn to drive.
    by_id = {t.id: t.status for t in plan.tasks}
    assert by_id["t0"] is TaskStatus.COMPLETED
    assert by_id["t1"] is TaskStatus.PENDING, (
        f"reachable t1 should remain PENDING for next turn, got {by_id['t1']}"
    )
    assert by_id["t2"] is TaskStatus.PENDING, (
        f"reachable t2 should remain PENDING for next turn, got {by_id['t2']}"
    )
    # No NOT_NEEDED transitions, no CANCELLED transitions — all
    # PENDING tasks are reachable.
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.NOT_NEEDED] == []
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.CANCELLED] == []


async def test_overlay_pending_stays_pending_when_predecessors_pending() -> None:
    """When the tree does nothing and all tasks remain PENDING with
    no broken predecessor chain, the overlay leaves every task PENDING.

    goldfive#208 contract: end-of-overlay alone is NOT a sufficient
    reason to mark NOT_NEEDED. The next turn (or a USER_STEER) can
    drive these tasks forward.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Tree does nothing — all three tasks stay PENDING.
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="anything",
    )

    assert outcome.success is True
    # Zero NOT_NEEDED transitions and zero CANCELLED transitions.
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.NOT_NEEDED] == []
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.CANCELLED] == []
    # All tasks remain PENDING for the next turn.
    for t in plan.tasks:
        assert t.status is TaskStatus.PENDING
    # And of course: no follow-up dispatch.
    assert adapter.follow_up_calls == []


async def test_overlay_cancels_unreachable_pending_when_predecessor_failed() -> None:
    """goldfive#208: a PENDING task whose predecessor went CANCELLED
    (with no live replacement) is structurally unreachable. End-of-
    overlay should CANCEL it (not NOT_NEEDED) so sinks see a coherent
    plan-end and the next turn doesn't try to drive a dead task.
    """
    plan = Plan(
        id="p_unreach",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="root", assignee_agent_id="agent_a", status=TaskStatus.CANCELLED),
            Task(id="t1", title="downstream", assignee_agent_id="agent_b"),
        ],
        edges=[TaskEdge(from_task_id="t0", to_task_id="t1")],
    )
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="anything",
    )

    assert outcome.success is True
    cancelled = [tid for tid, to in steerer.transitions if to is TaskStatus.CANCELLED]
    assert "t1" in cancelled, (
        f"unreachable t1 should be CANCELLED, got transitions={steerer.transitions!r}"
    )
    # NOT_NEEDED is not the right disposition for unreachable.
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.NOT_NEEDED] == []


async def test_overlay_does_not_emit_legacy_not_needed_reason() -> None:
    """Regression guard: the legacy reason string ``tree did not
    exercise; no follow-up dispatched`` must NOT appear post-#208.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,  # noqa: ARG001
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="anything",
    )

    # No detail field on any recorded transition should mention the
    # retired phrase.
    for tid, to, *_rest in (
        (t.id, t.to, getattr(t, "detail", "")) if hasattr(t, "to") else (None, None, "")
        for t in []
    ):
        # placeholder — StubSteerer's transitions records (id, to_status)
        del tid, to
    # The transitions list itself should not contain a NOT_NEEDED with
    # the legacy reason — the simpler assertion is that no NOT_NEEDED
    # transition was emitted at all in this scenario.
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.NOT_NEEDED] == []


async def test_overlay_no_pending_tasks_no_not_needed_transitions() -> None:
    """When every task is already terminal when the passthrough
    ends, the overlay does NOT emit spurious NOT_NEEDED transitions.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Simulate the tree completing every task directly.
        if session.plan is not None:
            for t in session.plan.tasks:
                t.status = TaskStatus.COMPLETED
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="go",
    )

    # No NOT_NEEDED transitions when nothing was pending.
    assert [tid for tid, to in steerer.transitions if to is TaskStatus.NOT_NEEDED] == []


# ---------------------------------------------------------------------------
# Deprecation: max_follow_up_rounds kwarg is accepted but warns + ignored.
# ---------------------------------------------------------------------------


def test_max_follow_up_rounds_kwarg_is_deprecated_and_ignored() -> None:
    """Back-compat: ``max_follow_up_rounds=`` on the executor is
    accepted but emits a ``DeprecationWarning`` and has no effect.

    goldfive#163 removed the follow-up loop; the parameter is
    retained for one release so external callers don't break on
    upgrade.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        executor = SequentialExecutor(overlay_mode=True, max_follow_up_rounds=5)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning for max_follow_up_rounds"
    assert "max_follow_up_rounds" in str(deprecations[0].message)
    # And the attribute is NOT stored on the instance (there's no
    # follow-up loop to parameterise).
    assert not hasattr(executor, "max_follow_up_rounds")


# ---------------------------------------------------------------------------
# Regression guards: STEER / CANCEL / adapter-error paths preserved.
# ---------------------------------------------------------------------------


async def test_overlay_steer_restart_still_works() -> None:
    """Regression: ``kind == "steer"`` from the passthrough control
    loop still restarts the invocation with the steer body. The
    goldfive#163 change only touched the post-result NOT_NEEDED
    sweep; the STEER branch must behave identically.

    Uses a simple stub that delivers a synthetic STEER by raising
    a sentinel and then completing on the second call. The full
    control-channel integration is covered in
    ``tests/test_overlay_steer.py``; this test is a belt-and-
    suspenders check that the restart loop in ``_run_overlay``
    still routes STEERs to ``steerer.observe`` and re-invokes
    passthrough.
    """
    # We exercise the STEER branch through the real
    # ``_invoke_passthrough_with_control`` code path by monkey-
    # patching a version that returns (kind, payload) directly.
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    class SteerOnceAdapter(OverlayStubAdapter):
        def __init__(self) -> None:
            super().__init__()

    adapter = SteerOnceAdapter()
    executor = SequentialExecutor(overlay_mode=True)

    call_idx = {"n": 0}

    async def _fake_invoke_passthrough_with_control(
        *, adapter, session, steerer, sinks, control, reconciler, user_input
    ):  # noqa: ARG001
        call_idx["n"] += 1
        adapter.passthrough_calls.append(user_input)
        if call_idx["n"] == 1:
            # Emulate a STEER arriving mid-invocation.
            msg = type(
                "StubSteerMsg",
                (),
                {"kind": type("K", (), {"value": "STEER"})(), "payload": {"note": "switch focus"}},
            )()
            return ("steer", msg)
        # Second invocation: mark every task COMPLETED directly.
        if session.plan is not None:
            for t in session.plan.tasks:
                t.status = TaskStatus.COMPLETED
        return ("result", InvocationResult(task_id="", text="done"))

    executor._invoke_passthrough_with_control = _fake_invoke_passthrough_with_control  # type: ignore[assignment]

    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="v0",
    )

    assert outcome.success is True
    # Two passthrough invocations: v0 (steered), then the steer
    # body (wrapped in the goldfive#152 "USER STEERING CONTROL"
    # header). We assert on the body substring rather than exact
    # equality so the header wording can evolve without flaking
    # this test.
    assert len(adapter.passthrough_calls) == 2
    assert adapter.passthrough_calls[0] == "v0"
    assert "switch focus" in adapter.passthrough_calls[1]
    # Steerer observed the STEER message.
    assert any(
        str(getattr(getattr(o, "kind", None), "value", "")).upper() == "STEER"
        for o in steerer.observed
    ), "steerer.observe was not called with the STEER message"
    # No follow-up dispatch (even though we had a steer in the middle).
    assert adapter.follow_up_calls == []


async def test_overlay_cancel_still_works() -> None:
    """Regression: ``kind == "cancelled"`` terminates the run with
    ``ExecutionOutcome(success=False)`` and a ``RunAborted`` sink
    event. The goldfive#163 change did not touch this branch.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _fake_invoke_passthrough_with_control(
        *, adapter, session, steerer, sinks, control, reconciler, user_input
    ):  # noqa: ARG001
        adapter.passthrough_calls.append(user_input)
        return ("cancelled", "user aborted")

    adapter = OverlayStubAdapter()
    executor = SequentialExecutor(overlay_mode=True)
    executor._invoke_passthrough_with_control = _fake_invoke_passthrough_with_control  # type: ignore[assignment]

    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="go",
    )

    assert outcome.success is False
    assert "aborted" in (outcome.reason or "")
    assert sink.payload_kinds()[-1] == "run_aborted"
    # No follow-ups, no NOT_NEEDED sweep on cancel — we exit early.
    assert adapter.follow_up_calls == []
    # Tasks remain PENDING (we aborted; didn't reach the sweep).
    for t in plan.tasks:
        assert t.status is TaskStatus.PENDING


async def test_overlay_adapter_error_still_works() -> None:
    """Regression: ``kind == "adapter_error"`` terminates the run
    with a ``RunAborted`` event and a descriptive reason.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()

    boom = RuntimeError("adapter exploded")

    async def _fake_invoke_passthrough_with_control(
        *, adapter, session, steerer, sinks, control, reconciler, user_input
    ):  # noqa: ARG001
        adapter.passthrough_calls.append(user_input)
        return ("adapter_error", boom)

    adapter = OverlayStubAdapter()
    executor = SequentialExecutor(overlay_mode=True)
    executor._invoke_passthrough_with_control = _fake_invoke_passthrough_with_control  # type: ignore[assignment]

    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="go",
    )

    assert outcome.success is False
    assert "adapter exploded" in (outcome.reason or "")
    assert sink.payload_kinds()[-1] == "run_aborted"
    assert adapter.follow_up_calls == []


# ---------------------------------------------------------------------------
# Overlay OFF: preserve the legacy per-task behaviour.
# ---------------------------------------------------------------------------


class LegacyStubAdapter:
    """Adapter with only ``invoke(task)`` — no passthrough hook."""

    def __init__(self) -> None:
        self.invocations: list[str] = []

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        self.invocations.append(task.id)
        for t in session.plan.tasks:
            if t.id == task.id:
                t.status = TaskStatus.COMPLETED
                return InvocationResult(task_id=task.id, text=f"done:{task.id}")
        return InvocationResult(task_id=task.id, text="")


async def test_overlay_off_falls_back_to_per_task_loop() -> None:
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()
    adapter = LegacyStubAdapter()

    executor = SequentialExecutor(overlay_mode=False, max_task_invocations=10)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    assert outcome.success is True
    assert adapter.invocations == ["t0", "t1", "t2"]


async def test_overlay_on_but_adapter_missing_passthrough_falls_back() -> None:
    """Ducktype guard: an adapter without ``invoke_passthrough`` should
    not crash the executor — we transparently fall back to the legacy
    per-task loop so callers with overlay_mode=True but a custom
    adapter still get a working run.
    """
    plan = _three_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer()
    sink = RecordingSink()
    adapter = LegacyStubAdapter()

    executor = SequentialExecutor(overlay_mode=True, max_task_invocations=10)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="anything",
    )

    assert outcome.success is True
    # Legacy per-task dispatch used because adapter lacks passthrough.
    assert adapter.invocations == ["t0", "t1", "t2"]

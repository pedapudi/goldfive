"""Executor integration tests for ControlChannel (part of #71).

These tests exercise the goldfive executors' ControlChannel hooks:
pause/resume between tasks, mid-task cancel, mid-task steer, rewind,
and the synthetic status-report emission.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from goldfive.control import (
    AckResult,
    ControlAck,
    ControlChannel,
    ControlKind,
    ControlMessage,
)
from goldfive.executors import ParallelDAGExecutor, SequentialExecutor
from goldfive.protocols import EventSink
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
# Test helpers — minimal stubs
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

    def drift_events(self) -> list[Any]:
        out: list[Any] = []
        for e in self.events:
            if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "drift_detected":
                out.append(e.drift_detected)
        return out


class StubSteerer:
    """Steerer stub that applies transitions directly and records observations.

    When ``refine_result`` is set, ``observe(msg)`` calls it on
    ``ControlMessage`` and swaps session.plan with the result. This
    emulates the real DefaultSteerer's USER_STEER behavior.
    """

    def __init__(self, *, refine_result: Plan | None = None) -> None:
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []
        self.refine_result = refine_result

    def bind(self, *, sinks: list[EventSink], planner: Any) -> None:
        self._sinks = sinks
        self._planner = planner

    async def observe(self, event: Any, session: Session) -> None:
        self.observed.append(event)
        # Emulate DefaultSteerer steering: when observing a STEER
        # ControlMessage, swap the plan on the session.
        kind = getattr(event, "kind", None)
        if (
            self.refine_result is not None
            and kind is not None
            and str(getattr(kind, "value", kind)).upper() == "STEER"
        ):
            session.plan = self.refine_result

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        if session.plan is None:
            return
        for t in session.plan.tasks:
            if t.id == task_id:
                t.status = to
                if to == TaskStatus.COMPLETED and detail:
                    session.completed_results[task_id] = detail
                return

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:
        return None


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[tuple[Plan, DriftEvent]] = []
        self._refine_result: Plan | None = None

    def set_refine_result(self, plan: Plan | None) -> None:
        self._refine_result = plan

    async def generate(
        self, *, goals: list, available_agents: list[str], context: Any | None = None
    ) -> Plan | None:
        return None

    async def refine(
        self, *, plan: Plan, drift: DriftEvent, goals: list
    ) -> Plan | None:
        self.refine_calls.append((plan, drift))
        return self._refine_result


class StubAdapter:
    def __init__(
        self,
        *,
        on_invoke: Callable[[Task, Session], Awaitable[InvocationResult]],
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
        return await self._on_invoke(task, session)


def _linear_plan(ids: list[str], run_id: str = "run-1") -> Plan:
    tasks = [Task(id=t, title=f"Task {t}") for t in ids]
    edges = [
        TaskEdge(from_task_id=a, to_task_id=b)
        for a, b in zip(ids, ids[1:], strict=False)
    ]
    return Plan(id="p0", run_id=run_id, goal_ids=[], tasks=tasks, edges=edges)


def _fresh_session(run_id: str = "run-1") -> Session:
    return Session(run_id=run_id)


async def _drain_acks(
    channel: ControlChannel, *, count: int, timeout: float = 1.0
) -> list[ControlAck]:
    acks: list[ControlAck] = []

    async def _consume() -> None:
        async for a in channel.acks():
            acks.append(a)
            if len(acks) >= count:
                return

    try:
        await asyncio.wait_for(_consume(), timeout=timeout)
    except TimeoutError:
        pass
    return acks


# ---------------------------------------------------------------------------
# Sequential executor: mid-task CANCEL
# ---------------------------------------------------------------------------


async def test_sequential_cancel_mid_task_aborts_run() -> None:
    plan = _linear_plan(["t0", "t1"])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    cancel_sent = asyncio.Event()
    task_started = asyncio.Event()

    async def _long_running(task: Task, session: Session) -> InvocationResult:
        task_started.set()
        # Adapter cooperates with cancellation: awaits a long sleep, which
        # raises CancelledError on task.cancel().
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancel_sent.set()
            raise
        return InvocationResult(task_id=task.id, text="done")

    adapter = StubAdapter(on_invoke=_long_running)

    async def _send_cancel() -> None:
        await task_started.wait()
        await channel.send(
            ControlMessage(
                kind=ControlKind.CANCEL, payload={"reason": "user aborted"}
            )
        )

    executor = SequentialExecutor(max_task_invocations=5)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        )
    )
    await _send_cancel()
    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is False
    assert "user aborted" in (outcome.reason or "")
    # Adapter task received CancelledError.
    assert cancel_sent.is_set()
    # RunAborted emitted.
    assert sink.payload_kinds()[-1] == "run_aborted"
    # Second task never ran.
    assert adapter.invocations == ["t0"]
    # Task was marked CANCELLED.
    assert plan.tasks[0].status == TaskStatus.CANCELLED
    # Ack published.
    acks = await _drain_acks(channel, count=1, timeout=1.0)
    assert len(acks) == 1 and acks[0].result == AckResult.SUCCESS


# ---------------------------------------------------------------------------
# Sequential executor: mid-task STEER
# ---------------------------------------------------------------------------


async def test_sequential_steer_mid_task_cancels_and_replans() -> None:
    plan = _linear_plan(["t0", "t1", "t2"])
    session = _fresh_session()

    # Refined plan: t0 cancelled, new t0b → t1b.
    refined = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=[],
        tasks=[
            Task(id="t0b", title="New Task 0"),
            Task(id="t1b", title="New Task 1"),
        ],
        edges=[TaskEdge(from_task_id="t0b", to_task_id="t1b")],
        revision_reason="user steer",
        revision_kind=DriftKind.USER_STEER.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=1,
    )
    steerer = StubSteerer(refine_result=refined)
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    task_started = asyncio.Event()

    async def _handler(task: Task, session: Session) -> InvocationResult:
        if task.id == "t0":
            task_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
        # Subsequent tasks: complete immediately.
        for t in session.plan.tasks:
            if t.id == task.id:
                t.status = TaskStatus.COMPLETED
                break
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(on_invoke=_handler)

    async def _send_steer() -> None:
        await task_started.wait()
        await channel.send(
            ControlMessage(
                kind=ControlKind.STEER,
                payload={"note": "change the approach"},
            )
        )

    executor = SequentialExecutor(max_task_invocations=8)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        )
    )
    await _send_steer()
    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    # The in-flight task was cancelled; refined plan's tasks executed.
    assert outcome.success is True
    assert adapter.invocations[0] == "t0"
    assert "t0b" in adapter.invocations and "t1b" in adapter.invocations
    # Steerer observed the STEER ControlMessage.
    steer_obs = [
        o for o in steerer.observed
        if str(getattr(getattr(o, "kind", None), "value", "")).upper() == "STEER"
    ]
    assert len(steer_obs) == 1
    # Original t0 was marked CANCELLED.
    original_t0 = next(t for t in plan.tasks if t.id == "t0")
    assert original_t0.status == TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Sequential executor: PAUSE/RESUME between tasks
# ---------------------------------------------------------------------------


async def test_sequential_pause_blocks_next_task_until_resume() -> None:
    plan = _linear_plan(["t0", "t1", "t2"])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    # t0 completes, then we PAUSE; t1 should not start until RESUME.
    t0_done = asyncio.Event()
    t1_started = asyncio.Event()

    async def _handler(task: Task, session: Session) -> InvocationResult:
        if task.id == "t0":
            for t in session.plan.tasks:
                if t.id == "t0":
                    t.status = TaskStatus.COMPLETED
            t0_done.set()
            return InvocationResult(task_id=task.id, text="done:t0")
        if task.id == "t1":
            t1_started.set()
        for t in session.plan.tasks:
            if t.id == task.id:
                t.status = TaskStatus.COMPLETED
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(on_invoke=_handler)

    # Send PAUSE before anything starts; pre-task drain picks it up.
    await channel.send(ControlMessage(kind=ControlKind.PAUSE))

    executor = SequentialExecutor(max_task_invocations=8)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        )
    )

    # Give the executor a moment to enter the paused loop.
    await asyncio.sleep(0.1)
    assert not t0_done.is_set(), "t0 must not have started under PAUSE"
    assert not runner_task.done()

    # Resume: executor should now run all three tasks.
    await channel.send(ControlMessage(kind=ControlKind.RESUME))

    outcome = await asyncio.wait_for(runner_task, timeout=5.0)
    assert outcome.success is True
    assert adapter.invocations == ["t0", "t1", "t2"]
    assert t0_done.is_set()
    assert t1_started.is_set()

    # Two acks published (pause + resume).
    acks = await _drain_acks(channel, count=2, timeout=1.0)
    assert len(acks) == 2
    assert all(a.result == AckResult.SUCCESS for a in acks)


# ---------------------------------------------------------------------------
# Sequential executor: REWIND_TO resets downstream tasks
# ---------------------------------------------------------------------------


async def test_sequential_rewind_between_tasks_re_executes_target() -> None:
    """Pre-queue PAUSE+REWIND+RESUME so the rewind happens cleanly between
    tasks (no mid-task race) and the executor re-executes the rewound
    task on resume.
    """
    plan = _linear_plan(["t0", "t1"])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    counts: dict[str, int] = {"t0": 0, "t1": 0}

    async def _handler(task: Task, session: Session) -> InvocationResult:
        counts[task.id] += 1
        for t in session.plan.tasks:
            if t.id == task.id:
                t.status = TaskStatus.COMPLETED
        # After t0's first run, queue PAUSE / REWIND t0 / RESUME so the
        # next pre-task drain reinstates t0 as PENDING before picking.
        if task.id == "t0" and counts["t0"] == 1:
            await channel.send(ControlMessage(kind=ControlKind.PAUSE))
            await channel.send(
                ControlMessage(
                    kind=ControlKind.REWIND_TO, payload={"task_id": "t0"}
                )
            )
            await channel.send(ControlMessage(kind=ControlKind.RESUME))
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(on_invoke=_handler)

    executor = SequentialExecutor(max_task_invocations=12)
    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        ),
        timeout=5.0,
    )

    assert outcome.success is True
    # t0 re-executed once after the rewind; t1 ran once (since rewind
    # was to t0, t1 was also downstream and re-set to PENDING).
    assert counts == {"t0": 2, "t1": 1}
    for t in plan.tasks:
        assert t.status == TaskStatus.COMPLETED


async def test_rewind_helper_resets_target_and_downstream_only() -> None:
    """Direct test of _rewind_plan (dispatch_control's REWIND_TO path)."""
    from goldfive.executors._control import dispatch_control

    plan = _linear_plan(["t0", "t1", "t2"])
    for t in plan.tasks:
        t.status = TaskStatus.COMPLETED
    session = _fresh_session()
    session.plan = plan
    session.completed_results = {"t0": "a", "t1": "b", "t2": "c"}
    steerer = StubSteerer()

    msg = ControlMessage(kind=ControlKind.REWIND_TO, payload={"task_id": "t1"})
    outcome = await dispatch_control(
        msg, session=session, steerer=steerer, sinks=[]
    )

    assert outcome.ack.result == AckResult.SUCCESS
    assert outcome.rewind_task_id == "t1"
    # t0 unchanged; t1 and t2 reset to PENDING with completed_results cleared.
    assert plan.tasks[0].status == TaskStatus.COMPLETED
    assert plan.tasks[1].status == TaskStatus.PENDING
    assert plan.tasks[2].status == TaskStatus.PENDING
    assert "t0" in session.completed_results
    assert "t1" not in session.completed_results
    assert "t2" not in session.completed_results

    # Unknown task id yields FAILURE ack.
    msg_unknown = ControlMessage(
        kind=ControlKind.REWIND_TO, payload={"task_id": "does_not_exist"}
    )
    outcome_unknown = await dispatch_control(
        msg_unknown, session=session, steerer=steerer, sinks=[]
    )
    assert outcome_unknown.ack.result == AckResult.FAILURE


# ---------------------------------------------------------------------------
# Sequential executor: STATUS_QUERY emits a synthetic event
# ---------------------------------------------------------------------------


async def test_sequential_status_query_emits_event_and_acks_success() -> None:
    plan = _linear_plan(["t0", "t1"])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    status_sent = asyncio.Event()

    async def _handler(task: Task, session: Session) -> InvocationResult:
        if task.id == "t0":
            # Send a STATUS_QUERY while t0 is running. Using a raw
            # string kind is the forward-compat path; ControlMessage's
            # kind field is not runtime-enforced.
            await channel.send(
                ControlMessage(kind="STATUS_QUERY", payload={})  # type: ignore[arg-type]
            )
            status_sent.set()
            # Brief sleep so the executor has time to pick up the
            # control message before the adapter returns.
            await asyncio.sleep(0.1)
        for t in session.plan.tasks:
            if t.id == task.id:
                t.status = TaskStatus.COMPLETED
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(on_invoke=_handler)

    executor = SequentialExecutor(max_task_invocations=8)
    outcome = await asyncio.wait_for(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        ),
        timeout=5.0,
    )

    assert outcome.success is True
    assert status_sent.is_set()
    # A DriftDetected event was emitted for the status report.
    drifts = sink.drift_events()
    assert any("status_query" in (getattr(d, "detail", "")) for d in drifts)
    # Ack was SUCCESS.
    acks = await _drain_acks(channel, count=1, timeout=1.0)
    assert len(acks) == 1 and acks[0].result == AckResult.SUCCESS


# ---------------------------------------------------------------------------
# Sequential executor: defensive 5s timeout on adapter that ignores cancel
# ---------------------------------------------------------------------------


async def test_sequential_cancel_abandons_uncooperative_adapter() -> None:
    """Adapter that swallows CancelledError: executor still aborts.

    The defensive 5s grace window means this test takes ~5s. We mark
    it with a short override by monkey-patching the helper's deadline.
    """
    import time as _time

    plan = _linear_plan(["t0"])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    started = asyncio.Event()

    async def _uncooperative(task: Task, session: Session) -> InvocationResult:
        started.set()
        # Swallow cancels for a bounded number of attempts, then let the
        # coroutine return so pytest-asyncio can tear down cleanly. The
        # executor has already aborted by then via its grace window.
        for _ in range(40):
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                # Swallow the cancel, keep going briefly.
                pass
        return InvocationResult(task_id=task.id, text="eventually returned")

    adapter = StubAdapter(on_invoke=_uncooperative)

    executor = SequentialExecutor(max_task_invocations=5)

    # Shrink the grace window via a targeted monkey-patch so the test
    # runs in ~0.3s, not 5s. Grab the original from __dict__ so the
    # restore preserves the staticmethod descriptor (attribute access
    # would unwrap it and the subsequent `self._cancel_invoke_task(...)`
    # calls would pass `self` as a stray first positional arg).
    original = SequentialExecutor.__dict__["_cancel_invoke_task"]

    @staticmethod
    async def _quick_cancel(invoke_task: asyncio.Task) -> None:
        if invoke_task.done():
            return
        invoke_task.cancel()
        deadline = _time.monotonic() + 0.3
        while _time.monotonic() < deadline:
            if invoke_task.done():
                return
            await asyncio.sleep(0.02)

    SequentialExecutor._cancel_invoke_task = _quick_cancel  # type: ignore[assignment]

    try:
        runner_task = asyncio.create_task(
            executor.run(
                plan=plan,
                session=session,
                adapter=adapter,
                steerer=steerer,
                planner=planner,
                sinks=[sink],
                control=channel,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await channel.send(ControlMessage(kind=ControlKind.CANCEL))
        outcome = await asyncio.wait_for(runner_task, timeout=3.0)
        assert outcome.success is False
        assert sink.payload_kinds()[-1] == "run_aborted"
    finally:
        SequentialExecutor._cancel_invoke_task = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Parallel executor: mid-stage CANCEL cancels all in-flight tasks
# ---------------------------------------------------------------------------


async def test_parallel_cancel_mid_stage_cancels_all_tasks() -> None:
    # One stage with three parallel tasks.
    tasks = [Task(id=t, title=f"Task {t}") for t in ("a", "b", "c")]
    plan = Plan(id="p0", run_id="run-1", goal_ids=[], tasks=tasks, edges=[])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    started: set[str] = set()
    cancelled: set[str] = set()
    all_started = asyncio.Event()

    async def _long_running(task: Task, session: Session) -> InvocationResult:
        started.add(task.id)
        if len(started) == 3:
            all_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.add(task.id)
            raise
        return InvocationResult(task_id=task.id, text="done")

    adapter = StubAdapter(on_invoke=_long_running)

    async def _send_cancel() -> None:
        await all_started.wait()
        await channel.send(ControlMessage(kind=ControlKind.CANCEL))

    executor = ParallelDAGExecutor(max_concurrency=0)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        )
    )
    await _send_cancel()
    outcome = await asyncio.wait_for(runner_task, timeout=5.0)

    assert outcome.success is False
    assert started == {"a", "b", "c"}
    assert cancelled == {"a", "b", "c"}
    assert sink.payload_kinds()[-1] == "run_aborted"


# ---------------------------------------------------------------------------
# Parallel executor: PAUSE mid-stage lets the stage finish, then blocks
# ---------------------------------------------------------------------------


async def test_parallel_pause_before_stage_blocks_until_resume() -> None:
    tasks = [Task(id=t, title=f"Task {t}") for t in ("a", "b")]
    plan = Plan(id="p0", run_id="run-1", goal_ids=[], tasks=tasks, edges=[])
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()
    channel = ControlChannel()

    started: set[str] = set()

    async def _handler(task: Task, session: Session) -> InvocationResult:
        started.add(task.id)
        # Let caller observe the state; don't block.
        return InvocationResult(task_id=task.id, text=f"done:{task.id}")

    adapter = StubAdapter(on_invoke=_handler)

    # PAUSE in advance: pre-stage control drain picks it up before any
    # invocation happens.
    await channel.send(ControlMessage(kind=ControlKind.PAUSE))

    executor = ParallelDAGExecutor(max_concurrency=0)

    runner_task = asyncio.create_task(
        executor.run(
            plan=plan,
            session=session,
            adapter=adapter,
            steerer=steerer,
            planner=planner,
            sinks=[sink],
            control=channel,
        )
    )

    await asyncio.sleep(0.1)
    assert started == set(), "no task should have started while paused"
    assert not runner_task.done()

    await channel.send(ControlMessage(kind=ControlKind.RESUME))
    outcome = await asyncio.wait_for(runner_task, timeout=5.0)
    assert outcome.success is True
    assert started == {"a", "b"}

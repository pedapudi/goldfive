"""observation_only carve-out for SequentialExecutor's abort-on-FAILED enforcement (goldfive#260).

The executor aborts the run when a task is FAILED with no ``supersedes``
replacement in the current plan revision. Under active steering this is
the right behaviour — the corrective refine produces that replacement
and the executor installs it before reaching the abort check. Under
:class:`~goldfive.config.SteeringConfig.observation_only` the refine is
dry-run, the replacement never lands on ``session.plan``, and the
abort fires on a state goldfive cannot fix but the coordinator's
autonomous flow may still recover from.

Per the design framing in goldfive#254/#260, ``observation_only`` is
PASSIVE — goldfive observes, doesn't enforce. The run should terminate
when the coordinator stops dispatching, not when goldfive can't supply
a corrective revision. These tests cover three sites where the
executor would otherwise emit ``run_aborted`` for
"failed-without-replacement":

* Sequential per-task loop, ``fail_fast=True`` (the in-loop abort).
* Sequential per-task loop, ``fail_fast=False`` (the post-loop abort).
* Overlay-mode terminal emission, ``fail_fast=True``.

Each is asserted under BOTH ``observation_only=False`` (still aborts —
regression guard) AND ``observation_only=True`` (carve-out fires).

Live reproduction (2026-05-11, session ``37632cbc``, gemma-4-26B):
pothos task ``create_slides`` FAILED at 16:41:11; seq=57 plan_revised
at 16:41:16 was the dry-run refine; seq=81 ``run_aborted`` at 16:41:42
with ``goldfive run aborted: one or more tasks failed without a live
replacement: create_slides`` defeated the observation-only contract.
"""

from __future__ import annotations

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
# Stubs (mirror tests/test_sequential_executor.py shapes; the steerer
# carries the ``_observation_only`` attribute the executor's carve-out
# reads).
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
    """Minimal Steerer that exposes ``_observation_only`` the way the
    real :class:`~goldfive.steerer.DefaultSteerer` does (the executor's
    carve-out reads the private attribute directly so the goldfive#260
    patch stays small).
    """

    def __init__(self, *, observation_only: bool = False) -> None:
        self._observation_only = observation_only
        self._sinks: list[EventSink] = []
        self._planner: Any = None
        self.observed: list[Any] = []
        self.transitions: list[tuple[str, TaskStatus]] = []
        self.emitted_plan_revised: list[Any] = []

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


class StubPlanner:
    async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def refine_steer(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class SequentialStubAdapter:
    """Adapter for the per-task SequentialExecutor path."""

    def __init__(
        self,
        *,
        steerer: StubSteerer,
        on_invoke: Callable[
            [Task, Session, StubSteerer], Awaitable[InvocationResult]
        ],
    ) -> None:
        self._steerer = steerer
        self._on_invoke = on_invoke
        self.invocations: list[str] = []

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        self.invocations.append(task.id)
        return await self._on_invoke(task, session, self._steerer)


class OverlayStubAdapter:
    """Adapter for the overlay (single-passthrough) path. Mirrors the
    one in ``tests/test_sequential_executor_overlay.py`` but trimmed
    to the surface this file exercises.
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
        return ["stub"]

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
# Helpers.
# ---------------------------------------------------------------------------


def _two_independent_plan(run_id: str = "r1") -> Plan:
    """Two independent root tasks. Lets a sibling proceed past a sibling
    failure under fail_fast=False (and lets observation_only's carve-out
    of the in-loop abort under fail_fast=True be observed: with no
    edges, the loop picks the second task after the first fails).
    """
    tasks = [
        Task(id="t0", title="t0", assignee_agent_id="agent_a"),
        Task(id="t1", title="t1", assignee_agent_id="agent_b"),
    ]
    return Plan(id="p0", run_id=run_id, goal_ids=[], tasks=tasks, edges=[])


def _linear_two_plan(run_id: str = "r1") -> Plan:
    """t0 -> t1. A failure on t0 with no replacement leaves t1
    structurally unreachable, but the in-loop abort fires before t1
    would be picked.
    """
    tasks = [
        Task(id="t0", title="t0", assignee_agent_id="agent_a"),
        Task(id="t1", title="t1", assignee_agent_id="agent_b"),
    ]
    edges = [TaskEdge(from_task_id="t0", to_task_id="t1")]
    return Plan(id="p0", run_id=run_id, goal_ids=[], tasks=tasks, edges=edges)


def _fresh_session(run_id: str = "r1") -> Session:
    return Session(run_id=run_id)


# ---------------------------------------------------------------------------
# Sequential per-task loop, fail_fast=True — site 1 of the carve-out.
# ---------------------------------------------------------------------------


async def test_active_steering_aborts_when_task_fails_without_replacement_fail_fast() -> None:
    """Regression guard: with ``observation_only=False`` (active
    steering — the default for goldfive runs), a FAILED task with no
    replacement still aborts the run under ``fail_fast=True``. This is
    the established behaviour the carve-out must preserve.
    """
    plan = _linear_two_plan()
    session = _fresh_session()
    steerer = StubSteerer(observation_only=False)
    sink = RecordingSink()

    async def _fail_t0(
        task: Task, session: Session, steerer: StubSteerer
    ) -> InvocationResult:
        if task.id == "t0":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = SequentialStubAdapter(steerer=steerer, on_invoke=_fail_t0)
    executor = SequentialExecutor(max_task_invocations=5, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    # Aborted at t0 — t1 never invoked.
    assert outcome.success is False
    assert adapter.invocations == ["t0"]
    kinds = sink.payload_kinds()
    assert "run_aborted" in kinds, f"expected run_aborted in {kinds!r}"
    assert "run_completed" not in kinds
    # t0 stayed FAILED.
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["t0"] is TaskStatus.FAILED


async def test_observation_only_skips_in_loop_abort_when_task_fails_without_replacement() -> None:
    """goldfive#260: with ``observation_only=True``, the in-loop
    fail_fast=True abort path is carved out. The executor logs and
    falls through; the loop continues picking eligible tasks until
    none remain, then emits ``run_completed``.

    The FAILED task stays FAILED (we don't pretend it succeeded) —
    only the abort enforcement is gated.
    """
    plan = _two_independent_plan()
    session = _fresh_session()
    steerer = StubSteerer(observation_only=True)
    sink = RecordingSink()

    async def _fail_t0_complete_t1(
        task: Task, session: Session, steerer: StubSteerer
    ) -> InvocationResult:
        if task.id == "t0":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = SequentialStubAdapter(steerer=steerer, on_invoke=_fail_t0_complete_t1)
    executor = SequentialExecutor(max_task_invocations=5, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    # Coordinator's autonomous flow still dispatched t1 (the executor
    # carved-out the abort and picked the next eligible task).
    assert set(adapter.invocations) == {"t0", "t1"}, (
        f"observation_only should let the loop continue past the failure; "
        f"got invocations={adapter.invocations!r}"
    )
    kinds = sink.payload_kinds()
    assert "run_aborted" not in kinds, (
        f"observation_only must NOT emit run_aborted for "
        f"failed-without-replacement; got kinds={kinds!r}"
    )
    assert "run_completed" in kinds, (
        f"observation_only should emit run_completed at natural "
        f"loop exit; got kinds={kinds!r}"
    )
    # FAILED stays FAILED — we don't pretend it succeeded.
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["t0"] is TaskStatus.FAILED, (
        f"FAILED task must remain FAILED under observation_only "
        f"(we don't pretend it succeeded); got {by_id!r}"
    )
    assert by_id["t1"] is TaskStatus.COMPLETED
    # The executor reports success=True because no abort fired — the
    # outcome mirrors the natural "coordinator stopped dispatching"
    # termination.
    assert outcome.success is True


# ---------------------------------------------------------------------------
# Sequential per-task loop, fail_fast=False — site 2 of the carve-out.
# ---------------------------------------------------------------------------


async def test_active_steering_aborts_when_task_fails_without_replacement_fail_fast_false() -> (
    None
):
    """Regression guard: with ``observation_only=False`` and
    ``fail_fast=False``, the post-loop "one or more tasks failed
    without a live replacement (fail_fast=False)" abort still fires.
    """
    plan = _two_independent_plan()
    session = _fresh_session()
    steerer = StubSteerer(observation_only=False)
    sink = RecordingSink()

    async def _fail_t0_complete_t1(
        task: Task, session: Session, steerer: StubSteerer
    ) -> InvocationResult:
        if task.id == "t0":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = SequentialStubAdapter(steerer=steerer, on_invoke=_fail_t0_complete_t1)
    executor = SequentialExecutor(max_task_invocations=5, fail_fast=False)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    assert outcome.success is False
    assert set(adapter.invocations) == {"t0", "t1"}
    kinds = sink.payload_kinds()
    assert "run_aborted" in kinds
    assert "run_completed" not in kinds


async def test_observation_only_skips_post_loop_abort_fail_fast_false() -> None:
    """goldfive#260: with ``observation_only=True`` and
    ``fail_fast=False``, the post-loop "failed without replacement
    (fail_fast=False)" abort is carved out. The run reaches
    ``run_completed`` naturally.
    """
    plan = _two_independent_plan()
    session = _fresh_session()
    steerer = StubSteerer(observation_only=True)
    sink = RecordingSink()

    async def _fail_t0_complete_t1(
        task: Task, session: Session, steerer: StubSteerer
    ) -> InvocationResult:
        if task.id == "t0":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = SequentialStubAdapter(steerer=steerer, on_invoke=_fail_t0_complete_t1)
    executor = SequentialExecutor(max_task_invocations=5, fail_fast=False)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    assert outcome.success is True
    kinds = sink.payload_kinds()
    assert "run_aborted" not in kinds, (
        f"observation_only must NOT emit run_aborted (fail_fast=False); got {kinds!r}"
    )
    assert "run_completed" in kinds
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["t0"] is TaskStatus.FAILED
    assert by_id["t1"] is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Overlay (single-passthrough) path — site 3 of the carve-out. This is
# the path the live e2e reproduction (pothos create_slides) hit.
# ---------------------------------------------------------------------------


def _overlay_two_task_plan() -> Plan:
    """A 2-task overlay plan. Lets the reconciler complete one task and
    fail another, with no replacement in the plan revision.
    """
    return Plan(
        id="p_ov",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="ok_task", title="ok", assignee_agent_id="agent_a"),
            Task(id="create_slides", title="fail", assignee_agent_id="agent_b"),
        ],
        edges=[],
    )


async def test_active_steering_aborts_overlay_when_task_fails_without_replacement() -> None:
    """Regression guard: overlay path with ``observation_only=False``
    aborts when a task ends FAILED with no live replacement.
    """
    plan = _overlay_two_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer(observation_only=False)
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        await steerer.transition("ok_task", TaskStatus.COMPLETED, session=session)
        await steerer.transition("create_slides", TaskStatus.FAILED, session=session)
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="do it",
    )

    assert outcome.success is False
    kinds = sink.payload_kinds()
    assert "run_aborted" in kinds, f"expected run_aborted in {kinds!r}"
    assert "run_completed" not in kinds


async def test_observation_only_skips_overlay_abort_when_task_fails_without_replacement() -> (
    None
):
    """goldfive#260: overlay path with ``observation_only=True`` carves
    out the "failed without a live replacement" abort. The terminal
    emission is ``run_completed``; the FAILED task stays FAILED.

    This is the live e2e reproduction (session ``37632cbc``, pothos
    ``create_slides``): the executor must NOT emit ``run_aborted`` —
    it should let the ADK coordinator's autonomous flow decide next
    steps.
    """
    plan = _overlay_two_task_plan()
    session = Session(run_id="r1")
    steerer = StubSteerer(observation_only=True)
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        await steerer.transition("ok_task", TaskStatus.COMPLETED, session=session)
        await steerer.transition("create_slides", TaskStatus.FAILED, session=session)
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=True)
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
    kinds = sink.payload_kinds()
    assert "run_aborted" not in kinds, (
        f"overlay + observation_only must NOT emit run_aborted for "
        f"failed-without-replacement (goldfive#260); got kinds={kinds!r}"
    )
    assert "run_completed" in kinds, (
        f"overlay + observation_only should emit run_completed at "
        f"natural termination; got kinds={kinds!r}"
    )
    # FAILED stays FAILED — observability of the failure is preserved.
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id["create_slides"] is TaskStatus.FAILED, (
        f"FAILED task must remain FAILED under observation_only; got {by_id!r}"
    )
    assert by_id["ok_task"] is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Coordinator's autonomous flow continues after the failure.
# ---------------------------------------------------------------------------


async def test_observation_only_coordinator_continues_dispatching_after_failure() -> None:
    """goldfive#260: with ``observation_only=True``, a failure on one
    task does not stop the coordinator's autonomous flow from
    dispatching subsequent tasks. The executor invokes the next
    eligible task and the run terminates with ``run_completed``.

    This is the "coordinator still dispatching after failure" scenario
    described in the issue: the executor must defer to the coordinator,
    not pre-empt with ``run_aborted``.
    """
    # Three independent tasks: t_fail, t_ok_a, t_ok_b. Under
    # observation_only the executor must invoke all three even though
    # t_fail ended FAILED.
    plan = Plan(
        id="p_continue",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t_fail", title="fails", assignee_agent_id="agent_a"),
            Task(id="t_ok_a", title="ok a", assignee_agent_id="agent_b"),
            Task(id="t_ok_b", title="ok b", assignee_agent_id="agent_c"),
        ],
        edges=[],
    )
    session = _fresh_session()
    steerer = StubSteerer(observation_only=True)
    sink = RecordingSink()

    async def _on_invoke(
        task: Task, session: Session, steerer: StubSteerer
    ) -> InvocationResult:
        if task.id == "t_fail":
            await steerer.transition(task.id, TaskStatus.FAILED, session=session)
            return InvocationResult(task_id=task.id, text="", stop_reason="failed")
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = SequentialStubAdapter(steerer=steerer, on_invoke=_on_invoke)
    executor = SequentialExecutor(max_task_invocations=10, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
    )

    # Coordinator's flow dispatched all three.
    assert set(adapter.invocations) == {"t_fail", "t_ok_a", "t_ok_b"}, (
        f"coordinator must keep dispatching past failure under "
        f"observation_only; got invocations={adapter.invocations!r}"
    )
    kinds = sink.payload_kinds()
    assert "run_aborted" not in kinds
    assert "run_completed" in kinds
    assert outcome.success is True
    by_id = {t.id: t.status for t in (outcome.session.plan or plan).tasks}
    assert by_id == {
        "t_fail": TaskStatus.FAILED,
        "t_ok_a": TaskStatus.COMPLETED,
        "t_ok_b": TaskStatus.COMPLETED,
    }

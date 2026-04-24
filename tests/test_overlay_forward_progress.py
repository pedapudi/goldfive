"""Post-refine forward-progress tests (goldfive#202).

Two structural fixes land together:

1. ``fail_fast`` respects replacement tasks: a FAILED task with a live
   replacement in the current plan revision (refine-spawned successor
   like ``retry_<id>`` or ``<id>_v2``) does not abort the run — the
   replacement is the forward-progress path.
2. The steerer's ABSORB dispatch queues a Level 2 nudge on
   ``session.pending_nudges`` for coordinator-stuck drift kinds
   (LOOPING_REASONING / LOOPING_TOOL_CALL / SELF_REPORTED_STUCK), and
   the overlay's scoped nudge-replay consumes it between invocations
   to tell the coordinator its plan changed.

Plus: the ``run_aborted_event`` reason names the fatally-failed task(s)
when fail_fast truly fires.

These tests use lightweight stubs — no ADK, no LLM, no network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from goldfive.executors.sequential import (
    SequentialExecutor,
    _fatally_failed_task_ids,
    _has_live_replacement,
)
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
# Stubs (mirrors tests/test_sequential_executor_overlay.py shapes).
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None

    def last_run_aborted_reason(self) -> str | None:
        for e in reversed(self.events):
            if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "run_aborted":
                return e.run_aborted.reason
        return None


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


class StubAdapter:
    """Minimal legacy-path adapter (used by fail_fast tests).

    The ``on_invoke`` callback drives task state for the invoked task.
    """

    def __init__(
        self,
        *,
        on_invoke: Callable[[Task, Session], Awaitable[InvocationResult]],
    ) -> None:
        self._on_invoke = on_invoke
        self.invocations: list[str] = []

    async def register_reporting_tools(self, tools: list[Any]) -> None:  # noqa: ARG002
        return None

    @property
    def available_agents(self) -> list[str]:
        return ["stub"]

    async def invoke(self, task: Task, session: Session) -> InvocationResult:
        self.invocations.append(task.id)
        return await self._on_invoke(task, session)


class OverlayStubAdapter:
    """Overlay-path adapter. ``passthrough_effect`` can mutate the plan
    via the steerer and queue nudges directly to simulate the real
    steerer's ABSORB-nudge behaviour.
    """

    def __init__(
        self,
        *,
        passthrough_effect: Callable[[str, Session, Any], Awaitable[InvocationResult | None]]
        | None = None,
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
        if self._passthrough_effect is not None:
            result = await self._passthrough_effect(user_message, session, reconciler)
            if result is not None:
                return result
        return InvocationResult(task_id="", text="")


# ---------------------------------------------------------------------------
# Helper: inline reconciler stub the overlay uses.
# ---------------------------------------------------------------------------


def _fresh_session(run_id: str = "run-1") -> Session:
    return Session(run_id=run_id)


# ---------------------------------------------------------------------------
# Helper-level tests for the replacement predicate.
# ---------------------------------------------------------------------------


def test_has_live_replacement_lineage_pending_replacement() -> None:
    """A PENDING ``retry_<id>`` is a live replacement for FAILED ``<id>``."""
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="a", status=TaskStatus.FAILED, assignee_agent_id="a"),
            Task(
                id="retry_t0",
                title="a retry",
                status=TaskStatus.PENDING,
                assignee_agent_id="a",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True


def test_has_live_replacement_versioned_pending_replacement() -> None:
    """A PENDING ``<id>_v2`` is a live replacement for FAILED ``<id>``."""
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(
                id="define_structure",
                title="define",
                status=TaskStatus.FAILED,
                assignee_agent_id="planner",
            ),
            Task(
                id="define_structure_v2",
                title="define (v2)",
                status=TaskStatus.PENDING,
                assignee_agent_id="planner",
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is True


def test_has_live_replacement_requires_pending_or_running_not_completed() -> None:
    """A COMPLETED lineage peer is a predecessor, not a replacement.

    Regression guard for the lineage-cap scenario: ``t0`` (COMPLETED)
    and ``retry_t0`` (COMPLETED) are predecessors of a FAILED
    ``retry_retry_t0`` — they ran BEFORE it. Marking them as a
    "replacement" would hide a genuine exhausted-retries failure.
    """
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="a", status=TaskStatus.COMPLETED),
            Task(id="retry_t0", title="a retry", status=TaskStatus.COMPLETED),
            Task(
                id="retry_retry_t0",
                title="a retry retry",
                status=TaskStatus.FAILED,
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[2]
    assert _has_live_replacement(plan, failed) is False


def test_has_live_replacement_assignee_mismatch_rejected() -> None:
    """A candidate with a DIFFERENT assignee is not a replacement."""
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="a", status=TaskStatus.FAILED, assignee_agent_id="a"),
            Task(
                id="retry_t0",
                title="a retry",
                status=TaskStatus.PENDING,
                assignee_agent_id="b",  # different agent
            ),
        ],
        edges=[],
    )
    failed = plan.tasks[0]
    assert _has_live_replacement(plan, failed) is False


def test_fatally_failed_task_ids_filters_replaced() -> None:
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=[
            # Replaced (PENDING successor with same lineage root): not fatal.
            Task(id="ta", title="a", status=TaskStatus.FAILED, assignee_agent_id="a"),
            Task(
                id="retry_ta",
                title="a retry",
                status=TaskStatus.PENDING,
                assignee_agent_id="a",
            ),
            # No replacement: fatal.
            Task(id="tb", title="b", status=TaskStatus.FAILED, assignee_agent_id="b"),
            Task(id="tc", title="c", status=TaskStatus.COMPLETED),
        ],
        edges=[],
    )
    assert _fatally_failed_task_ids(plan) == ["tb"]


# ---------------------------------------------------------------------------
# Legacy run() path: fail_fast respects replacement.
# ---------------------------------------------------------------------------


async def test_fail_fast_ignores_replaced_task() -> None:
    """Rev-0 plan has FAILED ``ta``; rev-1 has PENDING ``retry_ta`` on the
    same assignee. ``fail_fast=True`` must NOT abort — the replacement is
    live forward progress.
    """
    plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(
                id="ta",
                title="original",
                status=TaskStatus.FAILED,
                assignee_agent_id="agent_a",
            ),
            Task(
                id="retry_ta",
                title="retry of original",
                status=TaskStatus.PENDING,
                assignee_agent_id="agent_a",
            ),
        ],
        edges=[],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _complete_task(task: Task, session: Session) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id=task.id, text="ok")

    adapter = StubAdapter(on_invoke=_complete_task)

    executor = SequentialExecutor(max_task_invocations=8, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is True, outcome.reason
    # The executor should have invoked retry_ta (the replacement); the
    # pre-existing FAILED ta must not have caused an abort.
    assert adapter.invocations == ["retry_ta"]
    assert sink.last_run_aborted_reason() is None


async def test_fail_fast_aborts_on_truly_fatal_failure() -> None:
    """A FAILED task with no replacement aborts under fail_fast=True, and
    the aborted event names the offending task.
    """
    plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(id="ta", title="a"),
            Task(id="tb", title="b"),
        ],
        edges=[TaskEdge(from_task_id="ta", to_task_id="tb")],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _fail_ta(task: Task, session: Session) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.FAILED, session=session)
        return InvocationResult(task_id=task.id, text="", error="boom")

    adapter = StubAdapter(on_invoke=_fail_ta)

    executor = SequentialExecutor(max_task_invocations=8, fail_fast=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is False
    reason = sink.last_run_aborted_reason() or ""
    assert "ta" in reason, reason


async def test_run_aborted_event_carries_reason_with_task_id() -> None:
    """fail_fast=False + FAILED-without-replacement path: reason names the task."""
    plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="t0"),
            Task(id="t1", title="t1"),
        ],
        edges=[TaskEdge(from_task_id="t0", to_task_id="t1")],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    planner = StubPlanner()
    sink = RecordingSink()

    async def _fail(task: Task, session: Session) -> InvocationResult:
        await steerer.transition(task.id, TaskStatus.FAILED, session=session)
        return InvocationResult(task_id=task.id, text="", error="x")

    adapter = StubAdapter(on_invoke=_fail)

    executor = SequentialExecutor(max_task_invocations=8, fail_fast=False)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=planner,
        sinks=[sink],
    )

    assert outcome.success is False
    reason = sink.last_run_aborted_reason() or ""
    # The reason should name at least one failed task id.
    assert "t0" in reason, reason


# ---------------------------------------------------------------------------
# Overlay path: Level 2 nudge wired.
# ---------------------------------------------------------------------------


async def test_level_2_nudge_triggers_overlay_replay() -> None:
    """Overlay path: when the passthrough enqueues a nudge onto
    ``session.pending_nudges`` (mirroring the real steerer's post-ABSORB
    handling for LOOPING_REASONING), the overlay re-invokes passthrough
    with a synthesized framing message and proceeds with the replacement.
    """
    plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(
                id="define_structure",
                title="define",
                assignee_agent_id="planner",
            ),
            Task(
                id="draft_slide_1",
                title="draft",
                assignee_agent_id="writer",
            ),
        ],
        edges=[TaskEdge(from_task_id="define_structure", to_task_id="draft_slide_1")],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    sink = RecordingSink()

    replay_count = 0

    async def _passthrough(
        user_message: str, session: Session, reconciler: Any
    ) -> InvocationResult:
        nonlocal replay_count
        # First invocation: simulate the LOOPING_REASONING + refine path.
        # Mark ``define_structure`` FAILED, add a PENDING ``define_structure_v2``
        # replacement, and queue a nudge.
        if replay_count == 0:
            replay_count += 1
            session.plan.tasks[0].status = TaskStatus.FAILED
            session.plan.tasks.append(
                Task(
                    id="define_structure_v2",
                    title="define (v2)",
                    status=TaskStatus.PENDING,
                    assignee_agent_id="planner",
                )
            )
            # Wire replacement into the DAG so draft_slide_1 depends on it.
            session.plan.edges.append(
                TaskEdge(from_task_id="define_structure_v2", to_task_id="draft_slide_1")
            )
            session.pending_nudges.append(
                "The prior attempt looped on define_structure. Refined plan: "
                "define (v2). Please try a different approach."
            )
            return InvocationResult(task_id="", text="")
        # Second invocation (nudge replay): coordinator sees the framing
        # message, proceeds with the replacement + downstream.
        await steerer.transition("define_structure_v2", TaskStatus.COMPLETED, session=session)
        await steerer.transition("draft_slide_1", TaskStatus.COMPLETED, session=session)
        return InvocationResult(task_id="", text="done")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True)
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="make the deck",
    )

    # Two passthrough calls: first is the original user message, second is
    # the overlay's synthesized nudge-replay message.
    assert len(adapter.passthrough_calls) == 2, adapter.passthrough_calls
    assert adapter.passthrough_calls[0] == "make the deck"
    replay_msg = adapter.passthrough_calls[1]
    assert replay_msg.startswith("[GOLDFIVE PLAN REVISION"), replay_msg
    assert "define (v2)" in replay_msg or "define_structure" in replay_msg

    # Nudges were consumed.
    assert session.pending_nudges == []

    # Run completed cleanly — the FAILED define_structure is NOT fatal
    # because its replacement (COMPLETED define_structure_v2) was live
    # when fail_fast evaluated.
    assert outcome.success is True, outcome.reason
    assert sink.last_run_aborted_reason() is None


async def test_overlay_nudge_replay_cap_is_bounded() -> None:
    """A pathological steerer that keeps re-queueing a nudge every turn
    must not loop forever — the overlay caps replays at
    ``_MAX_NUDGE_REPLAYS`` and exits cleanly.
    """
    plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[Task(id="ta", title="a", assignee_agent_id="agent_a")],
        edges=[],
    )
    session = _fresh_session()
    steerer = StubSteerer()
    sink = RecordingSink()

    async def _passthrough(
        user_message: str,  # noqa: ARG001
        session: Session,
        reconciler: Any,  # noqa: ARG001
    ) -> InvocationResult:
        # Always keep a pending nudge; never complete the task.
        session.pending_nudges.append("still stuck")
        return InvocationResult(task_id="", text="")

    adapter = OverlayStubAdapter(passthrough_effect=_passthrough)
    executor = SequentialExecutor(overlay_mode=True, fail_fast=False)
    await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=StubPlanner(),
        sinks=[sink],
        user_input="go",
    )

    # 1 initial invocation + _MAX_NUDGE_REPLAYS replay invocations.
    max_replays = SequentialExecutor._MAX_NUDGE_REPLAYS
    assert len(adapter.passthrough_calls) == 1 + max_replays, adapter.passthrough_calls


# ---------------------------------------------------------------------------
# Steerer-level test: ABSORB on LOOPING_REASONING queues a nudge.
# ---------------------------------------------------------------------------


async def test_absorb_on_looping_reasoning_queues_nudge() -> None:
    """When DefaultSteerer absorbs a LOOPING_REASONING drift (Level 1
    refine), it also queues a Level 2 nudge on ``session.pending_nudges``
    so the overlay's replay path has something to consume.
    """
    from goldfive.steerer import DefaultSteerer
    from goldfive.types import Goal

    # Seed: a plan where the refine-spawned replacement is already live.
    # The refine stub returns this revised plan.
    revised = Plan(
        id="p1",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(
                id="define_structure",
                title="define",
                status=TaskStatus.FAILED,
                assignee_agent_id="planner",
            ),
            Task(
                id="define_structure_v2",
                title="define (v2)",
                status=TaskStatus.PENDING,
                assignee_agent_id="planner",
            ),
        ],
        edges=[],
        revision_index=1,
    )

    class _StubPlanner:
        async def generate(self, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        async def refine(self, **kwargs: Any) -> Plan:  # noqa: ARG002
            return revised

    session = _fresh_session()
    session.plan = Plan(
        id="p0",
        run_id="run-1",
        goal_ids=[],
        tasks=[
            Task(
                id="define_structure",
                title="define",
                status=TaskStatus.RUNNING,
                assignee_agent_id="planner",
            ),
        ],
        edges=[],
    )
    session.goals = [Goal(id="g0", summary="make it")]

    sink = RecordingSink()
    # goldfive-steer-unification: LOOPING_REASONING@WARNING is in the
    # promotion-eligible set under the default threshold, which routes
    # through the steer path (no ABSORB nudge). This test locks in the
    # LEGACY ABSORB-with-nudge behaviour for operators who disable
    # promotion; cover the promotion path in
    # tests/test_steer_unification.py.
    steerer = DefaultSteerer(goldfive_steer_threshold="off")
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    drift = DriftEvent(
        kind=DriftKind.LOOPING_REASONING,
        severity=DriftSeverity.WARNING,
        detail="tool-loop detector fired",
        current_task_id="define_structure",
    )
    await steerer._handle_drift(drift, session)

    # Refine applied (plan swapped in).
    assert session.plan is revised
    # Nudge queued for consumption by the overlay.
    assert len(session.pending_nudges) == 1
    nudge = session.pending_nudges[0]
    assert "define_structure" in nudge

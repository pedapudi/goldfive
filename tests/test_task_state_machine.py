"""Unit tests for :class:`goldfive.task_state_machine.TaskStateMachine`.

Wave C of the steerer split (Issue #N/A internal). These tests exercise
the extracted TSM directly via a tiny stub router so the module's
contracts are pinned independently of the broader steerer surface:

* The ``mark_task_*`` family preserves the immutable-Plan swap pattern
  (goldfive#247) and stamps ``session.task_last_progress_at`` on every
  transition (goldfive#271).
* Terminal-task guard: a re-call on an already-terminal task is a no-op
  (no double-emission, no second cascade fan-out).
* ``cascade_cancel_downstream`` BFSs the forward DAG, skips
  already-terminal nodes, and emits one ``TaskCancelled`` + one
  ``TaskTransitioned`` per downstream task with the
  ``upstream_failed:<id>`` reason prefix (goldfive#205).
* Cross-component calls (``note_agent_activity``,
  ``_spawn_drift_handler_background``,
  ``_maybe_run_goal_drift_on_task_boundary``) route through the
  router's ``self._steerer`` back-reference and are observed by a
  minimal mock — TSM never reaches into the drift observer or plan
  reviser directly.

Coverage of the wider behaviour (drift cascade triggered by
``mark_task_failed``, the supersedes integration on PlanRevised, etc.)
lives in the router-level test files
(``test_steerer_*``, ``test_cancel_propagation``,
``test_intervention_ladder``, ``test_drift_async_dispatch``); those
keep exercising the same TSM module via the full ``DefaultSteerer``
surface.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.task_state_machine import TaskStateMachine  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)


class _StubDrift:
    """Tiny stub of :class:`DriftObserver` exposing only the methods
    :class:`TaskStateMachine` reaches for via ``self._steerer.drift``.
    """

    def __init__(self) -> None:
        self.note_agent_activity_calls: list[dict[str, Any]] = []
        self.spawn_drift_calls: list[tuple[DriftEvent, Session]] = []
        self.goal_drift_boundary_calls: int = 0
        self.resolve_terminal_calls: list[tuple[str, TaskStatus]] = []

    def note_agent_activity(
        self,
        session: Session,
        *,
        kind: str,
        agent_name: str = "",
        task_id: str = "",
        detail: str = "",
    ) -> None:
        self.note_agent_activity_calls.append(
            {
                "kind": kind,
                "agent_name": agent_name,
                "task_id": task_id,
                "detail": detail,
            }
        )

    def _spawn_drift_handler_background(
        self, drift: DriftEvent, session: Session
    ) -> None:
        self.spawn_drift_calls.append((drift, session))

    async def _maybe_run_goal_drift_on_task_boundary(self, session: Session) -> None:
        self.goal_drift_boundary_calls += 1

    async def resolve_conditions_for_terminal_task(
        self, session: Session, *, task_id: str, to_status: TaskStatus
    ) -> None:
        self.resolve_terminal_calls.append((task_id, to_status))


class _StubRouter:
    """Minimal mock router that satisfies :class:`TaskStateMachine`.

    Records every cross-component call so tests can assert the TSM
    routes them through the router back-reference (rather than
    importing the sibling components directly).
    """

    def __init__(self) -> None:
        self._sink = _ListSink()
        self._sinks = [self._sink]
        self._adapter: Any = None
        # Cross-component routing now goes through ``self.drift``;
        # forward the spy attributes via properties so existing tests
        # that read ``router.note_agent_activity_calls`` stay readable.
        self.drift = _StubDrift()

    @property
    def note_agent_activity_calls(self) -> list[dict[str, Any]]:
        return self.drift.note_agent_activity_calls

    @property
    def spawn_drift_calls(self) -> list[tuple[DriftEvent, Session]]:
        return self.drift.spawn_drift_calls

    @property
    def goal_drift_boundary_calls(self) -> int:
        return self.drift.goal_drift_boundary_calls

    def _new_envelope(self, session: Session) -> Any:
        from goldfive.events import new_event

        return new_event(session.run_id, session.next_sequence(), session_id=session.id)

    async def _emit(self, event_pb: Any) -> None:
        from goldfive.events import emit

        await emit(self._sinks, event_pb)


def _seed_session_with_plan() -> Session:
    plan = Plan(
        id="plan-1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="A", title="A", assignee_agent_id="alpha"),
            Task(id="B", title="B", assignee_agent_id="beta"),
            Task(id="C", title="C", assignee_agent_id="gamma"),
        ],
        edges=[
            TaskEdge(from_task_id="A", to_task_id="B"),
            TaskEdge(from_task_id="B", to_task_id="C"),
        ],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


@pytest.mark.asyncio
async def test_mark_task_running_swaps_plan_and_emits_transition() -> None:
    """``mark_task_running`` derives a new Plan and emits both proto events."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    original_plan_id = id(session.plan)

    await tsm.mark_task_running("A", session=session, detail="kickoff", source="llm_report")

    # goldfive#247: Plan is frozen — the swap produces a new instance.
    assert id(session.plan) != original_plan_id
    a = next(t for t in session.plan.tasks if t.id == "A")
    assert a.status is TaskStatus.RUNNING
    assert session.current_task_id == "A"
    # Liveness watermark stamped so the structural progress-stall gate
    # can distinguish iterating vs stuck tasks.
    assert "A" in session.task_last_progress_at
    # Lineage seeded with the assignee for downstream judge attribution.
    assert session.task_lineage["A"] == {"alpha"}
    # Two events: TaskStarted + TaskTransitioned.
    kinds = [evt.WhichOneof("payload") for evt in router._sink.events]
    assert kinds == ["task_started", "task_transitioned"]


@pytest.mark.asyncio
async def test_mark_task_completed_routes_through_router_notes_and_boundary() -> None:
    """Completion routes ``note_agent_activity`` + goal-drift boundary via router."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    await tsm.mark_task_running("A", session=session)
    router._sink.events.clear()

    await tsm.mark_task_completed(
        "A", session=session, summary="done", source="llm_report"
    )

    a = next(t for t in session.plan.tasks if t.id == "A")
    assert a.status is TaskStatus.COMPLETED
    assert session.completed_results["A"] == "done"
    # Lineage cleared on terminal.
    assert "A" not in session.task_lineage
    # The TSM must route through the router back-reference, NOT reach
    # for the drift observer directly.
    assert len(router.note_agent_activity_calls) == 1
    assert router.note_agent_activity_calls[0]["kind"] == "agent_invocation_completed"
    assert router.note_agent_activity_calls[0]["agent_name"] == "alpha"
    assert router.goal_drift_boundary_calls == 1


@pytest.mark.asyncio
async def test_mark_task_failed_spawns_drift_cascade_via_router() -> None:
    """Recoverable FAILED spawns the drift cascade via the router stub."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    await tsm.mark_task_running("A", session=session)

    await tsm.mark_task_failed(
        "A", session=session, reason="boom", recoverable=True
    )

    a = next(t for t in session.plan.tasks if t.id == "A")
    assert a.status is TaskStatus.FAILED
    # iter-11A — recoverable failure spawns the cascade fire-and-forget
    # through the router. Captured by the stub spy.
    assert len(router.spawn_drift_calls) == 1
    drift, spawned_session = router.spawn_drift_calls[0]
    assert drift.kind.value == "task_failed_recoverable"
    assert spawned_session is session
    # No cascade on a recoverable failure (cascade is only for
    # ``recoverable=False``).
    b = next(t for t in session.plan.tasks if t.id == "B")
    assert b.status is TaskStatus.PENDING


@pytest.mark.asyncio
async def test_mark_task_failed_unrecoverable_cascades_downstream() -> None:
    """Unrecoverable FAILED cascade-cancels the forward closure."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    await tsm.mark_task_running("A", session=session)

    await tsm.mark_task_failed(
        "A", session=session, reason="fatal", recoverable=False
    )

    a = next(t for t in session.plan.tasks if t.id == "A")
    b = next(t for t in session.plan.tasks if t.id == "B")
    c = next(t for t in session.plan.tasks if t.id == "C")
    assert a.status is TaskStatus.FAILED
    assert b.status is TaskStatus.CANCELLED
    assert c.status is TaskStatus.CANCELLED
    # The fatal drift is still spawned through the router; cascade is
    # an additional side effect.
    assert len(router.spawn_drift_calls) == 1
    assert router.spawn_drift_calls[0][0].kind.value == "task_failed_fatal"


@pytest.mark.asyncio
async def test_mark_task_running_is_idempotent_on_terminal() -> None:
    """Terminal-task guard prevents re-emission of transitions."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    await tsm.mark_task_running("A", session=session)
    await tsm.mark_task_completed("A", session=session)
    before = len(router._sink.events)
    note_count_before = len(router.note_agent_activity_calls)
    boundary_count_before = router.goal_drift_boundary_calls

    # Re-call after COMPLETED — must be a no-op.
    await tsm.mark_task_running("A", session=session, detail="zombie")
    await tsm.mark_task_completed("A", session=session)
    await tsm.mark_task_failed("A", session=session, reason="x")
    await tsm.mark_task_blocked("A", session=session, blocker="x")
    await tsm.mark_task_cancelled("A", session=session)
    await tsm.mark_task_not_needed("A", session=session)

    assert len(router._sink.events) == before
    assert len(router.note_agent_activity_calls) == note_count_before
    assert router.goal_drift_boundary_calls == boundary_count_before
    assert len(router.spawn_drift_calls) == 0


@pytest.mark.asyncio
async def test_cascade_cancel_downstream_emits_upstream_failed_reason() -> None:
    """Cascade uses the goldfive#205 structured reason on every downstream."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    await tsm.mark_task_running("A", session=session)
    router._sink.events.clear()

    await tsm.cascade_cancel_downstream(session, "A", source="cancellation")

    # Both B and C get cancelled via the BFS; the initiator (A) itself
    # is NOT transitioned by this primitive (caller-owned).
    b = next(t for t in session.plan.tasks if t.id == "B")
    c = next(t for t in session.plan.tasks if t.id == "C")
    assert b.status is TaskStatus.CANCELLED
    assert c.status is TaskStatus.CANCELLED
    # The TaskCancelled events carry the upstream_failed:<id> prefix.
    cancel_events = [
        evt for evt in router._sink.events if evt.WhichOneof("payload") == "task_cancelled"
    ]
    assert len(cancel_events) == 2
    for evt in cancel_events:
        assert evt.task_cancelled.reason == "upstream_failed:A"


@pytest.mark.asyncio
async def test_cascade_cancel_downstream_skips_terminal_dependents() -> None:
    """A diamond-DAG dependent in COMPLETED is preserved, not re-cancelled."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()
    # Drive B to COMPLETED out of band; the cascade from A must skip it
    # and stop the traversal (C depends only on B, so it stays PENDING).
    await tsm.mark_task_running("A", session=session)
    await tsm.mark_task_completed("A", session=session)
    await tsm.mark_task_running("B", session=session)
    await tsm.mark_task_completed("B", session=session)
    router._sink.events.clear()

    # Now cascade from a fresh hypothetical fatal on A. B is COMPLETED
    # so the BFS short-circuits; C must stay PENDING.
    await tsm.cascade_cancel_downstream(session, "A", source="cancellation")

    b = next(t for t in session.plan.tasks if t.id == "B")
    c = next(t for t in session.plan.tasks if t.id == "C")
    assert b.status is TaskStatus.COMPLETED  # preserved
    assert c.status is TaskStatus.PENDING  # not cascaded past COMPLETED ancestor


@pytest.mark.asyncio
async def test_mark_task_not_needed_does_not_cascade() -> None:
    """NOT_NEEDED is a per-task observation, not a downstream invalidation."""
    router = _StubRouter()
    tsm = TaskStateMachine(router)
    session = _seed_session_with_plan()

    await tsm.mark_task_not_needed("B", session=session, reason="superseded")

    b = next(t for t in session.plan.tasks if t.id == "B")
    c = next(t for t in session.plan.tasks if t.id == "C")
    assert b.status is TaskStatus.NOT_NEEDED
    assert c.status is TaskStatus.PENDING  # NOT_NEEDED does not cascade


@pytest.mark.asyncio
async def test_find_task_static_helper() -> None:
    """``_find_task`` matches by id and returns ``None`` for missing inputs."""
    session = _seed_session_with_plan()
    assert TaskStateMachine._find_task(session, "A").id == "A"
    assert TaskStateMachine._find_task(session, "missing") is None
    assert TaskStateMachine._find_task(session, "") is None
    session.plan = None
    assert TaskStateMachine._find_task(session, "A") is None

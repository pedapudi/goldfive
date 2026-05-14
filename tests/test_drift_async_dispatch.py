"""Async drift-cascade dispatch + activity-pairing (iter-11A + 11B).

Iter-11A: ``mark_task_failed`` / ``mark_task_blocked`` previously
awaited ``planner.refine`` inline through :meth:`_handle_drift`. On a
slow local LLM (e.g. Qwen3.6-35B-A3B-FP8) the resulting
``report_task_failed`` tool call took ~2 minutes to return, which
blocked the agent's next ADK turn end-to-end. The fix spawns the
cascade fire-and-forget on
:attr:`DefaultSteerer._background_drifts` and drains it at
:meth:`shutdown`. This module pins:

* The reporting tool ack returns immediately (within 100ms) regardless
  of how long ``planner.refine`` takes.
* :meth:`shutdown` drains pending drift cascades within the timeout.
* No running event loop → spawn no-ops without crashing (synchronous
  test harnesses).

Iter-11B: ``mark_task_completed`` / ``mark_task_failed`` now write a
synthetic ``agent_invocation_completed`` entry to
:attr:`Session.recent_agent_activity` so the GOAL_DRIFT judge does not
read an orphan-start + task-COMPLETED shape and false-positive on
"looping". The real ``after_run_callback`` will append another
``agent_invocation_completed`` slightly later — duplicate completed
entries are harmless (the goal-drift prompt renders both fine and the
ring buffer trims naturally).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

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
)

# ---------------------------------------------------------------------------
# Stubs (intentionally local to this file — keep the spec self-contained).
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _NullPlanner:
    """Refine returns ``None`` synchronously and instantly."""

    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return None


class _SlowPlanner:
    """Refine sleeps ``delay`` seconds before returning ``None``.

    Used to assert that the reporting tool returns BEFORE refine
    settles (the iter-11A correctness contract).
    """

    def __init__(self, *, delay: float) -> None:
        self.delay = delay
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_done = asyncio.Event()

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        try:
            await asyncio.sleep(self.delay)
            return None
        finally:
            self.refine_done.set()


def _make_session(*, assignee: str = "worker") -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", assignee_agent_id=assignee),
            Task(id="t2", title="B", assignee_agent_id=assignee),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


def _bind(planner: Any) -> tuple[DefaultSteerer, Session, _ListSink, Any]:
    steerer = DefaultSteerer()
    session = _make_session()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink, planner


# ---------------------------------------------------------------------------
# 11B — synthetic activity-pair entry
# ---------------------------------------------------------------------------


async def test_mark_task_completed_appends_synthetic_invocation_completed() -> None:
    """``mark_task_completed`` pairs a prior ``agent_invocation_started``.

    Reproduces the GOAL_DRIFT judge's "orphan start + task COMPLETED →
    looping" false positive. After the synthetic write the activity
    buffer has both the start AND a paired completed keyed on the
    task's ``assignee_agent_id``.
    """
    steerer, session, _sink, _planner = _bind(_NullPlanner())
    # Adapter would normally write this from ``before_run_callback``.
    steerer.drift.note_agent_activity(
        session,
        kind="agent_invocation_started",
        agent_name="worker",
        task_id="t1",
    )
    assert len(session.recent_agent_activity) == 1

    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")

    # Paired entry appended.
    completed_entries = [
        e for e in session.recent_agent_activity if e["kind"] == "agent_invocation_completed"
    ]
    assert len(completed_entries) == 1
    assert completed_entries[0]["agent_name"] == "worker"
    assert completed_entries[0]["task_id"] == "t1"


async def test_mark_task_failed_appends_synthetic_invocation_completed() -> None:
    """Same pairing applies on the failed-task path.

    Same false-positive shape: orphan start + task FAILED would also
    look like "looping" to the goal-drift judge.
    """
    steerer, session, _sink, _planner = _bind(_NullPlanner())
    steerer.drift.note_agent_activity(
        session,
        kind="agent_invocation_started",
        agent_name="worker",
        task_id="t1",
    )

    await steerer.tasks.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    await steerer.drift._wait_background_drifts_idle()

    completed_entries = [
        e for e in session.recent_agent_activity if e["kind"] == "agent_invocation_completed"
    ]
    assert len(completed_entries) == 1
    assert completed_entries[0]["agent_name"] == "worker"
    assert completed_entries[0]["task_id"] == "t1"


async def test_mark_task_completed_skips_pairing_when_no_assignee() -> None:
    """Tasks without an ``assignee_agent_id`` skip the synthetic write.

    The pairing keys on the agent name; with no assignee the entry
    would carry an empty agent name and pollute the goal-drift prompt
    without disambiguating anything.
    """
    steerer, session, _sink, _planner = _bind(_NullPlanner())
    # Wipe the assignee on the test session's tasks.
    # goldfive#247: Plan + Task are frozen — derive a new plan with
    # every task's assignee cleared.
    assert session.plan is not None
    from tests._immutable_plan_helpers import force_task_replace

    for t in list(session.plan.tasks):
        force_task_replace(session, t.id, assignee_agent_id="")
    # Pre-existing started entry from a prior agent's run.
    steerer.drift.note_agent_activity(
        session,
        kind="agent_invocation_started",
        agent_name="worker",
        task_id="t1",
    )

    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")

    completed_entries = [
        e for e in session.recent_agent_activity if e["kind"] == "agent_invocation_completed"
    ]
    assert completed_entries == []


async def test_mark_task_failed_skips_pairing_when_no_assignee() -> None:
    """Mirror of the completed-path "no assignee" guard."""
    steerer, session, _sink, _planner = _bind(_NullPlanner())
    assert session.plan is not None
    # goldfive#247: derive a new plan with every task's assignee cleared.
    from tests._immutable_plan_helpers import force_task_replace

    for t in list(session.plan.tasks):
        force_task_replace(session, t.id, assignee_agent_id="")

    await steerer.tasks.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    await steerer.drift._wait_background_drifts_idle()

    completed_entries = [
        e for e in session.recent_agent_activity if e["kind"] == "agent_invocation_completed"
    ]
    assert completed_entries == []


async def test_duplicate_completed_entries_are_harmless() -> None:
    """Real ``after_run_callback`` will also append a completed entry.

    The synthetic + real pair lands two completed entries in the
    activity buffer. Both are well-formed ``dict``s with the expected
    keys; the goal-drift prompt renders them fine; the ring buffer
    trims naturally on overflow.
    """
    steerer, session, _sink, _planner = _bind(_NullPlanner())
    steerer.drift.note_agent_activity(
        session,
        kind="agent_invocation_started",
        agent_name="worker",
        task_id="t1",
    )

    await steerer.tasks.mark_task_completed("t1", session=session, summary="done")
    # Simulate the real after_run_callback firing slightly later.
    steerer.drift.note_agent_activity(
        session,
        kind="agent_invocation_completed",
        agent_name="worker",
        task_id="t1",
    )

    completed_entries = [
        e for e in session.recent_agent_activity if e["kind"] == "agent_invocation_completed"
    ]
    assert len(completed_entries) == 2
    # Each entry carries the canonical keys; the goal-drift prompt
    # renderer reads ``kind``, ``agent_name``, ``task_id`` and is
    # tolerant of duplicates.
    for entry in completed_entries:
        assert entry["kind"] == "agent_invocation_completed"
        assert entry["agent_name"] == "worker"
        assert entry["task_id"] == "t1"


# ---------------------------------------------------------------------------
# 11A — async drift dispatch
# ---------------------------------------------------------------------------


async def test_mark_task_failed_does_not_block_on_refine() -> None:
    """``mark_task_failed`` returns immediately even when refine is slow.

    The iter-11A correctness contract: a 30s refine round-trip must
    NOT block the reporting-tool return. ``mark_task_failed`` returns
    in well under 100ms; refine eventually runs in the background and
    is observable via ``_wait_background_drifts_idle``.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer, session, _sink, _planner = _bind(slow_planner)

    t0 = time.monotonic()
    await steerer.tasks.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    elapsed = time.monotonic() - t0

    # 100ms is generous — the synchronous portion is two event emits +
    # a status flip. Real measured latency is sub-ms.
    assert elapsed < 0.1, (
        f"mark_task_failed blocked for {elapsed * 1000:.1f}ms; "
        "expected <100ms (iter-11A contract: cascade is fire-and-forget)"
    )
    # Refine HAS NOT settled yet — the slow planner is still asleep.
    assert slow_planner.refine_calls == [] or not slow_planner.refine_done.is_set()
    # Cancel the spawned task so we don't have to wait the full 30s.
    pending = list(steerer._background_drifts)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def test_mark_task_failed_refine_runs_eventually() -> None:
    """The refine round-trip ALWAYS lands — just asynchronously.

    Pairs with the latency assertion above: the cascade is offloaded,
    not dropped. Tests that need the post-cascade plan state await
    ``_wait_background_drifts_idle``.
    """
    planner = _NullPlanner()
    steerer, session, _sink, _ = _bind(planner)

    await steerer.tasks.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    # No refine call yet — handler is still queued.
    assert planner.refine_calls == []
    await steerer.drift._wait_background_drifts_idle()
    # After the drain refine WAS called.
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.TASK_FAILED_RECOVERABLE


async def test_mark_task_blocked_does_not_block_on_refine() -> None:
    """Same async contract for ``mark_task_blocked``.

    The reporting tool ``report_task_blocked`` shares the same code
    path; pin it explicitly so a future refactor that re-synchronises
    just one of the two methods is caught here.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer, session, _sink, _planner = _bind(slow_planner)

    t0 = time.monotonic()
    await steerer.tasks.mark_task_blocked("t1", session=session, blocker="missing input")
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1, (
        f"mark_task_blocked blocked for {elapsed * 1000:.1f}ms; expected <100ms (iter-11A contract)"
    )
    pending = list(steerer._background_drifts)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def test_shutdown_drains_background_drifts() -> None:
    """``shutdown`` cancels still-running drift cascades within timeout.

    Mirror of the existing ``_background_judges`` drain. A drift
    cascade that outlives its run must be cancelled at teardown so
    ``asyncio.create_task`` handles do not leak past the event
    loop's lifetime.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer, session, _sink, _ = _bind(slow_planner)

    await steerer.tasks.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    # One pending drift task should be tracked.
    assert len(steerer._background_drifts) == 1

    t0 = time.monotonic()
    # Bounded timeout — slow_planner.delay (30s) >> shutdown timeout.
    await steerer.drift.shutdown(timeout=0.2)
    elapsed = time.monotonic() - t0

    # Shutdown should land within the timeout window plus the 0.5s
    # post-cancel grace; not 30s.
    assert elapsed < 1.5, (
        f"shutdown blocked for {elapsed:.2f}s; expected <1.5s (timeout=0.2s + 0.5s cancel grace)"
    )
    # All tracked tasks discarded by their done-callbacks.
    assert steerer._background_drifts == set()


async def test_shutdown_idempotent_with_no_background_drifts() -> None:
    """``shutdown`` on an empty drift set is a no-op (matches judge drain)."""
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    await steerer.drift.shutdown(timeout=5.0)
    # Second call still safe.
    await steerer.drift.shutdown(timeout=5.0)


def test_mark_task_failed_returns_immediately_when_no_loop() -> None:
    """No running event loop → ``_spawn_drift_handler_background`` no-ops.

    Synchronous test harnesses (or one-shot CLIs) that build a steerer
    outside ``asyncio.run`` must not crash when a drift would
    otherwise be spawned. The inline cascade is skipped — those
    callers couldn't await it anyway, and the production path always
    has a running loop.

    This test deliberately runs sync (no ``async def``) and uses
    :meth:`asyncio.new_event_loop` only to drive the
    ``mark_task_failed`` coroutine to completion — the spawn path
    itself observes "no running loop" and short-circuits.
    """
    steerer, session, _sink, _ = _bind(_NullPlanner())
    drift = DriftEvent(
        kind=DriftKind.TASK_FAILED_RECOVERABLE,
        severity=DriftSeverity.WARNING,
        detail="test drift",
        current_task_id="t1",
    )
    # Direct test: call the spawner WITHOUT a running loop. It should
    # log+return rather than raise.
    steerer.drift._spawn_drift_handler_background(drift, session)
    assert steerer._background_drifts == set()
    # Sanity: the same call WITH a loop does spawn a task.
    loop = asyncio.new_event_loop()
    try:

        async def _spawn() -> None:
            steerer.drift._spawn_drift_handler_background(drift, session)
            assert len(steerer._background_drifts) == 1
            await steerer.drift._wait_background_drifts_idle()
            assert steerer._background_drifts == set()

        loop.run_until_complete(_spawn())
    finally:
        loop.close()


async def test_background_drift_swallows_handler_exception() -> None:
    """A flaky cascade must not crash the run.

    Mirrors :meth:`_run_goal_drift_judge_background`: handler raises
    → log warning, swallow, mark task complete on the tracking set.
    """
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    session = _make_session()
    drift = DriftEvent(
        kind=DriftKind.TASK_FAILED_RECOVERABLE,
        severity=DriftSeverity.WARNING,
        detail="test drift",
        current_task_id="t1",
    )

    async def _raise(_drift: DriftEvent, _session: Session) -> None:
        raise RuntimeError("simulated handler failure")

    # Patch _handle_drift on the instance for this one test.
    steerer.drift.handle_drift = _raise  # type: ignore[assignment]
    steerer.drift._spawn_drift_handler_background(drift, session)
    assert len(steerer._background_drifts) == 1
    await steerer.drift._wait_background_drifts_idle()
    assert steerer._background_drifts == set()

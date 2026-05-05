"""Drift / judge background-task drain at run boundary (goldfive#243).

The brussels-sprouts e2e leak
-----------------------------
On 2026-04-30 a brussels-sprouts e2e session leaked a JUSTIFIED_DEVIATION
drift cascade past its session's :class:`RunAborted`:

* 21:49:26 — ``_spawn_drift_handler_background`` for a JUSTIFIED_DEVIATION
  drift fires (research_sprouts, web_developer_agent).
* 21:49:28 — ADK's runner closes the inner agent invocation (NOT the
  goldfive Runner).
* ~21:50ish — :class:`SequentialExecutor` emits ``run_aborted`` with
  reason "review_slides failed".
* 21:59:26 — openai SDK retry log surfaces for the LEAKED refine's first
  LLM attempt.
* 21:59:59 — refine attempt 1/2 validator-rejected.
* 22:00:15 — refine attempt 2/2 validator-rejected.
* 22:00:15 — second HUMAN_INTERVENTION_REQUIRED emitted on a
  long-aborted session.

Root cause: the only drain for ``DefaultSteerer._background_drifts`` /
``_background_judges`` lived on :meth:`Steerer.shutdown`, which is
invoked from :meth:`Runner.close`. In adk-web's long-running server
mode the goldfive ``Runner`` persists across user turns and ``close()``
is NOT invoked between turns. So the per-turn drain never fired.

The fix: add :meth:`Steerer.drain_session_background_tasks` and call it
from the executor right before each terminal ``run_aborted_event`` /
``run_completed_event`` emission. The drain has the same bounded-wait
+ cancel-stragglers semantics as :meth:`shutdown`; idempotent; only
filters tasks tagged with the terminating session's id.

Tests below pin:

1. A drift task dispatched mid-run that is still in flight at run-end
   is cancelled BEFORE the executor's terminal emission — no
   post-abort ``DriftDetected`` lands on the sink bus.
2. The drain is idempotent: a second call shortly after the first is a
   no-op.
3. The drain is session-scoped: tasks tagged for a different session
   are NOT touched.
4. User-authored drifts (``USER_STEER`` / ``USER_CANCEL`` /
   ``USER_PAUSE``) are NOT drained — they dispatch synchronously
   through :meth:`_handle_drift` and never land on
   ``_background_drifts`` in the first place.
5. A second drain at the next run boundary on the same long-lived
   steerer correctly cancels NEW background tasks (multi-turn drain,
   not just last-run drain).
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

from goldfive.executors.sequential import (  # noqa: E402
    SequentialExecutor,
    _drain_steerer_at_run_boundary,
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
# Lightweight stubs (intentionally local to keep this spec self-contained).
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _SlowPlanner:
    """``refine`` blocks on a long sleep simulating a slow LLM round-trip.

    The brussels-sprouts symptom was a refine pinned on a stuck HTTP
    connection that the openai SDK retries silently after a long
    backoff. We simulate the "won't return on its own" property by
    sleeping for many seconds; the drain's bounded wait + cancellation
    is what lets the run boundary terminate cleanly.
    """

    def __init__(self, *, delay: float = 30.0) -> None:
        self.delay = delay
        self.refine_calls: list[dict[str, Any]] = []
        self.refine_started = asyncio.Event()
        self.refine_completed = False
        self.refine_cancelled = False

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        self.refine_started.set()
        try:
            await asyncio.sleep(self.delay)
            self.refine_completed = True
            return None
        except asyncio.CancelledError:
            self.refine_cancelled = True
            raise


def _make_session(*, session_id: str = "r1") -> Session:
    """Build a 2-task linear plan rooted at ``t1``.

    ``Session.id`` is a property that aliases ``run_id``
    (goldfive#155); we therefore set ``run_id=session_id`` so callers
    can address a session by a single stable string in tests.
    """
    plan = Plan(
        id="p1",
        run_id=session_id,
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", assignee_agent_id="worker"),
            Task(id="t2", title="B", assignee_agent_id="worker"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(
        run_id=session_id,
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_drain_cancels_in_flight_drift_at_run_boundary() -> None:
    """The leak scenario: drift dispatched mid-run + slow LLM + run abort.

    This is the brussels-sprouts case in miniature. A
    ``mark_task_failed`` mid-turn dispatches a fire-and-forget refine
    cascade through ``_spawn_drift_handler_background``. The slow
    planner pins the cascade on a long sleep. We then call the
    executor's run-boundary drain helper. The drain must:

    1. Settle within its bounded timeout (not wait the full ``delay``).
    2. Cancel the in-flight refine task.
    3. Empty ``_background_drifts`` so a subsequent
       :meth:`Runner.close`'s drain is a clean no-op.

    Pin-by-time: the drain timeout is 0.5s; the slow planner sleeps for
    30s. Without the cancel-stragglers branch the test would hang for
    ~30s.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=slow_planner)
    session = _make_session(session_id="s-leak")

    # Mid-run: a reporting tool fires mark_task_failed which dispatches
    # the cascade fire-and-forget on _background_drifts.
    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )
    assert len(steerer._background_drifts) == 1
    # Wait for refine to actually start (so the cancel-stragglers
    # branch — not the gather branch — is the path under test).
    await asyncio.wait_for(slow_planner.refine_started.wait(), timeout=2.0)

    t0 = time.monotonic()
    await steerer.drain_session_background_tasks(
        session_id="s-leak", timeout=0.5
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, (
        f"drain blocked for {elapsed:.2f}s; expected <1.5s "
        "(timeout=0.5s + 0.5s cancel grace)"
    )
    assert slow_planner.refine_cancelled is True
    assert slow_planner.refine_completed is False
    assert steerer._background_drifts == set()


async def test_drain_is_idempotent() -> None:
    """A second drain shortly after the first is a no-op.

    Required by the goldfive#243 brief: the drain must be safe to call
    repeatedly so callers don't have to think about which terminal
    branch fires it.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=slow_planner)
    session = _make_session(session_id="s-idem")

    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )
    await asyncio.wait_for(slow_planner.refine_started.wait(), timeout=2.0)

    await steerer.drain_session_background_tasks(
        session_id="s-idem", timeout=0.5
    )
    assert steerer._background_drifts == set()
    # Second call: empty set, must be a no-op (and must NOT raise).
    await steerer.drain_session_background_tasks(
        session_id="s-idem", timeout=0.5
    )
    assert steerer._background_drifts == set()


async def test_drain_is_session_scoped() -> None:
    """Tasks tagged for a different session are NOT cancelled.

    Concurrent ``runner.run`` calls share the steerer in some
    deployments (see ``tests/test_steerer_concurrent_sessions.py``).
    The drain must filter by ``session.id`` so terminating turn N's
    run does not cancel turn M's still-live cascade on a different
    session.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=slow_planner)
    session_a = _make_session(session_id="s-A")
    session_b = _make_session(session_id="s-B")

    await steerer.mark_task_failed(
        "t1", session=session_a, reason="boom", recoverable=True
    )
    await steerer.mark_task_failed(
        "t1", session=session_b, reason="boom", recoverable=True
    )
    assert len(steerer._background_drifts) == 2

    # Drain only session A. Session B's task must survive.
    await steerer.drain_session_background_tasks(
        session_id="s-A", timeout=0.5
    )
    surviving = list(steerer._background_drifts)
    assert len(surviving) == 1
    assert surviving[0].get_name().endswith(":s-B"), (
        f"unexpected surviving task name: {surviving[0].get_name()!r}"
    )

    # Cleanup: cancel session B's straggler.
    surviving[0].cancel()
    try:
        await asyncio.gather(*surviving, return_exceptions=True)
    except Exception:  # noqa: BLE001
        pass


async def test_drain_post_abort_no_drift_detected_emitted() -> None:
    """No NEW ``DriftDetected`` (the brussels-sprouts symptom) post-drain.

    The brussels symptom: a HUMAN_INTERVENTION_REQUIRED ``DriftDetected``
    was emitted AFTER the run had already aborted, because the leaked
    refine's validator-rejected outcome promoted it to a critical
    drift on a dead session. After the drain, no further
    ``drift_detected`` events from the cancelled cascade may appear
    on the sink bus.

    The refine's own paired ``refine_failed(cancelled)`` envelope
    (CANCELLATION-CONTRACT.md §C4) IS allowed to land — it is the
    correct observability marker that the refine was cancelled
    mid-flight, not the spurious post-abort drift the brussels run
    produced.
    """

    def _payload_kind(e: Any) -> str:
        if hasattr(e, "WhichOneof"):
            return e.WhichOneof("payload") or ""
        if isinstance(e, dict):
            return str(e.get("kind", ""))
        return getattr(e, "payload_kind", "") or ""

    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=slow_planner)
    session = _make_session(session_id="s-post-abort")

    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )
    await asyncio.wait_for(slow_planner.refine_started.wait(), timeout=2.0)

    pre_drain_drift_events = sum(
        1 for e in sink.events if _payload_kind(e) == "drift_detected"
    )

    await steerer.drain_session_background_tasks(
        session_id="s-post-abort", timeout=0.5
    )
    # The drain returned. Give the loop a couple of yields to let any
    # straggler emission attempt land if it were going to.
    for _ in range(5):
        await asyncio.sleep(0)

    post_drain_drift_events = sum(
        1 for e in sink.events if _payload_kind(e) == "drift_detected"
    )
    # The brussels failure mode was specifically a NEW DriftDetected
    # (the second HUMAN_INTERVENTION_REQUIRED at 22:00:15) emitted
    # post-abort by the leaked cascade. The drain must prevent that.
    assert post_drain_drift_events == pre_drain_drift_events, (
        f"drain leaked {post_drain_drift_events - pre_drain_drift_events} "
        "drift_detected event(s) after cancel; expected 0 (the brussels "
        "post-abort HUMAN_INTERVENTION_REQUIRED leak)"
    )
    # And no refine round-trip "completed" — the cancel killed it
    # mid-LLM-call (mocked here as mid-sleep).
    assert slow_planner.refine_completed is False


async def test_drain_does_not_touch_user_steer_drifts() -> None:
    """USER_STEER drifts dispatch synchronously and are NOT background tasks.

    The goldfive#243 brief insists user-authored drifts represent
    operator intent that should survive across turns. They flow
    through :meth:`_handle_drift` synchronously from :meth:`observe`,
    so they NEVER land on ``_background_drifts``. This test pins
    that invariant: emitting a synthetic USER_STEER drift directly
    through ``_handle_drift`` does not populate the background set.
    """
    steerer = DefaultSteerer()

    class _NullPlanner:
        async def generate(self, **kwargs: Any) -> Plan | None:
            return None

        async def refine(self, **kwargs: Any) -> Plan | None:
            return None

    steerer.bind(sinks=[_ListSink()], planner=_NullPlanner())
    session = _make_session(session_id="s-user")

    user_drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="please pivot",
        current_task_id="t1",
        authored_by="user",
    )
    # Direct path: _handle_drift is the route observe() takes for
    # user-authored drifts. The fact that it does NOT spawn anything
    # onto _background_drifts is the structural guarantee we want.
    await steerer._handle_drift(user_drift, session)

    # No background tasks were created — the USER_STEER cascade ran
    # synchronously inline.
    assert steerer._background_drifts == set()
    # Therefore the drain has nothing to do for this session — it
    # cannot accidentally cancel operator-authored intent.
    await steerer.drain_session_background_tasks(
        session_id="s-user", timeout=0.5
    )


async def test_drain_runs_at_every_run_boundary_not_just_last() -> None:
    """Multi-turn invariant: drain runs at EACH run boundary.

    Pin the brief's "drain runs at EVERY run boundary, not just the
    last" requirement. Simulates two consecutive turns on the SAME
    long-lived steerer (the adk-web pattern, where each user turn
    creates a fresh :class:`Session` but shares the Runner +
    Steerer). Each turn dispatches a drift; each turn's drain must
    catch its own drift.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=slow_planner)

    # Turn 1 — fresh session.
    session_turn_1 = _make_session(session_id="s-turn-1")
    await steerer.mark_task_failed(
        "t1", session=session_turn_1, reason="boom", recoverable=True
    )
    assert len(steerer._background_drifts) == 1
    await asyncio.wait_for(slow_planner.refine_started.wait(), timeout=2.0)
    await steerer.drain_session_background_tasks(
        session_id="s-turn-1", timeout=0.5
    )
    assert steerer._background_drifts == set()

    # Reset the started-event so we can wait on turn 2's refine entry.
    slow_planner.refine_started = asyncio.Event()

    # Turn 2 on the same steerer (adk-web shared-Runner pattern), new
    # session id. The drain at turn 1's boundary did not poison the
    # steerer's spawn path: a new drift dispatched here lands on the
    # set and is cancellable by turn 2's drain.
    session_turn_2 = _make_session(session_id="s-turn-2")
    await steerer.mark_task_failed(
        "t1", session=session_turn_2, reason="boom again", recoverable=True
    )
    assert len(steerer._background_drifts) == 1
    await asyncio.wait_for(slow_planner.refine_started.wait(), timeout=2.0)
    await steerer.drain_session_background_tasks(
        session_id="s-turn-2", timeout=0.5
    )
    assert steerer._background_drifts == set()


async def test_drain_empty_session_id_warns_and_no_ops() -> None:
    """Empty ``session_id`` is a caller bug; refuse to drain.

    Without filtering, an empty-suffix match would cancel every
    pending background task across every session. The drain refuses
    that and warns instead.
    """
    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=slow_planner)
    session = _make_session(session_id="s-empty-guard")

    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=True
    )
    assert len(steerer._background_drifts) == 1

    # Empty session_id: must NOT cancel the in-flight task.
    await steerer.drain_session_background_tasks(session_id="", timeout=0.5)
    assert len(steerer._background_drifts) == 1

    # Cleanup.
    pending = list(steerer._background_drifts)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def test_executor_drain_helper_is_duck_typed() -> None:
    """The executor's drain helper tolerates steerers without the method.

    Custom :class:`~goldfive.protocols.Steerer` implementations that
    pre-date goldfive#243 do not expose
    ``drain_session_background_tasks``. The helper's ``getattr`` +
    ``callable`` check must short-circuit cleanly.
    """

    class _LegacySteerer:
        """Has no drain_session_background_tasks method."""

        async def shutdown(self, *, timeout: float = 5.0) -> None:
            return None

    session = _make_session()
    # Should not raise.
    await _drain_steerer_at_run_boundary(_LegacySteerer(), session)


async def test_executor_drain_helper_swallows_steerer_exceptions() -> None:
    """A drain that raises must not block the executor's run termination.

    Defensive contract: the run-boundary drain is best-effort. A
    pathological steerer that raises during drain is logged and
    swallowed so the terminal ``run_aborted`` / ``run_completed``
    emission still lands.
    """

    class _RaisingSteerer:
        async def drain_session_background_tasks(
            self, *, session_id: str, timeout: float = 2.0
        ) -> None:
            raise RuntimeError("simulated drain failure")

    session = _make_session()
    # Should not raise.
    await _drain_steerer_at_run_boundary(_RaisingSteerer(), session)


# ---------------------------------------------------------------------------
# End-to-end through SequentialExecutor.run — exercises the actual call site
# wiring, not just the helper.
# ---------------------------------------------------------------------------


async def test_executor_run_aborted_drains_in_flight_drift_end_to_end() -> None:
    """Wire-level test: ``executor.run`` drains drift cascade before RunAborted.

    Builds a real :class:`SequentialExecutor` + :class:`DefaultSteerer`
    + :class:`_SlowPlanner` and drives a fail_fast=True abort. The
    drift dispatched via the on_invoke callback must be drained at
    the run boundary; ``run_aborted_event`` must still emit on the
    sink; the slow refine must NOT have completed.
    """
    from collections.abc import Awaitable, Callable

    from goldfive.protocols import EventSink
    from goldfive.results import InvocationResult

    slow_planner = _SlowPlanner(delay=30.0)
    steerer = DefaultSteerer()
    sink = _ListSink()

    class _Adapter:
        def __init__(
            self,
            *,
            on_invoke: Callable[
                [Task, Session], Awaitable[InvocationResult]
            ],
        ) -> None:
            self._on_invoke = on_invoke
            self.invocations: list[str] = []

        async def register_reporting_tools(self, tools: list[Any]) -> None:
            return None

        @property
        def available_agents(self) -> list[str]:
            return ["worker"]

        async def invoke(
            self, task: Task, session: Session
        ) -> InvocationResult:
            self.invocations.append(task.id)
            return await self._on_invoke(task, session)

    async def _fail_first(
        task: Task, session: Session
    ) -> InvocationResult:
        # Mid-invocation: a reporting tool would call mark_task_failed,
        # which spawns the fire-and-forget refine cascade on the slow
        # planner. fail_fast=True will then abort the run.
        await steerer.mark_task_failed(
            task.id, session=session, reason="boom", recoverable=False
        )
        # Give the spawned task a yield to actually start the refine.
        await asyncio.sleep(0)
        return InvocationResult(task_id=task.id, text="", stop_reason="failed")

    adapter = _Adapter(on_invoke=_fail_first)
    session = _make_session(session_id="s-e2e")
    plan = session.plan
    assert plan is not None

    executor = SequentialExecutor(max_task_invocations=5, fail_fast=True)
    sinks: list[EventSink] = [sink]  # type: ignore[list-item]

    t0 = time.monotonic()
    outcome = await executor.run(
        plan=plan,
        session=session,
        adapter=adapter,
        steerer=steerer,
        planner=slow_planner,
        sinks=sinks,
    )
    elapsed = time.monotonic() - t0

    assert outcome.success is False
    # The run must terminate within a small bounded window even though
    # the slow planner's refine sleeps for 30s. The drain's bounded
    # wait + cancel-stragglers branch is what makes this possible.
    assert elapsed < 5.0, (
        f"executor.run blocked for {elapsed:.2f}s; "
        "drain at run-boundary did not cancel slow refine"
    )
    # Terminal RunAborted DID emit on the sink.
    payload_kinds = []
    for e in sink.events:
        if hasattr(e, "WhichOneof"):
            payload_kinds.append(e.WhichOneof("payload") or "")
    assert "run_aborted" in payload_kinds, (
        f"no run_aborted on sink; saw kinds={payload_kinds}"
    )
    # The slow refine was cancelled, not completed.
    assert slow_planner.refine_completed is False
    # No background tasks survived past the run boundary.
    assert steerer._background_drifts == set()

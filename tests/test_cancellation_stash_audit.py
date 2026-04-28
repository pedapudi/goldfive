"""Phase 3.5 regression tests: CancelledError must not bypass state-stash duties.

goldfive#271 Phase 3 (PR #290) catalogued five sites
(:doc:`docs/design/CANCELLATION-CONTRACT.md` §C2-C6) where an
``except Exception:`` block wrapped an ``await`` and owned a state-stash
duty. Because :class:`asyncio.CancelledError` has been a
:class:`BaseException` subclass since Py 3.8, those broad catches did
NOT fire on cancellation — control flowed past the stash entirely,
leaving sinks with unmatched ``refine_attempted`` events, the executor
without a CRITICAL drift mirror to explain why the plan didn't change,
and (in the reporting tool path) pending corrections wedged on session
state for a task the agent had already acknowledged.

C1 (the runner.py:411 case) shipped its ``try/finally`` in PR #287.
Phase 3.5 closes the remaining C2-C6 sites:

* C2 — :class:`ParallelDAGExecutor._refine` (steerer-bound + legacy).
* C3 — :class:`SequentialExecutor` refines route through the steerer
  and inherit the C4 fix.
* C4 — :class:`DefaultSteerer._handle_drift`,
  :meth:`DefaultSteerer._promote_drift_to_steer`, and
  :meth:`DefaultSteerer.observe_refine`.
* C5 — :func:`goldfive.reporting._handle_task_started`'s
  ``mark_task_running`` await (correction GC must run on cancel).
* C6 — sink emits stamp the sequence cursor BEFORE the await, so the
  cursor is already advanced by the time ``CancelledError`` lands.
  (No conversion required — covered for completeness.)

Each test fires a real :class:`asyncio.CancelledError` at the audited
``await`` and asserts the stash invariant holds.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import reporting as _reporting  # noqa: E402
from goldfive._correction_injection import (  # noqa: E402
    pending_correction_key,
    write_correction,
)
from goldfive.adapters import _adk_state_protocol as _sp  # noqa: E402
from goldfive.executors.parallel import ParallelDAGExecutor  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Session,
    Task,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ListSink:
    """Sink that records every emitted event (proto + dict)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    @property
    def proto_events(self) -> list[Any]:
        return [e for e in self.events if hasattr(e, "WhichOneof")]

    @property
    def dict_events(self) -> list[dict[str, Any]]:
        return [e for e in self.events if isinstance(e, dict)]

    def by_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.dict_events if e.get("kind") == kind]

    def by_proto_oneof(self, name: str) -> list[Any]:
        return [e for e in self.proto_events if e.WhichOneof("payload") == name]


class _StubPlanner:
    """Minimal Planner duck-type satisfying ``DefaultSteerer.bind``."""

    def __init__(self, *, refine_impl: Any = None) -> None:
        self._refine_impl = refine_impl

    async def generate(self, **_: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        if self._refine_impl is None:
            return None
        return await self._refine_impl(**kwargs)

    def set_drift_emitter(self, emitter: Any) -> None:  # pragma: no cover
        pass

    def set_span_context_provider(self, provider: Any) -> None:
        self.provider = provider


def _drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="audit-fixture",
        current_task_id="t1",
    )


def _session() -> Session:
    return Session(run_id="r1", goals=[Goal(id="g1", summary="g")])


# ---------------------------------------------------------------------------
# C4 — DefaultSteerer.observe_refine: CancelledError must still emit
# the paired ``refine_failed`` so sinks see the attempted/failed pair.
# ---------------------------------------------------------------------------


async def test_c4_observe_refine_emits_refine_failed_on_cancellederror_and_reraises() -> None:
    """Phase 3.5 §C4: ``CancelledError`` (BaseException, not Exception)
    raised inside the ``observe_refine`` cm body MUST still emit a paired
    ``refine_failed`` envelope before propagating. Without the
    ``except BaseException`` branch on this cm the sink saw an unmatched
    ``refine_attempted``.
    """
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_StubPlanner())
    session = _session()
    drift = _drift()

    captured_attempt_id: str = ""
    with pytest.raises(asyncio.CancelledError):
        async with steerer.observe_refine(session, drift) as attempt_id:
            captured_attempt_id = attempt_id
            raise asyncio.CancelledError()

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    assert len(attempted) == 1, "attempted must land before the cancel"
    assert len(failed) == 1, (
        "Phase 3.5 §C4: refine_failed MUST be emitted even when the "
        "refine body raises CancelledError. Pre-fix the except Exception "
        "branch did NOT catch CancelledError, so the paired event was "
        "skipped and sinks saw an unmatched refine_attempted."
    )
    assert failed[0]["payload"]["attempt_id"] == captured_attempt_id
    assert failed[0]["payload"]["failure_kind"] == "cancelled"


async def test_c4_observe_refine_aclose_mid_refine_emits_failed() -> None:
    """Same invariant as above but exercised via the ``aclose()`` path
    on the inner generator: simulate ADK closing the runner mid-refine
    by cancelling the awaiting task.
    """
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_StubPlanner())
    session = _session()
    drift = _drift()

    started = asyncio.Event()
    captured: dict[str, str] = {}

    async def _refine_user() -> None:
        async with steerer.observe_refine(session, drift) as attempt_id:
            captured["id"] = attempt_id
            started.set()
            # Simulate a long-running refine the canceller will interrupt.
            await asyncio.sleep(60)

    task = asyncio.create_task(_refine_user())
    await started.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    attempted = sink.by_kind("refine_attempted")
    failed = sink.by_kind("refine_failed")
    assert len(attempted) == 1
    assert len(failed) == 1
    assert failed[0]["payload"]["attempt_id"] == captured["id"]
    assert failed[0]["payload"]["failure_kind"] == "cancelled"


# ---------------------------------------------------------------------------
# C4 — DefaultSteerer._handle_drift: planner.refine cancelled mid-flight
# emits the paired ``refine_failed`` AND lets CancelledError propagate.
# ---------------------------------------------------------------------------


async def test_c4_handle_drift_emits_refine_failed_when_refine_cancelled() -> None:
    """``_handle_drift`` wraps ``await planner.refine`` in
    ``try / except Exception``. ``CancelledError`` bypassed it; Phase
    3.5 adds the ``except BaseException`` branch that emits the paired
    ``refine_failed`` before re-raising."""
    sink = ListSink()
    steerer = DefaultSteerer()

    async def _cancelling_refine(**_: Any) -> Plan | None:
        raise asyncio.CancelledError()

    planner = _StubPlanner(refine_impl=_cancelling_refine)
    steerer.bind(sinks=[sink], planner=planner)

    session = _session()
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", assignee_agent_id="agent_a", status=TaskStatus.PENDING)
        ],
        edges=[],
    )
    drift = _drift()

    with pytest.raises(asyncio.CancelledError):
        await steerer._handle_drift(drift, session)

    failed = sink.by_kind("refine_failed")
    assert len(failed) >= 1, (
        "Phase 3.5 §C4: refine_failed MUST be emitted even when "
        "planner.refine raises CancelledError. Pre-fix the except "
        "Exception branch did NOT catch it, so the paired observability "
        "event was skipped."
    )
    assert any(f["payload"]["failure_kind"] == "cancelled" for f in failed)


# ---------------------------------------------------------------------------
# C2 — ParallelDAGExecutor._refine: CancelledError must still emit the
# CRITICAL DriftDetected mirror so operators see "refine cancelled".
# ---------------------------------------------------------------------------


async def test_c2_parallel_refine_cancelled_emits_critical_mirror_steerer_path() -> None:
    """Steerer-bound parallel-executor refine path: CancelledError
    bypassed the ``except Exception`` block so the CRITICAL drift
    mirror was skipped. Phase 3.5 adds the BaseException branch."""
    sink = ListSink()
    steerer = DefaultSteerer()

    async def _cancelling_refine(**_: Any) -> Plan | None:
        raise asyncio.CancelledError()

    planner = _StubPlanner(refine_impl=_cancelling_refine)
    steerer.bind(sinks=[sink], planner=planner)
    executor = ParallelDAGExecutor()

    session = _session()
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", assignee_agent_id="agent_a", status=TaskStatus.PENDING)
        ],
        edges=[],
    )
    drift = _drift()

    with pytest.raises(asyncio.CancelledError):
        await executor._refine(
            plan=session.plan,
            drift=drift,
            planner=planner,
            session=session,
            sinks=[sink],
            steerer=steerer,
        )

    # CRITICAL mirror is emitted as a DriftDetected proto event with
    # severity=CRITICAL, NOT a dict. The legacy sink-visible signal.
    drift_protos = sink.by_proto_oneof("drift_detected")
    critical = [e for e in drift_protos if "cancelled" in e.drift_detected.detail.lower()]
    assert critical, (
        "Phase 3.5 §C2: the CRITICAL refine-cancelled mirror MUST land "
        "even when planner.refine raises CancelledError. Pre-fix the "
        "except Exception branch was bypassed and operators saw the "
        "original drift but no follow-up explaining why the plan didn't "
        "change."
    )


async def test_c2_parallel_refine_cancelled_emits_critical_mirror_legacy_path() -> None:
    """Legacy (no steerer) parallel-executor refine path: same invariant.
    The legacy fallback runs without ``observe_refine`` and owns its
    own CRITICAL mirror; this branch must also fire on cancel."""
    sink = ListSink()
    executor = ParallelDAGExecutor()

    async def _cancelling_refine(**_: Any) -> Plan | None:
        raise asyncio.CancelledError()

    planner = _StubPlanner(refine_impl=_cancelling_refine)

    session = _session()
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="t1", assignee_agent_id="agent_a", status=TaskStatus.PENDING)
        ],
        edges=[],
    )
    drift = _drift()

    with pytest.raises(asyncio.CancelledError):
        # No ``steerer=`` kwarg → legacy refine path.
        await executor._refine(
            plan=session.plan,
            drift=drift,
            planner=planner,
            session=session,
            sinks=[sink],
        )

    drift_protos = sink.by_proto_oneof("drift_detected")
    critical = [e for e in drift_protos if "cancelled" in e.drift_detected.detail.lower()]
    assert critical, (
        "Phase 3.5 §C2 (legacy path): the CRITICAL refine-cancelled "
        "mirror must also fire on the no-steerer path; otherwise legacy "
        "executors silently swallow cancellation observability."
    )


# ---------------------------------------------------------------------------
# C3 — SequentialExecutor refine routes through the steerer; the C4
# fix above covers it. This guard keeps that contract honest by
# pinning that the steerer's refine entry points (which sequential.py
# delegates to) emit refine_failed on CancelledError. Already covered
# by ``test_c4_handle_drift_emits_refine_failed_when_refine_cancelled``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C5 — _handle_task_started: correction GC must run even if
# ``await steerer.mark_task_running`` is cancelled mid-flight.
# ---------------------------------------------------------------------------


class _CancellingSteererForTaskStarted:
    """Steerer stub whose ``mark_task_running`` raises CancelledError."""

    def __init__(self) -> None:
        self._sinks: list[Any] = []
        self.calls: list[str] = []

    async def mark_task_running(
        self,
        task_id: str,
        *,
        session: Session,
        detail: str = "",
        source: str = "",
    ) -> None:
        self.calls.append(task_id)
        raise asyncio.CancelledError()

    # Optional protocol surface used by the reporting handler:
    async def transition(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass

    async def _wait_plan_stable(self, _session: Session) -> None:
        return None


async def test_c5_clear_correction_runs_when_mark_running_cancelled() -> None:
    """Phase 3.5 §C5: ``_handle_task_started`` calls
    ``await steerer.mark_task_running`` and then synchronously
    ``_clear_correction_on_started``. Pre-fix the post-await clear was
    skipped on CancelledError, leaving the pending correction wedged
    on session state for a task the agent had just acknowledged.
    The fix wraps the await in ``try/finally`` so the GC runs on
    every exit including BaseException.
    """
    session = _session()
    session.plan = Plan(
        id="p1",
        run_id=session.run_id,
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t-correction",
                title="corrected task",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            )
        ],
        edges=[],
    )

    # Seed a pending correction the GC should clear.
    correction_key = pending_correction_key("writer", "t-correction")
    write_correction(
        session,
        {
            "agent_name": "writer",
            "task_id": "t-correction",
            "drift_kind": "off_topic",
            "drift_reason": "fixture",
            "revision_number": 1,
            "issued_at_ms": 0,
        },
    )
    assert correction_key in session.state, "fixture sanity"

    steerer = _CancellingSteererForTaskStarted()

    with pytest.raises(asyncio.CancelledError):
        await _reporting._handle_task_started(
            {"task_id": "t-correction"},
            session,
            steerer,  # type: ignore[arg-type]
        )

    assert steerer.calls == ["t-correction"], "the await must have been entered"
    assert correction_key not in session.state, (
        "Phase 3.5 §C5: the pending correction MUST be cleared even "
        "when mark_task_running raises CancelledError. Pre-fix the "
        "post-await ``_clear_correction_on_started`` call was skipped "
        "and the next turn re-injected an already-acknowledged correction."
    )


# ---------------------------------------------------------------------------
# C6 — sink emits stamp the sequence cursor BEFORE the await; the
# cursor advance is therefore not bypassed by CancelledError. We keep
# a small assertion here to pin that contract so a future refactor
# that moves ``next_sequence()`` after the ``await`` regresses loudly.
# ---------------------------------------------------------------------------


async def test_c6_sequence_cursor_advances_before_emit_await() -> None:
    """Phase 3.5 §C6: ``session.next_sequence()`` is called BEFORE the
    ``await emit(...)`` in every reporting helper that fires a sink
    event. Pinning this here means a future refactor that moves the
    cursor advance after the await would be caught by this test —
    that re-ordering would be a state-stash bypass.
    """
    session = _session()
    seq_before = session._next_sequence

    class _CancellingSink:
        async def emit(self, _event: Any) -> None:
            raise asyncio.CancelledError()

        async def close(self) -> None:  # pragma: no cover
            pass

    class _StubSteerer:
        def __init__(self, sinks: list[Any]) -> None:
            self._sinks = sinks

    sink = _CancellingSink()
    steerer = _StubSteerer([sink])

    with pytest.raises(asyncio.CancelledError):
        await _reporting._emit_task_transition_refused(
            session=session,
            steerer=steerer,  # type: ignore[arg-type]
            task_id="t1",
            attempted_from=TaskStatus.PENDING,
            attempted_to=TaskStatus.RUNNING,
            reason="cancelled-fixture",
            pin_revision=0,
            current_revision=0,
        )

    assert session._next_sequence > seq_before, (
        "Phase 3.5 §C6: next_sequence() must advance before the await, "
        "so a CancelledError mid-emit cannot leave the cursor in a "
        "rewound state. If this assertion regresses, a refactor moved "
        "the cursor advance after the sink await — which would be a "
        "real state-stash bypass."
    )

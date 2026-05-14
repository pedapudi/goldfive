"""Regression tests: ``DefaultSteerer`` ``_active_session`` must not leak
across concurrent sessions sharing one Steerer (PR #294 audit follow-up).

Empirical evidence motivating these tests
-----------------------------------------
A single :class:`~goldfive.steerer.DefaultSteerer` is shared on the
:class:`~goldfive.runner.Runner`, so two concurrent ``runner.run(...)``
calls for two different outer sessions race on the steerer's instance
state. Pre-fix, ``DefaultSteerer._active_session`` was a plain instance
attribute that got assigned at the start of each refine-bracketed block
(``_handle_drift``, ``_promote_drift_to_steer``, ``observe_refine``,
the user-steer state-application block) and cleared in a ``finally``.

When session A entered ``observe_refine`` and was suspended awaiting an
LLM round-trip, session B could enter its own ``observe_refine`` and
overwrite ``_active_session`` to point at session B. Session A's
planner-side callbacks (``_emit_planner_refine_validation_failed``,
``_span_context_for_planner``) then resolved to session B's run_id /
session_id / sink-list — exactly the leak observed in
``/tmp/demo-v14.log`` where v14's PlanRevised events were stamped with
v12's run_id.

The fix swaps ``_active_session`` for a per-instance
:class:`contextvars.ContextVar`. ``ContextVar.set`` returns a token that
``ContextVar.reset`` rolls back at block exit, AND the
``contextvars.copy_context``-style isolation that ``asyncio.gather``
applies to each child task means session A's value is invisible from
inside session B's task and vice-versa.

These tests pin that contract:

1. Two concurrent ``observe_refine`` blocks each see THEIR OWN session
   when the planner's span-context provider is invoked from inside the
   block — not the most-recently-set one.
2. The planner's drift-emitter callback
   (``_emit_planner_refine_validation_failed``) routes a planner-side
   drift to the calling task's session, not the other concurrent task's.
3. After both blocks exit, the ContextVar is reset to its default
   (``None``) — no stale pointer survives the bracketed window.

Failure mode pre-fix: tests 1 and 2 would intermittently see the OTHER
session's run_id under the ``asyncio.gather`` interleaving below — the
``await asyncio.sleep(0)`` yield points are sufficient for one task's
``self._active_session = session`` write to overwrite the other's
before the original task wakes up. With the ContextVar fix the writes
are per-task and the leak is impossible.
"""

from __future__ import annotations

import asyncio
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
)


class _ListSink:
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

    def proto_drifts(self) -> list[Any]:
        """Proto-envelope events whose oneof payload is ``drift_detected``."""
        return [e for e in self.proto_events if e.WhichOneof("payload") == "drift_detected"]


class _StubPlanner:
    """Minimal planner stub.

    ``set_span_context_provider`` captures the steerer's provider so the
    test can call it from inside the steerer's session-bracketed block
    (mimicking what the real planner does from inside ``planner.refine``
    when it builds a ``GoldfiveLLMCallStart`` span).

    ``set_drift_emitter`` captures the planner-side drift emitter the
    steerer wires into ``_emit_planner_refine_validation_failed`` --
    the callback whose run-id stamping was leaking pre-fix.
    """

    def __init__(self) -> None:
        self.provider: Any = None
        self.drift_emitter: Any = None

    async def generate(self, **_: Any) -> Plan | None:
        return None

    async def refine(self, **_: Any) -> Plan | None:
        return None

    def set_drift_emitter(self, emitter: Any) -> None:
        self.drift_emitter = emitter

    def set_span_context_provider(self, provider: Any) -> None:
        self.provider = provider


def _drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=DriftSeverity.WARNING,
        detail="d",
        current_task_id="t1",
    )


# ---------------------------------------------------------------------------
# Test 1 — span-context provider isolation across concurrent observe_refine
# ---------------------------------------------------------------------------


async def test_span_ctx_provider_isolated_across_concurrent_observe_refine() -> None:
    """Two ``observe_refine`` blocks running concurrently on one Steerer
    must each see their own session through the planner's span-context
    provider — not the other concurrent task's session.

    Pre-fix this fails under interleaving: whichever task assigned
    ``self._active_session`` last "wins" and both callbacks resolve to
    that session's run_id.
    """
    sink = _ListSink()
    steerer = DefaultSteerer()
    planner = _StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    session_a = Session(run_id="run-A", goals=[Goal(id="g", summary="ga")])
    session_b = Session(run_id="run-B", goals=[Goal(id="g", summary="gb")])
    drift = _drift()

    # Each task captures the run_id reported by the planner's
    # span-context provider FROM INSIDE its own observe_refine block,
    # AFTER yielding control to the event loop. The yield is what makes
    # the leak observable: it gives the OTHER task a chance to set its
    # own session pointer first. Pre-fix, both tasks would observe
    # whichever session entered the block last.
    captured: dict[str, str] = {}

    async def _refine_for(label: str, session: Session) -> None:
        async with steerer.plans.observe_refine(session, drift):
            # Yield so the other task can enter its own observe_refine
            # and (pre-fix) clobber _active_session.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            ctx = planner.provider()
            assert ctx is not None, f"{label}: span-ctx provider returned None inside block"
            _sinks_arg, run_id, session_id, _task_id, _seq_fn = ctx
            captured[f"{label}_run_id"] = run_id
            captured[f"{label}_session_id"] = session_id

    await asyncio.gather(
        _refine_for("A", session_a),
        _refine_for("B", session_b),
    )

    # Each task must have observed its own session, not the other's.
    assert captured["A_run_id"] == "run-A", (
        f"session A's planner span-ctx leaked: got run_id={captured['A_run_id']}, "
        "expected run-A; the steerer's _active_session was overwritten by "
        "session B's concurrent observe_refine block."
    )
    assert captured["B_run_id"] == "run-B", (
        f"session B's planner span-ctx leaked: got run_id={captured['B_run_id']}, expected run-B."
    )
    assert captured["A_session_id"] == session_a.id
    assert captured["B_session_id"] == session_b.id

    # After both blocks exit, the ContextVar must be reset to its
    # default — no stale pointer survives.
    assert planner.provider() is None


# ---------------------------------------------------------------------------
# Test 2 — planner drift emitter routes to the calling task's session
# ---------------------------------------------------------------------------


async def test_planner_drift_emitter_routes_to_calling_task_session() -> None:
    """When the planner emits a ``REFINE_VALIDATION_FAILED`` from inside
    one task's refine, the resulting ``DriftDetected`` envelope must be
    stamped with that task's session run_id — not the OTHER concurrent
    task's run_id.

    Pre-fix, ``_emit_planner_refine_validation_failed`` reads
    ``self._active_session`` directly; whichever task wrote it last
    determines the run_id on every concurrent emit, regardless of
    which task actually invoked the planner-side emitter.
    """
    sink = _ListSink()
    steerer = DefaultSteerer()
    planner = _StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)
    assert planner.drift_emitter is not None, "DefaultSteerer.bind must wire drift emitter"

    session_a = Session(run_id="run-A", goals=[Goal(id="g", summary="ga")])
    session_b = Session(run_id="run-B", goals=[Goal(id="g", summary="gb")])

    drift_a = DriftEvent(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        severity=DriftSeverity.WARNING,
        detail="A-side validation failed",
        current_task_id="task-A",
    )
    drift_b = DriftEvent(
        kind=DriftKind.REFINE_VALIDATION_FAILED,
        severity=DriftSeverity.WARNING,
        detail="B-side validation failed",
        current_task_id="task-B",
    )

    assert sink.proto_drifts() == []

    async def _refine_for(session: Session, drift: DriftEvent) -> None:
        async with steerer.plans.observe_refine(session, _drift()):
            # Yield to interleave with the other task's bracketed
            # window — pre-fix, this is where _active_session gets
            # overwritten.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Invoke the planner-side drift emitter from inside the
            # block. The callback reads the per-task ContextVar to
            # resolve which session the drift belongs to.
            await planner.drift_emitter(drift)

    await asyncio.gather(
        _refine_for(session_a, drift_a),
        _refine_for(session_b, drift_b),
    )

    # Two drift_detected proto envelopes on the wire, one per session.
    # We pair the drift to its session by detail string (A vs B) and
    # assert the run_id / session_id stamped on the envelope matches
    # the originating task.
    drift_events = sink.proto_drifts()
    assert len(drift_events) == 2, (
        f"expected 2 drift_detected proto envelopes, got {len(drift_events)}: "
        f"{[e.drift_detected.detail for e in drift_events]}"
    )

    by_detail: dict[str, Any] = {}
    for evt in drift_events:
        by_detail[evt.drift_detected.detail] = evt

    a_evt = by_detail.get("A-side validation failed")
    b_evt = by_detail.get("B-side validation failed")
    assert a_evt is not None, (
        f"A-side drift event missing: {[e.drift_detected.detail for e in drift_events]}"
    )
    assert b_evt is not None, (
        f"B-side drift event missing: {[e.drift_detected.detail for e in drift_events]}"
    )

    # The leak: pre-fix, BOTH events would carry whichever session was
    # last assigned to ``self._active_session``. Post-fix, the
    # ContextVar isolates per-task, so each event carries its own
    # run_id / session_id.
    assert a_evt.run_id == session_a.run_id, (
        f"A-side drift leaked to session B: run_id={a_evt.run_id}, "
        f"expected {session_a.run_id}. "
        "DefaultSteerer._active_session was overwritten by the concurrent "
        "session B observe_refine block before the planner's drift emitter "
        "ran."
    )
    assert b_evt.run_id == session_b.run_id, (
        f"B-side drift leaked to session A: run_id={b_evt.run_id}, expected {session_b.run_id}."
    )
    assert a_evt.session_id == session_a.id
    assert b_evt.session_id == session_b.id


# ---------------------------------------------------------------------------
# Test 3 — ContextVar resets to default after each block
# ---------------------------------------------------------------------------


async def test_active_session_resets_to_default_after_block() -> None:
    """After ``observe_refine`` exits cleanly, the planner's span-context
    provider must report ``None`` again — the ContextVar token must be
    reset, not just overwritten with a fresh ``set(None)`` (which would
    be a different leak vector if a parent context relied on a previous
    value).
    """
    sink = _ListSink()
    steerer = DefaultSteerer()
    planner = _StubPlanner()
    steerer.bind(sinks=[sink], planner=planner)

    session = Session(run_id="run-X", goals=[Goal(id="g", summary="g")])
    drift = _drift()

    assert planner.provider() is None

    async with steerer.plans.observe_refine(session, drift):
        ctx = planner.provider()
        assert ctx is not None
        _sinks_arg, run_id, _session_id, _task_id, _seq_fn = ctx
        assert run_id == "run-X"

    # Default restored.
    assert planner.provider() is None

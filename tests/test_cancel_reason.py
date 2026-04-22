"""Regression tests for goldfive#205 — structured cancel reasons.

Every ``TaskCancelled`` envelope should carry a structured ``reason``
string so downstream consumers (harmonograf's Trajectory view) can
answer "why was this task cancelled?" at a glance. The format is a
colon-prefixed tag followed by a provenance id or human context:

* ``upstream_failed:<upstream_task_id>`` — cascade from a FAILED /
  CANCELLED ancestor.
* ``run_aborted:<abort_reason>`` — orphan sweep at run-abort.
* ``user_cancel:<annotation_id>`` — user-initiated CANCEL control.
* ``user_steer:<annotation_id>`` — STEER-driven supersession of
  the in-flight task.
* ``superseded_by_revision:<replacement_id>`` — refine replaced
  this task.

These tests are the guard rails: every cancel emit site MUST stamp
one of these prefixes. An empty reason on a real cancel is a bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.control import AckResult, ControlAck, ControlKind, ControlMessage
from goldfive.executors._control import ControlOutcome, dispatch_control
from goldfive.steerer import DefaultSteerer
from goldfive.types import (
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Test fixtures (mirror those in tests/test_steerer.py in shape, kept local
# here so this file is self-contained).
# ---------------------------------------------------------------------------


def _make_plan(task_ids: tuple[str, ...]) -> Plan:
    tasks = [Task(id=tid, title=tid) for tid in task_ids]
    # Linear chain: t0 -> t1 -> t2 -> ...
    edges = [
        TaskEdge(from_task_id=task_ids[i], to_task_id=task_ids[i + 1])
        for i in range(len(task_ids) - 1)
    ]
    return Plan(id="p1", run_id="r1", goal_ids=[], tasks=tasks, edges=edges)


def _make_session(plan: Plan) -> Session:
    return Session(
        run_id="r1",
        plan=plan,
        goals=[Goal(id="g1", summary="test")],
    )


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


class _StubPlanner:
    """Returns None from refine so cascades run but refine does nothing."""

    async def refine(self, *, plan: Plan, drift: Any, goals: Any, **kw: Any) -> Plan | None:
        return None

    def set_drift_emitter(self, emitter: Any) -> None:
        return None


def _cancelled_reasons(sink: _ListSink) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in sink.events:
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "task_cancelled":
            out[e.task_cancelled.task_id] = e.task_cancelled.reason
    return out


# ---------------------------------------------------------------------------
# Cascade: upstream_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_failure_stamps_reason() -> None:
    """Fatal failure on t1 cascades to t2/t3 with ``upstream_failed:t1``."""
    steerer = DefaultSteerer()
    plan = _make_plan(("t1", "t2", "t3"))
    session = _make_session(plan)
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    await steerer.mark_task_failed(
        "t1", session=session, reason="boom", recoverable=False
    )

    reasons = _cancelled_reasons(sink)
    assert reasons == {
        "t2": "upstream_failed:t1",
        "t3": "upstream_failed:t1",
    }


@pytest.mark.asyncio
async def test_cancel_cascade_stamps_upstream_failed() -> None:
    """mark_task_cancelled cascades downstream with ``upstream_failed:<id>``."""
    steerer = DefaultSteerer()
    plan = _make_plan(("t1", "t2", "t3"))
    session = _make_session(plan)
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    await steerer.mark_task_cancelled(
        "t1", session=session, reason="user cancelled"
    )

    reasons = _cancelled_reasons(sink)
    # Initiator keeps the caller's reason verbatim (opaque passthrough).
    assert reasons["t1"] == "user cancelled"
    # Cascaded tasks carry the structured prefix.
    assert reasons["t2"] == "upstream_failed:t1"
    assert reasons["t3"] == "upstream_failed:t1"


# ---------------------------------------------------------------------------
# User CANCEL: user_cancel:<annotation_id>
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_cancel_dispatch_stamps_annotation_prefix() -> None:
    """A CANCEL ControlMessage produces a ``user_cancel:<annotation_id>`` prefix."""

    msg = ControlMessage(
        id="ctl-1",
        kind=ControlKind.CANCEL,
        payload={
            "reason": "user hit cancel",
            "annotation_id": "ann-abc",
        },
    )
    session = _make_session(_make_plan(("t1", "t2")))

    class _NoopSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            return None

        async def transition(self, *a: Any, **kw: Any) -> None:
            return None

    outcome = await dispatch_control(
        msg, session=session, steerer=_NoopSteerer(), sinks=[]
    )

    assert outcome.cancel_run is True
    assert outcome.cancel_reason == "user hit cancel"
    assert outcome.cancel_reason_prefix == "user_cancel:ann-abc"


@pytest.mark.asyncio
async def test_user_cancel_without_annotation_id_falls_back_to_control_id() -> None:
    """CANCEL without annotation_id still carries a non-empty prefix."""
    msg = ControlMessage(
        id="ctl-xyz",
        kind=ControlKind.CANCEL,
        payload={"reason": "abort"},
    )
    session = _make_session(_make_plan(("t1",)))

    class _NoopSteerer:
        async def observe(self, *a: Any, **kw: Any) -> None:
            return None

        async def transition(self, *a: Any, **kw: Any) -> None:
            return None

    outcome = await dispatch_control(
        msg, session=session, steerer=_NoopSteerer(), sinks=[]
    )

    # Falls back to the control-message id so the prefix is never empty.
    assert outcome.cancel_reason_prefix == "user_cancel:ctl-xyz"


# ---------------------------------------------------------------------------
# Transition kwarg: cancel_reason overrides detail for CANCELLED / FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transition_cancel_reason_kwarg_overrides_detail() -> None:
    """``transition(..., CANCELLED, detail=X, cancel_reason=Y)`` emits Y."""
    steerer = DefaultSteerer()
    plan = _make_plan(("t1",))
    session = _make_session(plan)
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    await steerer.transition(
        "t1",
        TaskStatus.CANCELLED,
        detail="human readable",
        cancel_reason="run_aborted:fail_fast",
        session=session,
    )

    reasons = _cancelled_reasons(sink)
    assert reasons["t1"] == "run_aborted:fail_fast"


@pytest.mark.asyncio
async def test_transition_falls_back_to_detail_when_no_cancel_reason() -> None:
    """Pre-#205 callers that only pass ``detail`` still work."""
    steerer = DefaultSteerer()
    plan = _make_plan(("t1",))
    session = _make_session(plan)
    sink = _ListSink()
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    await steerer.transition(
        "t1", TaskStatus.CANCELLED, detail="legacy", session=session
    )

    reasons = _cancelled_reasons(sink)
    assert reasons["t1"] == "legacy"


# ---------------------------------------------------------------------------
# ControlOutcome field hygiene — the dataclass carries the prefix field.
# ---------------------------------------------------------------------------


def test_control_outcome_has_cancel_reason_prefix_field() -> None:
    """Regression guard: the field the executors depend on exists by name."""
    fields = ControlOutcome.__dataclass_fields__
    assert "cancel_reason_prefix" in fields
    # Default is the empty string so existing constructions don't break.
    oc = ControlOutcome(
        ack=ControlAck(control_id="x", result=AckResult.SUCCESS)
    )
    assert oc.cancel_reason_prefix == ""

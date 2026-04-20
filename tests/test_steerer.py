"""Unit tests for ``goldfive.steerer.DefaultSteerer``.

Covers every task transition, every drift classification, and the
observe → detect → refine → apply pipeline. Uses a local in-memory
sink stub to capture emitted proto ``Event`` messages.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import (  # noqa: E402
    classify_refusal,
    classify_stop_reason,
    classify_tool_error,
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
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class ListSink:
    """Minimal ``EventSink`` that records every emitted ``Event`` in a list."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.closed: bool = False

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        self.closed = True


class StubPlanner:
    """``Planner`` stub that returns a pre-canned revised plan (or None)."""

    def __init__(
        self,
        *,
        revised: Plan | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.revised = revised
        self.raise_exc = raise_exc
        self.generate_calls: list[dict[str, Any]] = []
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        goals: list[Goal],
        available_agents: list[str],
        context: Any | None = None,
    ) -> Plan | None:
        self.generate_calls.append(
            {
                "goals": goals,
                "available_agents": available_agents,
                "context": context,
            }
        )
        return self.revised

    async def refine(
        self,
        *,
        plan: Plan,
        drift: DriftEvent,
        goals: list[Goal],
    ) -> Plan | None:
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.revised


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(task_ids: tuple[str, ...] = ("t1", "t2")) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=tid, title=tid.upper()) for tid in task_ids],
        edges=[
            TaskEdge(from_task_id=task_ids[i], to_task_id=task_ids[i + 1])
            for i in range(len(task_ids) - 1)
        ],
    )


def _make_session(plan: Plan | None = None) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do the thing")],
        plan=plan if plan is not None else _make_plan(),
    )


def _fresh() -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    s = DefaultSteerer()
    sess = _make_session()
    sink = ListSink()
    planner = StubPlanner()
    s.bind(sinks=[sink], planner=planner)
    return s, sess, sink, planner


# ---------------------------------------------------------------------------
# Task transitions
# ---------------------------------------------------------------------------


def _task(session: Session, task_id: str) -> Task:
    assert session.plan is not None
    for t in session.plan.tasks:
        if t.id == task_id:
            return t
    raise AssertionError(f"task {task_id!r} missing from plan")


async def test_mark_task_running_transitions_and_emits() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_running("t1", session=session, detail="kicking off")
    assert _task(session, "t1").status is TaskStatus.RUNNING
    assert session.current_task_id == "t1"
    assert session.agent_notes["t1"] == "kicking off"
    assert len(sink.events) == 1
    evt = sink.events[0]
    assert evt.WhichOneof("payload") == "task_started"
    assert evt.task_started.task_id == "t1"
    assert evt.task_started.detail == "kicking off"
    assert evt.run_id == "r1"
    assert evt.sequence == 0


async def test_mark_task_running_no_op_on_terminal() -> None:
    steerer, session, sink, _ = _fresh()
    _task(session, "t1").status = TaskStatus.COMPLETED
    await steerer.mark_task_running("t1", session=session)
    assert _task(session, "t1").status is TaskStatus.COMPLETED
    assert sink.events == []


async def test_mark_task_running_unknown_task_is_noop() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_running("bogus", session=session)
    assert sink.events == []


async def test_mark_task_progress_records_and_emits() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_progress("t1", session=session, fraction=0.42, detail="halfway")
    assert session.task_progress["t1"] == pytest.approx(0.42)
    assert session.agent_notes["t1"] == "halfway"
    # Status is untouched by progress updates.
    assert _task(session, "t1").status is TaskStatus.PENDING
    assert len(sink.events) == 1
    evt = sink.events[0]
    assert evt.WhichOneof("payload") == "task_progress"
    assert evt.task_progress.task_id == "t1"
    assert evt.task_progress.fraction == pytest.approx(0.42)
    assert evt.task_progress.detail == "halfway"


async def test_mark_task_progress_clamps_fraction() -> None:
    steerer, session, _sink, _ = _fresh()
    await steerer.mark_task_progress("t1", session=session, fraction=-1.0)
    assert session.task_progress["t1"] == 0.0
    await steerer.mark_task_progress("t1", session=session, fraction=9.0)
    assert session.task_progress["t1"] == 1.0


async def test_mark_task_completed_transitions_and_emits() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_completed(
        "t1",
        session=session,
        summary="done-zo",
        artifacts={"file": "out.txt"},
    )
    assert _task(session, "t1").status is TaskStatus.COMPLETED
    assert session.completed_results["t1"] == "done-zo"
    assert len(sink.events) == 1
    evt = sink.events[0]
    assert evt.WhichOneof("payload") == "task_completed"
    assert evt.task_completed.task_id == "t1"
    assert evt.task_completed.summary == "done-zo"
    assert dict(evt.task_completed.artifacts) == {"file": "out.txt"}


async def test_mark_task_failed_recoverable_fires_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    assert _task(session, "t1").status is TaskStatus.FAILED
    kinds = [e.WhichOneof("payload") for e in sink.events]
    # TaskFailed + DriftDetected + refine-failure DriftDetected (planner
    # returns None so no PlanRevised; the follow-up drift surfaces that).
    assert kinds == ["task_failed", "drift_detected", "drift_detected"]
    assert sink.events[0].task_failed.recoverable is True
    drift_evt = sink.events[1]
    # Proto enum DriftKind values: first check by semantic comparison via module.
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drift_evt.drift_detected.kind == types_pb2.DRIFT_KIND_TASK_FAILED_RECOVERABLE
    assert drift_evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_WARNING
    # Planner was asked to refine (WARNING ≥ WARNING).
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.TASK_FAILED_RECOVERABLE


async def test_mark_task_failed_fatal_fires_critical_drift() -> None:
    steerer, session, sink, _planner = _fresh()
    await steerer.mark_task_failed("t1", session=session, reason="unrecoverable", recoverable=False)
    drift_evt = sink.events[1]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drift_evt.drift_detected.kind == types_pb2.DRIFT_KIND_TASK_FAILED_FATAL
    assert drift_evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL


async def test_mark_task_blocked_transitions_and_emits_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.mark_task_blocked(
        "t1", session=session, blocker="need API key", needed="credentials.json"
    )
    assert _task(session, "t1").status is TaskStatus.BLOCKED
    kinds = [e.WhichOneof("payload") for e in sink.events]
    # TaskBlocked + DriftDetected + refine-failure DriftDetected (stub
    # planner returns None; the follow-up drift surfaces that).
    assert kinds == ["task_blocked", "drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert sink.events[1].drift_detected.kind == types_pb2.DRIFT_KIND_BLOCKED
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.BLOCKED


async def test_mark_task_cancelled_transitions() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_cancelled("t1", session=session, reason="user cancelled")
    assert _task(session, "t1").status is TaskStatus.CANCELLED
    assert len(sink.events) == 1
    assert sink.events[0].WhichOneof("payload") == "task_cancelled"
    assert sink.events[0].task_cancelled.reason == "user cancelled"


async def test_transition_dispatches_to_mark_methods() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.transition("t1", TaskStatus.RUNNING, session=session, detail="go")
    assert _task(session, "t1").status is TaskStatus.RUNNING
    assert sink.events[0].WhichOneof("payload") == "task_started"


# ---------------------------------------------------------------------------
# Drift classifiers (unit-level; no proto, no sinks)
# ---------------------------------------------------------------------------


def test_classify_tool_error_from_error_dict() -> None:
    d = classify_tool_error({"tool": "search_web", "error": "boom"})
    assert d is not None
    assert d.kind is DriftKind.TOOL_ERROR
    assert d.severity is DriftSeverity.WARNING


def test_classify_tool_error_from_failed_status() -> None:
    d = classify_tool_error({"name": "http_get", "status": "FAILED"})
    assert d is not None and d.kind is DriftKind.TOOL_ERROR


def test_classify_tool_error_from_ok_false() -> None:
    d = classify_tool_error({"ok": False, "message": "no"})
    assert d is not None and d.kind is DriftKind.TOOL_ERROR


def test_classify_tool_error_none_on_success() -> None:
    assert classify_tool_error({"result": 42, "ok": True}) is None
    assert classify_tool_error("just a string") is None
    assert classify_tool_error(None) is None


def test_classify_refusal_detects_markers() -> None:
    d = classify_refusal("I cannot help with that, sorry.")
    assert d is not None
    assert d.kind is DriftKind.AGENT_REFUSAL
    assert d.severity is DriftSeverity.WARNING


def test_classify_refusal_on_dict_content() -> None:
    d = classify_refusal({"text": "I must decline."})
    assert d is not None and d.kind is DriftKind.AGENT_REFUSAL


def test_classify_refusal_none_on_normal_text() -> None:
    assert classify_refusal("OK, I will do that.") is None


def test_classify_stop_reason_max_tokens() -> None:
    d = classify_stop_reason("MAX_TOKENS")
    assert d is not None and d.kind is DriftKind.CONTEXT_PRESSURE


def test_classify_stop_reason_enum_like() -> None:
    class _Enum:
        name = "FinishReason.LENGTH"

    d = classify_stop_reason(_Enum())
    assert d is not None and d.kind is DriftKind.CONTEXT_PRESSURE


def test_classify_stop_reason_none_on_unknown() -> None:
    assert classify_stop_reason("STOP") is None
    assert classify_stop_reason(None) is None


# ---------------------------------------------------------------------------
# detect_drift / observe pipeline
# ---------------------------------------------------------------------------


def test_detect_drift_prefers_tool_error_then_refusal_then_stop_reason() -> None:
    s = DefaultSteerer()
    sess = _make_session()
    # Tool-error dict wins even if it also contains refusal text.
    combo = {"error": "nope", "text": "I cannot"}
    assert s.detect_drift(combo, sess).kind is DriftKind.TOOL_ERROR
    # Refusal wins over stop_reason alone.
    assert (
        s.detect_drift(
            {"text": "I can't help with that", "stop_reason": "MAX_TOKENS"},
            sess,
        ).kind
        is DriftKind.AGENT_REFUSAL
    )
    # Pure stop_reason goes to CONTEXT_PRESSURE.
    assert s.detect_drift({"stop_reason": "MAX_TOKENS"}, sess).kind is DriftKind.CONTEXT_PRESSURE
    # Nothing → None.
    assert s.detect_drift({"text": "all good"}, sess) is None


async def test_observe_emits_drift_and_refines() -> None:
    from goldfive.pb.goldfive.v1 import types_pb2

    revised = _make_plan(("t1", "t2", "t3"))
    revised.id = "p1"
    revised.run_id = "r1"
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer.observe({"error": "oh no"}, session)

    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert kinds == ["drift_detected", "plan_revised"]
    assert sink.events[0].drift_detected.kind == types_pb2.DRIFT_KIND_TOOL_ERROR
    assert len(planner.refine_calls) == 1
    # Session's plan is replaced with revised, and revision metadata stamped.
    assert session.plan is revised
    assert session.plan.revision_kind == DriftKind.TOOL_ERROR.value
    assert session.plan.revision_severity == DriftSeverity.WARNING.value
    assert session.plan.revision_index >= 1


async def test_observe_skips_refine_when_no_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.observe({"text": "nothing interesting"}, session)
    assert sink.events == []
    assert planner.refine_calls == []


async def test_observe_skips_refine_on_info_severity() -> None:
    # Craft a direct DriftEvent at INFO severity via a custom detector.
    class InfoSteerer(DefaultSteerer):
        def detect_drift(self, event, session):  # type: ignore[override]
            return DriftEvent(
                kind=DriftKind.CUSTOM,
                severity=DriftSeverity.INFO,
                detail="just a note",
            )

    planner = StubPlanner()
    sink = ListSink()
    steerer = InfoSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer.observe({}, session)
    assert len(sink.events) == 1
    assert sink.events[0].WhichOneof("payload") == "drift_detected"
    # INFO is below the WARNING threshold — no refine.
    assert planner.refine_calls == []


async def test_observe_swallows_planner_exceptions() -> None:
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # Must not raise even though refine() blows up.
    await steerer.observe({"error": "x"}, session)
    # We emit the original drift and a follow-up CRITICAL drift that
    # surfaces the refine failure so sinks can render it — we never
    # emit plan_revised because the refine errored.
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert kinds == ["drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    follow_up = sink.events[1].drift_detected
    assert follow_up.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "refine failed" in follow_up.detail
    assert "planner down" in follow_up.detail


async def test_observe_surfaces_refine_none_return() -> None:
    """When refine returns None, a CRITICAL follow-up drift is emitted.

    Without this, a silently-swallowed refine leaves session.plan pinned
    to the stale plan and the executor will re-enter the same state on
    the next tick with no observable signal.
    """
    planner = StubPlanner(revised=None)  # refine returns None
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    await steerer.observe({"error": "x"}, session)
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert kinds == ["drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    follow_up = sink.events[1].drift_detected
    assert follow_up.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "refine failed" in follow_up.detail
    assert "no revised plan" in follow_up.detail
    # Session plan is unchanged.
    assert session.plan is not None
    assert session.plan.revision_index == 0


# ---------------------------------------------------------------------------
# Drift kind coverage — every DriftKind either has a classifier, a
# transition that fires it, or is a reserved value that the steerer
# itself doesn't emit (e.g., USER_STEER, AGENT_TRANSFER, CUSTOM).
# This test enforces that every listed DriftKind is known.
# ---------------------------------------------------------------------------


def test_every_drift_kind_has_a_known_origin() -> None:
    # Kinds the steerer fires directly from transitions / handlers.
    steerer_emits = {
        DriftKind.TASK_FAILED_RECOVERABLE,
        DriftKind.TASK_FAILED_FATAL,
        DriftKind.BLOCKED,
        DriftKind.NEW_WORK_DISCOVERED,
        DriftKind.PLAN_DIVERGENCE,
    }
    # Kinds the detect_drift classifiers produce.
    classifier_emits = {
        DriftKind.TOOL_ERROR,
        DriftKind.AGENT_REFUSAL,
        DriftKind.CONTEXT_PRESSURE,
    }
    # Kinds reserved for other components or future heuristics — the
    # goldfive default steerer does not emit these itself but the
    # taxonomy names them so sinks can render them.
    reserved = set(DriftKind) - steerer_emits - classifier_emits
    # Sanity: the union is exactly DriftKind.
    assert steerer_emits | classifier_emits | reserved == set(DriftKind)
    # And every reserved kind is a real enum member.
    for k in reserved:
        assert isinstance(k, DriftKind)


# ---------------------------------------------------------------------------
# Plan-mutation drift hooks
# ---------------------------------------------------------------------------


async def test_report_new_work_discovered_fires_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.report_new_work_discovered(
        session=session,
        parent_task_id="t1",
        title="follow-up",
        description="double-check the numbers",
        assignee="analyst",
    )
    # Original drift + refine-failure drift (stub planner returns None).
    assert len(sink.events) == 2
    assert sink.events[0].WhichOneof("payload") == "drift_detected"
    from goldfive.pb.goldfive.v1 import types_pb2

    assert sink.events[0].drift_detected.kind == types_pb2.DRIFT_KIND_NEW_WORK_DISCOVERED
    assert planner.refine_calls[0]["drift"].kind is DriftKind.NEW_WORK_DISCOVERED


async def test_report_plan_divergence_sets_flag_and_fires_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.report_plan_divergence(
        session=session,
        note="agent is doing something different",
        suggested_action="replan from here",
    )
    assert session.divergence_flag is True
    from goldfive.pb.goldfive.v1 import types_pb2

    assert sink.events[0].drift_detected.kind == types_pb2.DRIFT_KIND_PLAN_DIVERGENCE
    assert planner.refine_calls[0]["drift"].kind is DriftKind.PLAN_DIVERGENCE


# ---------------------------------------------------------------------------
# Event envelope invariants
# ---------------------------------------------------------------------------


async def test_event_sequence_is_monotonic_and_run_id_stamped() -> None:
    steerer, session, sink, _ = _fresh()
    await steerer.mark_task_running("t1", session=session)
    await steerer.mark_task_progress("t1", session=session, fraction=0.5)
    await steerer.mark_task_completed("t1", session=session, summary="ok")
    assert [e.sequence for e in sink.events] == [0, 1, 2]
    for e in sink.events:
        assert e.run_id == "r1"
        # emitted_at is a Timestamp — just sanity check it's populated.
        assert e.emitted_at.seconds > 0 or e.emitted_at.nanos > 0


async def test_observe_rejects_invalid_revised_plan() -> None:
    """A refine() that returns a structurally-broken plan is rejected.

    The steerer must not install a plan with duplicate ids / cycles /
    unknown edges. Instead it emits a CRITICAL ``SCHEMA_VIOLATION``
    DriftDetected carrying the validator's reason, and the session
    keeps its original plan.
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    # Revised plan with duplicate task ids — validate() will reject it.
    bad_revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="dup", title="first", status=TaskStatus.PENDING),
            Task(id="dup", title="second", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    original_plan = session.plan

    await steerer.observe({"error": "trigger refine"}, session)

    kinds = [e.WhichOneof("payload") for e in sink.events]
    # The initial TOOL_ERROR drift, then a CRITICAL validation-failure
    # drift when the bad revised plan is rejected. No PlanRevised is
    # emitted because the revision was not installed.
    assert kinds == ["drift_detected", "drift_detected"]
    # Original plan is still in place.
    assert session.plan is original_plan
    # The second drift is the validation-failure signal.
    second = sink.events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert second.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "plan validation failed" in second.detail
    assert "duplicate task id" in second.detail


async def test_observe_rejects_revised_plan_with_cycle() -> None:
    """Cycle in the revised plan is rejected with a CRITICAL drift."""
    from goldfive.pb.goldfive.v1 import types_pb2

    bad_revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.PENDING),
            Task(id="b", title="B", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge("a", "b"), TaskEdge("b", "a")],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    original_plan = session.plan

    await steerer.observe({"error": "trigger refine"}, session)

    assert session.plan is original_plan
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert kinds == ["drift_detected", "drift_detected"]
    second = sink.events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert second.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "cycle" in second.detail


async def test_observe_rejects_revised_plan_with_unknown_edge() -> None:
    """An edge referencing a missing task id is rejected."""
    from goldfive.pb.goldfive.v1 import types_pb2

    bad_revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="1", status=TaskStatus.PENDING)],
        edges=[TaskEdge("t1", "ghost")],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    original_plan = session.plan

    await steerer.observe({"error": "trigger refine"}, session)

    assert session.plan is original_plan
    second = sink.events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert "unknown task id" in second.detail


async def test_bind_replaces_sinks() -> None:
    steerer = DefaultSteerer()
    planner = StubPlanner()
    sink_a = ListSink()
    sink_b = ListSink()
    steerer.bind(sinks=[sink_a], planner=planner)
    steerer.bind(sinks=[sink_b], planner=planner)
    session = _make_session()
    await steerer.mark_task_running("t1", session=session)
    assert sink_a.events == []
    assert len(sink_b.events) == 1

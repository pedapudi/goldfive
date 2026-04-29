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

    @property
    def proto_events(self) -> list[Any]:
        """Return only the per-status proto ``Event`` envelopes recorded.

        goldfive a4 added dict-envelope events (``refine_attempted`` /
        ``refine_failed`` / correlation ``plan_revised``) on the same
        sink fan-out. goldfive#251 R4 added ``task_transitioned`` proto
        envelopes — observability-only, attached after every per-status
        envelope. Tests that assert on proto event order can use
        this filter to ignore both the dict sidecars and the R4
        observability events while still exercising
        the production emit path.
        """
        out: list[Any] = []
        for e in self.events:
            which = getattr(e, "WhichOneof", None)
            if which is None:
                continue
            try:
                if which("payload") == "task_transitioned":
                    continue
            except Exception:
                pass
            out.append(e)
        return out

    @property
    def dict_events(self) -> list[dict[str, Any]]:
        """Return only the dict-envelope events recorded (a4 observability)."""
        return [e for e in self.events if isinstance(e, dict)]


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
    # task_started + the goldfive#251 R4 task_transitioned observability
    # envelope.
    started = [e for e in sink.events if e.WhichOneof("payload") == "task_started"]
    assert len(started) == 1
    evt = started[0]
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
    # progress is a liveness tick; no transition event from R4.
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
    completed = [
        e for e in sink.events if e.WhichOneof("payload") == "task_completed"
    ]
    assert len(completed) == 1
    evt = completed[0]
    assert evt.task_completed.task_id == "t1"
    assert evt.task_completed.summary == "done-zo"
    assert dict(evt.task_completed.artifacts) == {"file": "out.txt"}


async def test_mark_task_failed_recoverable_fires_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.mark_task_failed("t1", session=session, reason="boom", recoverable=True)
    assert _task(session, "t1").status is TaskStatus.FAILED
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    # TaskFailed + DriftDetected + refine-failure DriftDetected (planner
    # returns None so no PlanRevised; the follow-up drift surfaces that).
    assert kinds == ["task_failed", "drift_detected", "drift_detected"]
    assert sink.proto_events[0].task_failed.recoverable is True
    drift_evt = sink.proto_events[1]
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
    # Event order: TaskFailed(t1) → TaskCancelled(t2, cascade) → DriftDetected(TASK_FAILED_FATAL) →
    # refine-failure DriftDetected (stub planner returns None). The
    # cascade fires before the fatal drift so planner.refine (if it ran)
    # would see the post-cascade plan shape.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["task_failed", "task_cancelled", "drift_detected", "drift_detected"]
    assert sink.proto_events[0].task_failed.task_id == "t1"
    assert sink.proto_events[0].task_failed.recoverable is False
    # Downstream cascade picked up t2 via the shared primitive.
    assert sink.proto_events[1].task_cancelled.task_id == "t2"
    # goldfive#205: cascade reason is structured as ``upstream_failed:<source_id>``
    # so harmonograf's Trajectory view can render "why was this task cancelled?".
    assert sink.proto_events[1].task_cancelled.reason == "upstream_failed:t1"
    drift_evt = sink.proto_events[2]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert drift_evt.drift_detected.kind == types_pb2.DRIFT_KIND_TASK_FAILED_FATAL
    assert drift_evt.drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL


async def test_mark_task_blocked_transitions_and_emits_drift() -> None:
    steerer, session, sink, planner = _fresh()
    await steerer.mark_task_blocked(
        "t1", session=session, blocker="need API key", needed="credentials.json"
    )
    assert _task(session, "t1").status is TaskStatus.BLOCKED
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    # TaskBlocked + DriftDetected + refine-failure DriftDetected (stub
    # planner returns None; the follow-up drift surfaces that).
    assert kinds == ["task_blocked", "drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    assert sink.proto_events[1].drift_detected.kind == types_pb2.DRIFT_KIND_BLOCKED
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.BLOCKED


async def test_mark_task_cancelled_transitions() -> None:
    steerer, session, sink, _ = _fresh()
    # The default fixture plan has t1 -> t2; the cascade emits a
    # TaskCancelled for t2 as well. Verify both the primary transition
    # and the cascaded one; detailed cascade behaviour is exercised in
    # the dedicated cascade tests below.
    await steerer.mark_task_cancelled("t1", session=session, reason="user cancelled")
    assert _task(session, "t1").status is TaskStatus.CANCELLED
    assert _task(session, "t2").status is TaskStatus.CANCELLED
    cancelled = [
        e for e in sink.events if e.WhichOneof("payload") == "task_cancelled"
    ]
    assert len(cancelled) == 2
    assert cancelled[0].task_cancelled.task_id == "t1"
    assert cancelled[0].task_cancelled.reason == "user cancelled"
    assert cancelled[1].task_cancelled.task_id == "t2"


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


@pytest.mark.parametrize(
    ("text", "expected_severity"),
    [
        # CRITICAL — policy / safety refusals.
        ("I must decline to answer this.", DriftSeverity.CRITICAL),
        ("I cannot assist with that request.", DriftSeverity.CRITICAL),
        ("That request is against my guidelines.", DriftSeverity.CRITICAL),
        ("For safety reasons, I won't elaborate.", DriftSeverity.CRITICAL),
        ("I will not proceed with this task.", DriftSeverity.CRITICAL),
        # WARNING — capability / scope refusals.
        ("I cannot help with that, sorry.", DriftSeverity.WARNING),
        ("That is beyond my capabilities right now.", DriftSeverity.WARNING),
        ("That's outside my scope as a planner.", DriftSeverity.WARNING),
        ("I was unable to find the file.", DriftSeverity.WARNING),
        ("No viable approach presents itself.", DriftSeverity.WARNING),
        ("This is not something I can do.", DriftSeverity.WARNING),
        # INFO — hedging / deferral.
        ("I may not be the best fit for this.", DriftSeverity.INFO),
        ("I think this might be the wrong approach.", DriftSeverity.INFO),
        ("I'm not particularly well suited for image work.", DriftSeverity.INFO),
        ("I'm not confident this will succeed.", DriftSeverity.INFO),
    ],
)
def test_classify_refusal_severity_tiers(
    text: str, expected_severity: DriftSeverity
) -> None:
    """Each tier maps to its documented ``DriftSeverity``."""
    d = classify_refusal(text)
    assert d is not None
    assert d.kind is DriftKind.AGENT_REFUSAL
    assert d.severity is expected_severity


def test_classify_refusal_critical_wins_over_warning_substring() -> None:
    """First-match-wins across tiers: a CRITICAL marker in the same
    text must not be downgraded because a WARNING marker also
    happens to appear.
    """
    # "i cannot" (WARNING) appears before "i must decline" (CRITICAL)
    # in the raw string, but the scan order is tier-first, not
    # position-first, so CRITICAL must win.
    text = "I cannot continue here; I must decline on policy grounds."
    d = classify_refusal(text)
    assert d is not None
    assert d.kind is DriftKind.AGENT_REFUSAL
    assert d.severity is DriftSeverity.CRITICAL


def test_classify_refusal_warning_wins_over_info_substring() -> None:
    """WARNING tier beats INFO when both markers coexist."""
    text = "I'm not confident, and in fact I cannot do this."
    d = classify_refusal(text)
    assert d is not None
    assert d.kind is DriftKind.AGENT_REFUSAL
    assert d.severity is DriftSeverity.WARNING


def test_classify_refusal_flat_markers_back_compat() -> None:
    """The deprecated flat ``LLM_REFUSAL_MARKERS`` tuple still exposes
    every tiered marker so external callers that imported the name
    continue to work.
    """
    from goldfive.drift import (
        LLM_REFUSAL_MARKERS,
        LLM_REFUSAL_MARKERS_CRITICAL,
        LLM_REFUSAL_MARKERS_INFO,
        LLM_REFUSAL_MARKERS_WARNING,
    )

    expected = set(
        LLM_REFUSAL_MARKERS_CRITICAL
        + LLM_REFUSAL_MARKERS_WARNING
        + LLM_REFUSAL_MARKERS_INFO
    )
    assert set(LLM_REFUSAL_MARKERS) == expected


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

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
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
    # First failure: emit original drift + refine-failure visibility
    # drift. Counter bumps to 1 (below the threshold), so we stay in
    # visibility-only mode — no REPEATED_FAILURE, no mark_task_failed.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    follow_up = sink.proto_events[1].drift_detected
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
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "drift_detected"]
    from goldfive.pb.goldfive.v1 import types_pb2

    follow_up = sink.proto_events[1].drift_detected
    assert follow_up.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "refine failed" in follow_up.detail
    assert "no revised plan" in follow_up.detail
    # Session plan is unchanged.
    assert session.plan is not None
    assert session.plan.revision_index == 0


# ---------------------------------------------------------------------------
# Refine-failure retry backoff (TASK-LIFECYCLE.md §7.3)
# ---------------------------------------------------------------------------


def _tool_error_drift(
    task_id: str = "t1",
    *,
    severity: DriftSeverity = DriftSeverity.WARNING,
    agent_id: str = "",
) -> DriftEvent:
    """Build a canned WARNING-severity drift routed through _handle_drift.

    ``severity`` and ``agent_id`` are parameterised so callers exercising
    the per-condition refine gate (goldfive I5) can mint a DISTINCT
    drift condition on a re-emit. The gate keys on
    ``(kind, task, agent, turn)`` — without varying one of these, the
    second emit collapses onto the same condition_id and is treated as
    a no-op replay (skipped). The failure-counter, by contrast, keys
    only on ``(kind, task)`` so the assertions below still trip.
    """
    return DriftEvent(
        kind=DriftKind.TOOL_ERROR,
        severity=severity,
        detail="simulated tool error",
        current_task_id=task_id,
        current_agent_id=agent_id,
    )


async def test_refine_failure_counter_increments_on_exception() -> None:
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")
    assert session.refine_failure_counts.get(key, 0) == 0

    # Same drift is classified by detect_drift — use a dict the tool-error
    # classifier picks up and that carries a task via current_task_id.
    # Simpler: drive _handle_drift directly so the drift identity is stable.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-a"), session)
    assert session.refine_failure_counts[key] == 1
    assert len(planner.refine_calls) == 1

    # Second observation must mint a DISTINCT drift condition so the
    # I5 per-condition refine gate (which keys on kind+task+agent+turn)
    # doesn't suppress it. Vary ``current_agent_id``: same counter key
    # (counter only keys on kind+task) but a fresh lifecycle bucket.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-b"), session)
    # Second failure hits the threshold: _register_refine_failure calls
    # mark_task_failed, which fires a TASK_FAILED_FATAL drift that routes
    # through _handle_drift and triggers one more refine attempt (keyed
    # on a different (kind, task) tuple, so the TOOL_ERROR counter stays
    # at 2 and the TASK_FAILED_FATAL counter reaches 1).
    assert session.refine_failure_counts[key] == 2
    fatal_key = (DriftKind.TASK_FAILED_FATAL.value, "t1")
    assert session.refine_failure_counts.get(fatal_key, 0) == 1


async def test_refine_failure_counter_increments_on_none_return() -> None:
    # revised=None is the default — refine returns None without raising.
    planner = StubPlanner(revised=None)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")
    assert session.refine_failure_counts.get(key, 0) == 0

    await steerer._handle_drift(_tool_error_drift(agent_id="agent-a"), session)
    assert session.refine_failure_counts[key] == 1
    assert len(planner.refine_calls) == 1

    # Distinct ``current_agent_id`` mints a fresh drift condition (I5
    # gate exemption) while preserving the (kind, task) failure-counter
    # key.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-b"), session)
    # Same cascade as the exception case: the TOOL_ERROR counter is
    # clamped at the threshold (2), the cascaded TASK_FAILED_FATAL drift
    # kicks off its own independent counter at 1.
    assert session.refine_failure_counts[key] == 2
    fatal_key = (DriftKind.TASK_FAILED_FATAL.value, "t1")
    assert session.refine_failure_counts.get(fatal_key, 0) == 1


async def test_two_consecutive_refine_failures_marks_task_failed() -> None:
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    # First failure: visibility-only, task stays PENDING.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-a"), session)
    assert _task(session, "t1").status is TaskStatus.PENDING

    # Second failure: backoff trips — task marked FAILED, REPEATED_FAILURE
    # drift emitted. Distinct ``current_agent_id`` so the I5 lifecycle
    # gate doesn't fold this onto the prior condition.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-b"), session)
    assert _task(session, "t1").status is TaskStatus.FAILED

    from goldfive.pb.goldfive.v1 import types_pb2

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    # Expected stream across both ticks (inclusive of the unrecoverable
    # downstream cascade — PLAN-LIFECYCLE.md §6.2 step 3 — that
    # mark_task_failed(recoverable=False) now drives through the shared
    # cascade_cancel_downstream primitive):
    #   tick 1: drift_detected (TOOL_ERROR)
    #   tick 2: drift_detected (TOOL_ERROR),
    #           task_failed(t1),
    #           task_cancelled(t2, cascade from t1),
    #           drift_detected (TASK_FAILED_FATAL — from mark_task_failed),
    #           drift_detected (REPEATED_FAILURE).
    assert "task_failed" in kinds
    drift_kinds = [
        e.drift_detected.kind
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
    ]
    assert types_pb2.DRIFT_KIND_REPEATED_FAILURE in drift_kinds
    # The REPEATED_FAILURE drift should be CRITICAL severity.
    repeated_events = [
        e
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_REPEATED_FAILURE
    ]
    assert len(repeated_events) == 1
    assert repeated_events[0].drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    # After the threshold trip, a *third* trigger of the same drift must
    # short-circuit without calling refine again — the counter wall
    # prevents the loop that §7.3 targets.
    calls_before = len(planner.refine_calls)
    await steerer._handle_drift(_tool_error_drift(), session)
    assert len(planner.refine_calls) == calls_before


async def test_successful_refine_resets_failure_counter() -> None:
    # Planner that we can flip between "raise" and "succeed" modes.
    revised = _make_plan(("t1", "t2", "t3"))
    planner = StubPlanner(raise_exc=RuntimeError("planner down"))
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()

    key = (DriftKind.TOOL_ERROR.value, "t1")

    await steerer._handle_drift(_tool_error_drift(agent_id="agent-a"), session)
    assert session.refine_failure_counts[key] == 1

    # Flip planner to success mode BEFORE the counter hits the threshold
    # so the next call takes the happy path and must clear the counter.
    planner.raise_exc = None
    planner.revised = revised

    # Distinct ``current_agent_id`` mints a fresh drift condition (I5
    # gate exemption) so the success path runs and clears the failure
    # counter.
    await steerer._handle_drift(_tool_error_drift(agent_id="agent-b"), session)
    # Successful refine clears the key entirely (pop) — the counter is
    # reset, not decremented, so a fresh run of failures starts at 0.
    assert key not in session.refine_failure_counts
    # And the plan actually got revised on the success.
    assert session.plan is not None
    assert [t.id for t in session.plan.tasks] == ["t1", "t2", "t3"]


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
        # Emitted by the refine-failure backoff in _handle_drift once the
        # per-(kind, task) counter hits REFINE_FAILURE_THRESHOLD.
        DriftKind.REPEATED_FAILURE,
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
    # goldfive a4: also a refine_attempted + refine_failed dict envelope
    # — filter to proto events for the count assertion.
    assert len(sink.proto_events) == 2
    assert sink.proto_events[0].WhichOneof("payload") == "drift_detected"
    from goldfive.pb.goldfive.v1 import types_pb2

    assert sink.proto_events[0].drift_detected.kind == types_pb2.DRIFT_KIND_NEW_WORK_DISCOVERED
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
    # mark_task_running and mark_task_completed each emit a per-status
    # envelope plus a goldfive#251 R4 task_transitioned envelope; progress
    # is a liveness tick (no transition). 5 envelopes total, monotonic.
    seqs = [e.sequence for e in sink.events]
    assert seqs == sorted(seqs)
    assert seqs == list(range(len(seqs)))
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

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    # The initial TOOL_ERROR drift, then a CRITICAL validation-failure
    # drift when the bad revised plan is rejected. No PlanRevised is
    # emitted because the revision was not installed.
    assert kinds == ["drift_detected", "drift_detected"]
    # Original plan is still in place.
    assert session.plan is original_plan
    # The second drift is the validation-failure signal.
    second = sink.proto_events[1].drift_detected
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
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "drift_detected"]
    second = sink.proto_events[1].drift_detected
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
    second = sink.proto_events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert "unknown task id" in second.detail


async def test_apply_revision_emits_schema_violation_on_terminal_regression() -> None:
    """A refine that regresses a terminal task's status is rejected.

    PLAN-LIFECYCLE.md §3.1: terminal tasks are frozen — once a task
    lands in COMPLETED / FAILED / CANCELLED, subsequent revisions must
    preserve id AND status. The steerer must reject a revision that
    flips a previously-COMPLETED task back to PENDING, emit a CRITICAL
    SCHEMA_VIOLATION drift with the validator's reason, and keep the
    original plan installed.
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    # Seed the session with a plan where t1 is COMPLETED, t2 is PENDING.
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="done", status=TaskStatus.COMPLETED),
            Task(id="t2", title="next", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge("t1", "t2")],
    )
    # Bad revision: t1 has regressed from COMPLETED -> PENDING.
    bad_revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="done", status=TaskStatus.PENDING),
            Task(id="t2", title="next", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge("t1", "t2")],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session(plan=prior)

    await steerer.observe({"error": "trigger refine"}, session)

    # Plan unchanged; two drift events (the original trigger + the
    # schema-violation report); no PlanRevised emitted.
    assert session.plan is prior
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "drift_detected"]
    second = sink.proto_events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert second.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "plan validation failed" in second.detail
    assert "terminal task 't1' regressed" in second.detail


async def test_apply_revision_emits_schema_violation_on_missing_terminal_edge() -> None:
    """A refine that drops a terminal->terminal edge is rejected.

    PLAN-LIFECYCLE.md §3.2: edges whose both endpoints were terminal in
    the outgoing plan are frozen and must appear verbatim in any
    revision. The steerer must reject a revision that drops such an
    edge, emit a CRITICAL SCHEMA_VIOLATION, and keep the prior plan.
    """
    from goldfive.pb.goldfive.v1 import types_pb2

    # Seed the session with a plan where both t1 and t2 are terminal
    # with an edge between them.
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="2", status=TaskStatus.COMPLETED),
            Task(id="t3", title="3", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge("t1", "t2"), TaskEdge("t2", "t3")],
    )
    # Bad revision: preserves terminals but drops the t1 -> t2 edge.
    bad_revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="1", status=TaskStatus.COMPLETED),
            Task(id="t2", title="2", status=TaskStatus.COMPLETED),
            Task(id="t3", title="3", status=TaskStatus.PENDING),
        ],
        edges=[TaskEdge("t2", "t3")],
    )
    planner = StubPlanner(revised=bad_revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session(plan=prior)

    await steerer.observe({"error": "trigger refine"}, session)

    assert session.plan is prior
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "drift_detected"]
    second = sink.proto_events[1].drift_detected
    assert second.kind == types_pb2.DRIFT_KIND_SCHEMA_VIOLATION
    assert second.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "plan validation failed" in second.detail
    assert "terminal->terminal edge 't1' -> 't2'" in second.detail


async def test_plan_revised_carries_diff() -> None:
    """The ``PlanRevised`` event populates ``diff`` with add/remove/modify deltas.

    Closes PLAN-LIFECYCLE.md §8 gap #4: sinks should not have to re-fetch
    the old plan to render "what changed". The steerer's
    ``_emit_plan_revised`` captures the outgoing plan before
    ``_apply_revision`` swaps it in, so the emitted event carries a
    populated ``PlanRevisionDiff`` covering tasks added, removed,
    modified, and edges added / removed.
    """
    # Outgoing plan: t1 -> t2, with t2's title about to change.
    old = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="old title"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    # Revised plan: t1 preserved; t2 re-titled (modified); t3 added;
    # original t1->t2 edge dropped in favour of t1->t3.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1"),
            Task(id="t2", title="new title"),
            Task(id="t3", title="T3"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t3")],
    )
    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session(old)

    await steerer.observe({"error": "trigger refine"}, session)

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["drift_detected", "plan_revised"]
    pr = sink.proto_events[1].plan_revised
    assert list(pr.diff.added_task_ids) == ["t3"]
    assert list(pr.diff.removed_task_ids) == []
    assert list(pr.diff.modified_task_ids) == ["t2"]
    added = [(e.from_task_id, e.to_task_id) for e in pr.diff.added_edges]
    removed = [(e.from_task_id, e.to_task_id) for e in pr.diff.removed_edges]
    assert added == [("t1", "t3")]
    assert removed == [("t1", "t2")]


async def test_no_op_revision_is_rejected_and_escalates() -> None:
    """Revised plan structurally identical to the old → no-op rejection.

    goldfive#271: the steerer's structural guarantee at the install
    boundary catches refines that don't actually change anything (the
    Qwen judge re-firing on a corrected task pattern). Instead of
    bumping ``revision_index`` for an unchanged plan, the steerer
    treats the handler as exhausted and escalates to
    HUMAN_INTERVENTION_REQUIRED.

    Pre-#271 behaviour: emitted a ``plan_revised`` with empty diff
    lists. Post-#271 behaviour: skips the install entirely.
    """
    # Build the "revised" plan so it is structurally equal to the
    # outgoing plan but is a distinct object (so the helper doesn't
    # short-circuit on identity).
    tasks_a = [Task(id="t1", title="T1"), Task(id="t2", title="T2")]
    tasks_b = [Task(id="t1", title="T1"), Task(id="t2", title="T2")]
    edges_a = [TaskEdge(from_task_id="t1", to_task_id="t2")]
    edges_b = [TaskEdge(from_task_id="t1", to_task_id="t2")]
    old = Plan(id="p1", run_id="r1", goal_ids=["g1"], tasks=tasks_a, edges=edges_a)
    revised = Plan(id="p1", run_id="r1", goal_ids=["g1"], tasks=tasks_b, edges=edges_b)

    planner = StubPlanner(revised=revised)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session(old)

    await steerer.observe({"error": "trigger refine"}, session)

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    # No plan_revised emitted — handler exhausted instead.
    assert "plan_revised" not in kinds
    # Two drift_detected events: the original drift + the
    # HUMAN_INTERVENTION_REQUIRED escalation.
    drift_kinds = [
        e.drift_detected.kind
        for e in sink.proto_events
        if e.WhichOneof("payload") == "drift_detected"
    ]
    assert len(drift_kinds) >= 2
    # Session paused for human intervention.
    assert session.paused_for_human_intervention is True


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
    # task_started + R4 task_transitioned.
    assert len(sink_b.events) == 2


# ---------------------------------------------------------------------------
# Cancellation cascade (TASK-LIFECYCLE.md §6.1, STATE-MACHINE.md
# §"Cascade on task cancellation")
# ---------------------------------------------------------------------------


async def test_mark_task_cancelled_cascades_to_downstream_pending() -> None:
    """t1 -> t2 -> t3 linear chain; cancelling t1 cascades to t2 and t3.

    Regression guard for the "make a presentation about waffles" bug:
    a USER_STEER that cancels the current task while refine fails must
    still end up cancelling downstream PENDING tasks, otherwise the
    executor silently runs only the independent branches and reports
    success while leaving the dependent branch stuck PENDING forever.
    """
    steerer = DefaultSteerer()
    plan = _make_plan(("t1", "t2", "t3"))
    session = _make_session(plan)
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    await steerer.mark_task_cancelled("t1", session=session, reason="user cancelled")

    # All three tasks end up CANCELLED.
    assert _task(session, "t1").status is TaskStatus.CANCELLED
    assert _task(session, "t2").status is TaskStatus.CANCELLED
    assert _task(session, "t3").status is TaskStatus.CANCELLED

    # One TaskCancelled event per task, in cascade order (t1 first,
    # then BFS downstream).
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["task_cancelled", "task_cancelled", "task_cancelled"]
    cancelled_evts = [
        e for e in sink.events if e.WhichOneof("payload") == "task_cancelled"
    ]
    reasons = [e.task_cancelled.reason for e in cancelled_evts]
    ids = [e.task_cancelled.task_id for e in cancelled_evts]
    assert ids == ["t1", "t2", "t3"]
    # The initiator keeps the caller's reason; the cascaded tasks
    # carry a structured ``upstream_failed:<initiator>`` reason
    # (goldfive#205) so operators — and harmonograf's Trajectory view —
    # can trace the blast radius back to the trigger.
    assert reasons[0] == "user cancelled"
    for r in reasons[1:]:
        assert r == "upstream_failed:t1"


async def test_mark_task_cancelled_does_not_cascade_to_completed() -> None:
    """t1 -> t2 where t2 is already COMPLETED: t2 stays COMPLETED.

    Terminal statuses absorb. A late cancel on an upstream task must
    not retroactively un-complete a downstream done-task — all
    refines already preserve completed tasks verbatim and the cascade
    here must do the same.
    """
    steerer = DefaultSteerer()
    plan = _make_plan(("t1", "t2"))
    session = _make_session(plan)
    _task(session, "t2").status = TaskStatus.COMPLETED
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    await steerer.mark_task_cancelled("t1", session=session, reason="steer")

    assert _task(session, "t1").status is TaskStatus.CANCELLED
    assert _task(session, "t2").status is TaskStatus.COMPLETED
    # Exactly one TaskCancelled event (for t1). t2 was preserved.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["task_cancelled"]
    assert sink.events[0].task_cancelled.task_id == "t1"


async def test_mark_task_cancelled_multiple_downstream_paths() -> None:
    """Diamond DAG t1 -> {t2, t3} -> t4: all three downstream cancel.

    Confirms the BFS de-duplicates: t4 is reachable via both t2 and
    t3 but must be cancelled (and emit TaskCancelled) exactly once.
    """
    steerer = DefaultSteerer()
    plan = Plan(
        id="p-diamond",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id=tid, title=tid.upper()) for tid in ("t1", "t2", "t3", "t4")],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t1", to_task_id="t3"),
            TaskEdge(from_task_id="t2", to_task_id="t4"),
            TaskEdge(from_task_id="t3", to_task_id="t4"),
        ],
    )
    session = _make_session(plan)
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    await steerer.mark_task_cancelled("t1", session=session, reason="steer")

    for tid in ("t1", "t2", "t3", "t4"):
        assert _task(session, tid).status is TaskStatus.CANCELLED, tid

    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert kinds == ["task_cancelled"] * 4
    ids = [
        e.task_cancelled.task_id
        for e in sink.events
        if e.WhichOneof("payload") == "task_cancelled"
    ]
    # t1 first; then its direct children (t2, t3 in edge-order); then t4.
    # De-duplication: t4 appears exactly once despite two paths.
    assert ids[0] == "t1"
    assert set(ids[1:3]) == {"t2", "t3"}
    assert ids[3] == "t4"
    assert ids.count("t4") == 1


async def test_mark_task_cancelled_does_not_re_cancel() -> None:
    """Cancelling an already-CANCELLED task is a full no-op.

    No re-emission of TaskCancelled for the task itself and — critically
    — no re-cascade into downstream tasks, which would double-emit
    TaskCancelled events on every redundant call.
    """
    steerer = DefaultSteerer()
    plan = _make_plan(("t1", "t2"))
    session = _make_session(plan)
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())

    await steerer.mark_task_cancelled("t1", session=session, reason="first")
    first_event_count = len(sink.events)
    # t1 task_cancelled + R4 task_transitioned + cascaded t2 task_cancelled
    # + R4 task_transitioned = 4 envelopes.
    assert first_event_count == 4

    # Second call on the same already-terminal task: no new events.
    await steerer.mark_task_cancelled("t1", session=session, reason="again")
    assert len(sink.events) == first_event_count
    assert _task(session, "t1").status is TaskStatus.CANCELLED
    assert _task(session, "t2").status is TaskStatus.CANCELLED


async def test_cascade_primitive_shared_between_recoverable_and_unrecoverable_paths() -> None:
    """Both cascade paths fan out CANCELs through the same primitive.

    Closes PLAN-LIFECYCLE.md §8 gap #3: the recoverable cascade (§6.3 —
    driven by ``mark_task_cancelled``) and the unrecoverable cascade
    (§6.2 — driven by ``mark_task_failed(recoverable=False)``) share
    :meth:`DefaultSteerer.cascade_cancel_downstream`. Both paths must
    therefore emit ``TaskCancelled`` events for the same downstream set
    with identical reasons.

    This is the regression guard against future divergence: if somebody
    adds a second BFS-cancel implementation (e.g. an "unrecoverable
    cascade" method on the executor that mutates status directly), the
    downstream event stream for the two paths will drift and this test
    will flag it.
    """

    # Diamond DAG t1 -> {t2, t3} -> t4 so the test exercises both
    # forward-BFS and the dedup-across-paths guarantee on both paths.
    def _build() -> tuple[DefaultSteerer, Session, ListSink]:
        plan = Plan(
            id="p-cascade-parity",
            run_id="r1",
            goal_ids=["g1"],
            tasks=[Task(id=tid, title=tid.upper()) for tid in ("t1", "t2", "t3", "t4")],
            edges=[
                TaskEdge(from_task_id="t1", to_task_id="t2"),
                TaskEdge(from_task_id="t1", to_task_id="t3"),
                TaskEdge(from_task_id="t2", to_task_id="t4"),
                TaskEdge(from_task_id="t3", to_task_id="t4"),
            ],
        )
        session = _make_session(plan)
        sink = ListSink()
        steerer = DefaultSteerer()
        steerer.bind(sinks=[sink], planner=StubPlanner())
        return steerer, session, sink

    # --- Recoverable path: mark_task_cancelled("t1") -------------------
    rec_steerer, rec_session, rec_sink = _build()
    await rec_steerer.mark_task_cancelled("t1", session=rec_session, reason="steer")
    rec_cancelled = [
        e.task_cancelled
        for e in rec_sink.proto_events
        if e.WhichOneof("payload") == "task_cancelled"
    ]
    # Initiator t1 + downstream {t2, t3, t4} = 4 TaskCancelled events.
    assert [c.task_id for c in rec_cancelled][0] == "t1"
    rec_downstream_ids = sorted(c.task_id for c in rec_cancelled[1:])
    assert rec_downstream_ids == ["t2", "t3", "t4"]
    rec_downstream_reasons = {c.task_id: c.reason for c in rec_cancelled[1:]}

    # --- Unrecoverable path: mark_task_failed("t1", recoverable=False) -
    fat_steerer, fat_session, fat_sink = _build()
    await fat_steerer.mark_task_failed("t1", session=fat_session, reason="fatal", recoverable=False)
    fat_cancelled = [
        e.task_cancelled
        for e in fat_sink.proto_events
        if e.WhichOneof("payload") == "task_cancelled"
    ]
    # Initiator t1 is FAILED (not CANCELLED), so only the *downstream*
    # set shows up as TaskCancelled events — same three tasks as the
    # recoverable path produced downstream.
    fat_downstream_ids = sorted(c.task_id for c in fat_cancelled)
    assert fat_downstream_ids == ["t2", "t3", "t4"]
    fat_downstream_reasons = {c.task_id: c.reason for c in fat_cancelled}

    # Core parity assertion: the downstream TaskCancelled events emitted
    # by the two paths carry the same reason strings
    # (``upstream_failed:t1`` post goldfive#205), proving both paths
    # funnel through cascade_cancel_downstream.
    assert rec_downstream_reasons == fat_downstream_reasons
    for tid, reason in fat_downstream_reasons.items():
        assert reason == "upstream_failed:t1", (tid, reason)

    # Final plan shape parity on the downstream set (initiator differs:
    # CANCELLED on the recoverable path, FAILED on the unrecoverable).
    for tid in ("t2", "t3", "t4"):
        assert _task(rec_session, tid).status is TaskStatus.CANCELLED, tid
        assert _task(fat_session, tid).status is TaskStatus.CANCELLED, tid
    assert _task(rec_session, "t1").status is TaskStatus.CANCELLED
    assert _task(fat_session, "t1").status is TaskStatus.FAILED

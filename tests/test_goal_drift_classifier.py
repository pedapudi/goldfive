"""Unit tests for the periodic GOAL_DRIFT trajectory check (goldfive#143).

Two layers:

* The pure :func:`goldfive.drift.classify_goal_drift` classifier: takes
  goals + plan + recent activity, calls an LLM-judge, returns a
  ``DriftEvent`` at CRITICAL severity when the judge says the tree is
  off-goal, ``None`` otherwise. Robust to LLM failures.
* The :class:`~goldfive.steerer.DefaultSteerer` periodic-invocation
  wrapper (:meth:`note_agent_turn` + :meth:`maybe_run_goal_drift_check`):
  caller-managed counter fires the judge at the configured interval,
  never more than one LLM call per check, and routes CRITICAL drift
  through ``_handle_drift`` so the #142 ladder can route it to Level 4.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.drift import classify_goal_drift  # noqa: E402
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
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    def __init__(self) -> None:
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, *, goals, available_agents, context=None):
        return None

    async def refine(self, *, plan, drift, goals):
        self.refine_calls.append({"plan": plan, "drift": drift, "goals": goals})
        return None


def _stub_call_llm(responses: list[Any]):
    """Async call_llm stub that pops responses in order.

    dict entries are json-encoded; strings are returned verbatim;
    Exception instances are raised. Any entry raising is a synthetic
    plumbing failure the classifier must absorb.
    """
    queue = list(responses)
    calls: list[tuple[str, str, str]] = []

    async def _call_llm(system: str, user: str, model: str) -> str:
        calls.append((system, user, model))
        if not queue:
            raise AssertionError("stub call_llm exhausted")
        resp = queue.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)

    _call_llm.calls = calls  # type: ignore[attr-defined]
    return _call_llm


def _make_session() -> Session:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="Gather facts", description="Collect sources"),
            Task(id="t2", title="Write draft", description="Draft the memo"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="Publish a memo on topic X")],
        plan=plan,
        current_task_id="t1",
    )


# ---------------------------------------------------------------------------
# Pure classifier tests
# ---------------------------------------------------------------------------


async def test_classifier_on_track_returns_none() -> None:
    """Judge says progressing=True → classifier returns None."""
    call_llm = _stub_call_llm([{"progressing": True}])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship the memo")],
        plan=Plan(id="p", run_id="r", goal_ids=["g1"], tasks=[], edges=[]),
        observed_actions=[
            {"kind": "agent_invocation_completed", "agent_name": "writer", "task_id": "t1"},
        ],
        model="test-model",
        call_llm=call_llm,
    )
    assert drift is None
    # Exactly one LLM call made — cost-bounded contract.
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


async def test_classifier_off_track_returns_critical_drift() -> None:
    """Judge says progressing=False → CRITICAL GOAL_DRIFT with reason."""
    reason = "agents are looping on research with no writing"
    call_llm = _stub_call_llm([{"progressing": False, "reason": reason}])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship the memo")],
        plan=Plan(
            id="p",
            run_id="r",
            goal_ids=["g1"],
            tasks=[Task(id="t1", title="Research", status=TaskStatus.RUNNING)],
            edges=[],
        ),
        observed_actions=[
            {"kind": "agent_invocation_completed", "agent_name": "researcher", "task_id": "t1"}
            for _ in range(5)
        ],
        model="test-model",
        call_llm=call_llm,
        current_task_id="t1",
        current_agent_id="researcher",
    )
    assert isinstance(drift, DriftEvent)
    assert drift.kind is DriftKind.GOAL_DRIFT
    assert drift.severity is DriftSeverity.CRITICAL
    assert reason in drift.detail
    assert drift.current_task_id == "t1"
    assert drift.current_agent_id == "researcher"


async def test_classifier_malformed_json_returns_none() -> None:
    """Garbage response → None, never a false-positive GOAL_DRIFT."""
    call_llm = _stub_call_llm(["this is not json at all, sorry"])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=None,
        model="m",
        call_llm=call_llm,
    )
    assert drift is None


async def test_classifier_missing_progressing_field_returns_none() -> None:
    """JSON without a boolean 'progressing' key → None."""
    call_llm = _stub_call_llm([{"result": "yes", "reason": "ok"}])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=None,
        model="m",
        call_llm=call_llm,
    )
    assert drift is None


async def test_classifier_non_boolean_progressing_returns_none() -> None:
    """progressing is a string, not a bool → None (no false positive)."""
    call_llm = _stub_call_llm([{"progressing": "no", "reason": "stuck"}])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=None,
        model="m",
        call_llm=call_llm,
    )
    assert drift is None


async def test_classifier_call_llm_raises_returns_none() -> None:
    """call_llm throws → classifier absorbs; no drift, no re-raise."""
    call_llm = _stub_call_llm([RuntimeError("network down")])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=None,
        model="m",
        call_llm=call_llm,
    )
    assert drift is None


async def test_classifier_tolerates_fenced_markdown() -> None:
    """Judge wraps JSON in ```json ... ``` — classifier extracts it."""
    call_llm = _stub_call_llm(['```json\n{"progressing": false, "reason": "off"}\n```'])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=[{"kind": "x", "agent_name": "a", "task_id": "t", "detail": "d"}],
        model="m",
        call_llm=call_llm,
    )
    assert isinstance(drift, DriftEvent)
    assert drift.kind is DriftKind.GOAL_DRIFT
    assert "off" in drift.detail


async def test_classifier_no_reason_still_emits_drift() -> None:
    """progressing=False without a reason still produces GOAL_DRIFT."""
    call_llm = _stub_call_llm([{"progressing": False}])
    drift = await classify_goal_drift(
        goals=[Goal(id="g1", summary="ship X")],
        plan=None,
        observed_actions=None,
        model="m",
        call_llm=call_llm,
    )
    assert isinstance(drift, DriftEvent)
    assert drift.kind is DriftKind.GOAL_DRIFT
    assert drift.severity is DriftSeverity.CRITICAL


async def test_classifier_one_llm_call_per_invocation() -> None:
    """Caller-managed counter: classifier never loops on one call."""
    call_llm = _stub_call_llm(
        [
            {"progressing": False, "reason": "a"},
            {"progressing": False, "reason": "b"},
        ]
    )
    # Two separate invocations — the classifier issues one LLM call each.
    d1 = await classify_goal_drift(
        goals=[Goal(id="g1", summary="X")],
        plan=None,
        observed_actions=None,
        model="",
        call_llm=call_llm,
    )
    d2 = await classify_goal_drift(
        goals=[Goal(id="g1", summary="X")],
        plan=None,
        observed_actions=None,
        model="",
        call_llm=call_llm,
    )
    assert d1 is not None and "a" in d1.detail
    assert d2 is not None and "b" in d2.detail
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Steerer periodic-invocation tests
# ---------------------------------------------------------------------------


async def test_steerer_counter_fires_at_interval() -> None:
    """5 agent turns produce exactly one GOAL_DRIFT judge call."""
    call_llm = _stub_call_llm(
        [
            {"progressing": True},
            {"progressing": True},
        ]
    )
    steerer = DefaultSteerer(
        goal_drift_check_interval=5,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    for _ in range(4):
        await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 0, "should not fire before the 5th turn"  # type: ignore[attr-defined]
    await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 1, "should fire on the 5th turn"  # type: ignore[attr-defined]
    # Counter resets after check; next 4 should not re-fire.
    for _ in range(4):
        await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]


async def test_steerer_off_track_emits_critical_drift_and_pauses() -> None:
    """Judge returns False → CRITICAL GOAL_DRIFT → Level 4 pause.

    goldfive#142's intervention ladder routes CRITICAL GOAL_DRIFT to
    :data:`InterventionLevel.PAUSE_ESCALATE` rather than calling
    ``planner.refine`` -- goal drift is a structural signal that refine
    cannot recover from. The Steerer emits a ``HUMAN_INTERVENTION_REQUIRED``
    escalation drift and flips ``session.paused_for_human_intervention``;
    the executor blocks until a user-initiated RESUME / STEER arrives.
    """
    call_llm = _stub_call_llm([{"progressing": False, "reason": "off in the weeds"}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=2,
        goal_drift_call_llm=call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    for _ in range(2):
        await steerer.note_agent_turn(session)

    drifts = [e for e in sink.events if e.WhichOneof("payload") == "drift_detected"]
    assert len(drifts) >= 1
    from goldfive.pb.goldfive.v1 import types_pb2

    primary = drifts[0]
    assert primary.drift_detected.kind == types_pb2.DRIFT_KIND_GOAL_DRIFT
    assert primary.drift_detected.severity == types_pb2.DRIFT_SEVERITY_CRITICAL
    assert "off in the weeds" in primary.drift_detected.detail
    # Level 4 pause-escalate: no refine, a HUMAN_INTERVENTION_REQUIRED
    # follow-up drift, and the pause flag is set.
    assert planner.refine_calls == []
    assert any(
        e.WhichOneof("payload") == "drift_detected"
        and e.drift_detected.kind == types_pb2.DRIFT_KIND_HUMAN_INTERVENTION_REQUIRED
        for e in sink.events
    )
    assert session.paused_for_human_intervention is True


async def test_steerer_disabled_by_default_no_check_ever_fires() -> None:
    """Without goal_drift_call_llm the counter is inert."""
    steerer = DefaultSteerer()  # no callable
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    for _ in range(100):
        await steerer.note_agent_turn(session)
    assert session._agent_turns_since_goal_check == 0


async def test_steerer_malformed_response_does_not_crash_run() -> None:
    """Judge returns garbage → no drift, run continues."""
    call_llm = _stub_call_llm(["garbage not json"])
    steerer = DefaultSteerer(
        goal_drift_check_interval=1,
        goal_drift_call_llm=call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    await steerer.note_agent_turn(session)
    drifts = [e for e in sink.events if e.WhichOneof("payload") == "drift_detected"]
    assert drifts == []
    assert planner.refine_calls == []


async def test_steerer_activity_window_bounded() -> None:
    """note_agent_activity trims to goal_drift_activity_window."""
    steerer = DefaultSteerer(goal_drift_activity_window=3)
    session = _make_session()
    for i in range(10):
        steerer.note_agent_activity(
            session,
            kind="agent_invocation_started",
            agent_name=f"a{i}",
            task_id=f"t{i}",
        )
    assert len(session.recent_agent_activity) == 3
    # Newest three retained (ring buffer drops oldest).
    assert [e["agent_name"] for e in session.recent_agent_activity] == ["a7", "a8", "a9"]


async def test_steerer_counter_is_not_task_scoped() -> None:
    """GOAL_DRIFT counter persists across task transitions (trajectory-level)."""
    call_llm = _stub_call_llm([{"progressing": True}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=3,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    session.current_task_id = "t1"
    await steerer.note_agent_turn(session)
    # Task transition.
    session.current_task_id = "t2"
    await steerer.note_agent_turn(session)
    await steerer.note_agent_turn(session)
    # Third call should fire the check, not be held back by transition.
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


async def test_steerer_maybe_run_goal_drift_check_is_public() -> None:
    """Operators can trigger a one-shot check outside the interval."""
    call_llm = _stub_call_llm([{"progressing": False, "reason": "manual check fired drift"}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=999,  # effectively never auto-fires
        goal_drift_call_llm=call_llm,
    )
    planner = StubPlanner()
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=planner)
    session = _make_session()
    await steerer.maybe_run_goal_drift_check(session)
    # Counter NOT advanced by direct call.
    assert session._agent_turns_since_goal_check == 0
    # But drift was emitted.
    drifts = [e for e in sink.events if e.WhichOneof("payload") == "drift_detected"]
    assert len(drifts) >= 1


# ---------------------------------------------------------------------------
# Runner flag integration
# ---------------------------------------------------------------------------


async def test_runner_goal_drift_enabled_false_detaches_callable() -> None:
    """Runner(goal_drift_enabled=False) forcibly disables the check."""
    from goldfive.runner import Runner

    # Build a minimal Runner wrapping no-op pieces; we only care about
    # the __init__ side-effect on the steerer.
    class _NoopAdapter:
        available_agents: list[str] = []

        async def register_reporting_tools(self, tools):
            return None

    class _NoopExecutor:
        async def run(self, **kwargs):
            raise AssertionError("not invoked")

    call_llm = _stub_call_llm([{"progressing": False, "reason": "x"}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=1,
        goal_drift_call_llm=call_llm,
    )
    assert steerer._goal_drift_call_llm is call_llm
    runner = Runner(
        agent=_NoopAdapter(),
        planner=StubPlanner(),
        executor=_NoopExecutor(),
        steerer=steerer,
        goal_drift_enabled=False,
    )
    assert runner.goal_drift_enabled is False
    # Steerer's callable is detached — no LLM call can fire now.
    assert steerer._goal_drift_call_llm is None
    session = _make_session()
    await steerer.note_agent_turn(session)
    assert len(call_llm.calls) == 0  # type: ignore[attr-defined]


async def test_runner_goal_drift_enabled_true_is_default_and_preserves_wiring() -> None:
    """Runner default (True) leaves the steerer's callable untouched."""
    from goldfive.runner import Runner

    class _NoopAdapter:
        available_agents: list[str] = []

        async def register_reporting_tools(self, tools):
            return None

    class _NoopExecutor:
        async def run(self, **kwargs):
            raise AssertionError("not invoked")

    call_llm = _stub_call_llm([{"progressing": True}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=1,
        goal_drift_call_llm=call_llm,
    )
    runner = Runner(
        agent=_NoopAdapter(),
        planner=StubPlanner(),
        executor=_NoopExecutor(),
        steerer=steerer,
    )
    assert runner.goal_drift_enabled is True
    assert steerer._goal_drift_call_llm is call_llm


# ---------------------------------------------------------------------------
# Task-boundary trigger (goldfive#219)
# ---------------------------------------------------------------------------


async def test_task_boundary_fires_goal_drift_check_on_mark_task_completed() -> None:
    """mark_task_completed fires the judge exactly once after emit.

    Regression on goldfive#219: short pipelines (e.g. coordinator →
    research → web_dev → reviewer) complete before the turn counter
    reaches the configured interval, so the turn-based judge never
    fires. Task transitions are the natural fallback.
    """
    call_llm = _stub_call_llm([{"progressing": True}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=100,  # Way above any turn count.
        goal_drift_call_llm=call_llm,
    )
    sink = ListSink()
    steerer.bind(sinks=[sink], planner=StubPlanner())
    session = _make_session()

    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # Turn counter is reset on task-boundary fire so we don't double-pay.
    assert session._agent_turns_since_goal_check == 0


async def test_task_boundary_fires_on_mark_task_failed_and_cancelled() -> None:
    """Both FAILED and CANCELLED transitions trigger the judge."""
    # Provide four responses: one per transition we'll drive below. The
    # 10s rate limit is enforced via _last_goal_drift_check_ts; we reset
    # it between calls to simulate "enough real time has passed".
    call_llm = _stub_call_llm(
        [{"progressing": True}, {"progressing": True}, {"progressing": True}]
    )
    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()

    await steerer.mark_task_failed("t1", session=session, reason="boom")
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]

    # Rewind the rate-limit clock so the next boundary fires.
    session._last_goal_drift_check_ts = 0.0
    await steerer.mark_task_cancelled("t2", session=session, reason="no longer needed")
    assert len(call_llm.calls) == 2  # type: ignore[attr-defined]


async def test_task_boundary_fires_independent_of_turn_counter() -> None:
    """Turn counter at 0 (never incremented) → task boundary still fires."""
    call_llm = _stub_call_llm([{"progressing": True}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=5,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    assert session._agent_turns_since_goal_check == 0

    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]


async def test_task_boundary_rate_limited_within_ten_seconds() -> None:
    """Two task transitions 1s apart → judge only fires once."""
    call_llm = _stub_call_llm([{"progressing": True}, {"progressing": True}])
    steerer = DefaultSteerer(
        goal_drift_check_interval=100,
        goal_drift_call_llm=call_llm,
    )
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()

    # First transition primes the timestamp.
    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    first_ts = session._last_goal_drift_check_ts
    assert first_ts > 0.0

    # Second transition <1s later: should be suppressed by the 10s guard.
    await steerer.mark_task_completed("t2", session=session, summary="done")
    assert len(call_llm.calls) == 1  # type: ignore[attr-defined]
    # Timestamp must NOT advance on a suppressed call.
    assert session._last_goal_drift_check_ts == first_ts


async def test_task_boundary_noop_when_goal_drift_not_wired() -> None:
    """Without a ``goal_drift_call_llm`` the boundary trigger is silent.

    Also asserts ``_last_goal_drift_check_ts`` stays at its default so
    the timestamp is only advanced when a real judge call is made.
    """
    steerer = DefaultSteerer(goal_drift_call_llm=None)
    steerer.bind(sinks=[ListSink()], planner=StubPlanner())
    session = _make_session()
    await steerer.mark_task_completed("t1", session=session, summary="done")
    assert session._last_goal_drift_check_ts == 0.0


# ---------------------------------------------------------------------------
# Judge-response logging (goldfive#219)
# ---------------------------------------------------------------------------


async def test_classifier_logs_on_track_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """progressing=true → DEBUG log with 'on-track' so operators can diagnose."""
    call_llm = _stub_call_llm([{"progressing": True, "reason": "writing in progress"}])
    with caplog.at_level("DEBUG", logger="goldfive.drift.goals"):
        drift = await classify_goal_drift(
            goals=[Goal(id="g1", summary="ship memo")],
            plan=None,
            observed_actions=[],
            model="m",
            call_llm=call_llm,
        )
    assert drift is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("on-track" in m for m in messages), messages
    # Raw response is also logged at DEBUG for every call.
    assert any("raw response" in m for m in messages), messages


async def test_classifier_logs_drift_detected_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """progressing=false → INFO log with 'drift detected' so operators see it
    without having to crank logging to DEBUG."""
    reason = "researching raccoons instead of solar panels"
    call_llm = _stub_call_llm([{"progressing": False, "reason": reason}])
    with caplog.at_level("INFO", logger="goldfive.drift.goals"):
        drift = await classify_goal_drift(
            goals=[Goal(id="g1", summary="ship memo")],
            plan=None,
            observed_actions=[],
            model="m",
            call_llm=call_llm,
        )
    assert drift is not None
    info_msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("drift detected" in m and "raccoons" in m for m in info_msgs), info_msgs


async def test_classifier_logs_malformed_json_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed response → DEBUG log including the raw text snippet."""
    call_llm = _stub_call_llm(["this is not JSON at all -- raccoons everywhere"])
    with caplog.at_level("DEBUG", logger="goldfive.drift.goals"):
        drift = await classify_goal_drift(
            goals=[Goal(id="g1", summary="ship memo")],
            plan=None,
            observed_actions=[],
            model="m",
            call_llm=call_llm,
        )
    assert drift is None
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "not JSON" in m and "raccoons" in m for m in messages
    ), messages


async def test_classifier_logs_missing_progressing_key_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dict without 'progressing' key → DEBUG log with 'lacks boolean'."""
    call_llm = _stub_call_llm([{"reason": "no key"}])
    with caplog.at_level("DEBUG", logger="goldfive.drift.goals"):
        drift = await classify_goal_drift(
            goals=[Goal(id="g1", summary="ship memo")],
            plan=None,
            observed_actions=[],
            model="m",
            call_llm=call_llm,
        )
    assert drift is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("lacks boolean" in m for m in messages), messages

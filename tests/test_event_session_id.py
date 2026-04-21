"""Tests for ``Event.session_id`` plumbing (goldfive#155).

Pins the per-event session-routing field added to the ``Event`` envelope:

* Every typed factory in :mod:`goldfive.events` accepts ``session_id=""``
  as an optional keyword and populates the proto field.
* Proto roundtrip preserves the field.
* Runner / Steerer / Executor stamp ``session.id`` on every Event
  they emit through the sinks.
* Back-compat: callers that don't pass ``session_id`` still produce
  valid envelopes with an empty ``session_id`` string.

Downstream (harmonograf) consumes the field to route per-event to the
correct session without relying on stream-level Hello metadata; the
default empty-string preserves pre-#155 Hello-based routing.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import (  # noqa: E402
    CallableAdapter,
    Goal,
    InMemorySink,
    InvocationResult,
    PassthroughGoalDeriver,
    Plan,
    ReportingToolSpec,
    Runner,
    SequentialExecutor,
    Session,
    StaticPlanner,
    Task,
    TaskEdge,
)
from goldfive.events import (  # noqa: E402
    agent_invocation_completed_event,
    agent_invocation_started_event,
    approval_granted_event,
    approval_rejected_event,
    approval_requested_event,
    conversation_ended_event,
    conversation_started_event,
    delegation_observed_event,
    drift_detected_event,
    goal_derived_event,
    new_event,
    plan_revised_event,
    plan_submitted_event,
    run_aborted_event,
    run_completed_event,
    run_started_event,
    task_blocked_event,
    task_cancelled_event,
    task_completed_event,
    task_failed_event,
    task_progress_event,
    task_started_event,
)
from goldfive.types import DriftEvent, DriftKind, DriftSeverity  # noqa: E402

SESSION_ID = "sess-155-test"


# ---------------------------------------------------------------------------
# Factory-level: every factory populates session_id when supplied.
# ---------------------------------------------------------------------------


def _factory_samples() -> list[tuple[str, Any, dict[str, Any]]]:
    """Return (label, factory, kwargs) tuples exercising every Event factory.

    The positional ``run_id`` / ``sequence`` args are passed inline; the
    keyword args plus ``session_id=SESSION_ID`` exercise each factory's
    session-id plumbing.
    """
    plan = Plan(
        id="p1",
        run_id="r",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="T1")],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.USER_STEER,
        severity=DriftSeverity.WARNING,
        detail="test",
    )
    return [
        ("run_started", run_started_event, {"goal_summary": "hi"}),
        ("run_completed", run_completed_event, {"outcome_summary": "done"}),
        ("run_aborted", run_aborted_event, {"reason": "boom"}),
        ("goal_derived", goal_derived_event, {"goals": [Goal(id="g1", summary="x")]}),
        ("plan_submitted", plan_submitted_event, {"plan": plan}),
        (
            "plan_revised",
            plan_revised_event,
            {
                "plan": plan,
                "drift": drift,
                "revision_index": 1,
            },
        ),
        ("task_started", task_started_event, {"task_id": "t1", "detail": "d"}),
        (
            "task_progress",
            task_progress_event,
            {"task_id": "t1", "fraction": 0.5, "detail": "d"},
        ),
        (
            "task_completed",
            task_completed_event,
            {"task_id": "t1", "summary": "done", "artifacts": {"k": "v"}},
        ),
        (
            "task_failed",
            task_failed_event,
            {"task_id": "t1", "reason": "no", "recoverable": True},
        ),
        (
            "task_blocked",
            task_blocked_event,
            {"task_id": "t1", "blocker": "b", "needed": "n"},
        ),
        ("task_cancelled", task_cancelled_event, {"task_id": "t1", "reason": "r"}),
        ("drift_detected", drift_detected_event, {"drift": drift}),
        (
            "conversation_started",
            conversation_started_event,
            {"conversation_id": "c1"},
        ),
        (
            "conversation_ended",
            conversation_ended_event,
            {"conversation_id": "c1", "turn_count": 2, "reason": "done"},
        ),
        (
            "approval_requested",
            approval_requested_event,
            {
                "target_id": "a1",
                "kind": "task",
                "prompt": "ok?",
                "task_id": "t1",
                "metadata": {"k": "v"},
            },
        ),
        (
            "approval_granted",
            approval_granted_event,
            {"target_id": "a1", "detail": "ok"},
        ),
        (
            "approval_rejected",
            approval_rejected_event,
            {"target_id": "a1", "detail": "no"},
        ),
        (
            "agent_invocation_started",
            agent_invocation_started_event,
            {"agent_name": "writer"},
        ),
        (
            "agent_invocation_completed",
            agent_invocation_completed_event,
            {"agent_name": "writer", "summary": "ok"},
        ),
        (
            "delegation_observed",
            delegation_observed_event,
            {"from_agent": "a", "to_agent": "b"},
        ),
    ]


def test_event_factories_accept_session_id() -> None:
    """Every factory populates ``Event.session_id`` when supplied."""
    for label, fn, kwargs in _factory_samples():
        evt = fn("run-1", 0, session_id=SESSION_ID, **kwargs)
        assert evt.session_id == SESSION_ID, f"{label} did not populate session_id"
        # Sibling scalar fields stay intact.
        assert evt.run_id == "run-1", f"{label} dropped run_id"
        assert evt.sequence == 0, f"{label} dropped sequence"


def test_event_factory_signature_optional_kwarg() -> None:
    """Every factory accepts ``session_id`` as an optional keyword-only arg."""
    for label, fn, _kwargs in _factory_samples():
        sig = inspect.signature(fn)
        assert "session_id" in sig.parameters, f"{label} missing session_id param"
        param = sig.parameters["session_id"]
        assert param.default == "", f"{label} session_id default is not empty string"


def test_backcompat_empty_session_id_default() -> None:
    """Factories called without session_id still produce valid Events."""
    for label, fn, kwargs in _factory_samples():
        evt = fn("run-1", 0, **kwargs)
        assert evt.session_id == "", f"{label} populated session_id without caller input"
        assert evt.run_id == "run-1", label


def test_new_event_populates_session_id() -> None:
    """``new_event`` is the shared envelope builder — it must accept the field."""
    evt = new_event("run-1", 3, session_id=SESSION_ID)
    assert evt.session_id == SESSION_ID
    assert evt.run_id == "run-1"
    assert evt.sequence == 3

    # Default empty string.
    evt2 = new_event("run-1", 4)
    assert evt2.session_id == ""


# ---------------------------------------------------------------------------
# Proto roundtrip.
# ---------------------------------------------------------------------------


def test_event_protobuf_roundtrip_with_session_id() -> None:
    """SerializeToString → FromString preserves ``session_id``."""
    from goldfive.pb.goldfive.v1 import events_pb2

    evt = run_started_event(
        "run-1",
        0,
        goal_summary="hi",
        session_id=SESSION_ID,
    )
    wire = evt.SerializeToString()
    restored = events_pb2.Event()
    restored.ParseFromString(wire)
    assert restored.session_id == SESSION_ID
    assert restored.run_id == "run-1"
    assert restored.run_started.goal_summary == "hi"


# ---------------------------------------------------------------------------
# Runner integration: every emitted Event carries session.id.
# ---------------------------------------------------------------------------


def _hand_built_plan() -> Plan:
    return Plan(
        id="plan-fixture",
        run_id="",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", assignee_agent_id="writer"),
            Task(id="t2", title="T2", assignee_agent_id="writer"),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
        summary="tiny plan",
    )


async def _happy_agent(
    task: Task, session: Session, tools: list[ReportingToolSpec]
) -> InvocationResult:
    _ = tools
    _ = session
    return InvocationResult(task_id=task.id, text=f"done: {task.title}")


async def test_runner_stamps_session_id_on_emitted_events() -> None:
    """Every Event observed by a sink carries ``session.id`` as its ``session_id``."""
    sink = InMemorySink()
    runner = Runner(
        agent=CallableAdapter(_happy_agent, available_agents=["writer"]),
        planner=StaticPlanner(_hand_built_plan()),
        executor=SequentialExecutor(),
        goal_deriver=PassthroughGoalDeriver("go"),
        sinks=[sink],
    )

    outcome = await runner.run("go")
    await runner.close()

    assert outcome.success, outcome.reason
    expected = outcome.session.id
    assert expected, "session.id must be populated for the test to be meaningful"

    # Every proto envelope in the sink must carry the session_id. Dict-
    # shaped events (from `make_event`) are never produced by the
    # Runner — but if a third-party executor yielded one, we don't
    # assert on it here.
    proto_events = [e for e in sink.events if hasattr(e, "session_id")]
    assert proto_events, "sink saw no proto events — test fixture is broken"
    mismatches = [
        (e.WhichOneof("payload"), e.session_id, e.sequence)
        for e in proto_events
        if e.session_id != expected
    ]
    assert not mismatches, f"events missing session_id stamp: {mismatches!r}"


async def test_session_id_property_mirrors_run_id() -> None:
    """``Session.id`` aliases ``run_id`` for the stamping contract."""
    s = Session(run_id="r-42")
    assert s.id == "r-42"
    assert s.id == s.run_id

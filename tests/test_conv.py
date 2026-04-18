"""Round-trip tests for goldfive.conv.

These tests require the generated proto stubs under ``goldfive.pb``
(produced by issue #3). When those stubs are not present the whole module
is skipped so ``pytest`` stays green on branches where only the Python-side
dataclasses have landed.
"""

from __future__ import annotations

import importlib.util

import pytest

from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Goal,
    Plan,
    Task,
    TaskEdge,
    TaskStatus,
)


def _pb_available() -> bool:
    try:
        return importlib.util.find_spec("goldfive.pb.goldfive.v1.types_pb2") is not None
    except (ModuleNotFoundError, ImportError):
        return False


_PB_AVAILABLE = _pb_available()

pytestmark = pytest.mark.skipif(
    not _PB_AVAILABLE,
    reason="goldfive protobuf stubs not generated yet (depends on issue #3)",
)

if _PB_AVAILABLE:
    # The pb-dependent converters are only imported when the stubs exist.
    from goldfive.conv import (  # noqa: E402
        from_pb_drift_event,
        from_pb_goal,
        from_pb_plan,
        from_pb_task,
        from_pb_task_edge,
        to_pb_drift_event,
        to_pb_goal,
        to_pb_plan,
        to_pb_task,
        to_pb_task_edge,
    )


def test_task_round_trip() -> None:
    t = Task(
        id="t1",
        title="Research",
        description="look up papers",
        assignee_agent_id="researcher",
        status=TaskStatus.RUNNING,
        predicted_start_ms=100,
        predicted_duration_ms=5000,
        bound_span_id="span-abc",
    )
    assert from_pb_task(to_pb_task(t)) == t


def test_task_default_status_round_trip() -> None:
    t = Task(id="t1", title="A")
    assert from_pb_task(to_pb_task(t)) == t


def test_task_edge_round_trip() -> None:
    e = TaskEdge(from_task_id="a", to_task_id="b")
    assert from_pb_task_edge(to_pb_task_edge(e)) == e


def test_plan_round_trip() -> None:
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1", "g2"],
        tasks=[
            Task(id="t1", title="A", status=TaskStatus.COMPLETED),
            Task(id="t2", title="B", assignee_agent_id="writer"),
        ],
        edges=[TaskEdge("t1", "t2")],
        summary="research then write",
        revision_reason="observed tool error",
        revision_kind=DriftKind.TOOL_ERROR.value,
        revision_severity=DriftSeverity.WARNING.value,
        revision_index=2,
    )
    assert from_pb_plan(to_pb_plan(plan)) == plan


def test_plan_empty_round_trip() -> None:
    plan = Plan(id="p", run_id="r", goal_ids=[], tasks=[], edges=[])
    assert from_pb_plan(to_pb_plan(plan)) == plan


def test_goal_round_trip_drops_predicate() -> None:
    # success_predicate is a live callable — not serialisable. It is
    # intentionally dropped on round-trip.
    goal = Goal(
        id="g1",
        summary="ship it",
        success_predicate=lambda s: True,
        metadata={"owner": "pm", "priority": "high"},
    )
    recovered = from_pb_goal(to_pb_goal(goal))
    assert recovered.id == goal.id
    assert recovered.summary == goal.summary
    assert recovered.metadata == goal.metadata
    assert recovered.success_predicate is None


def test_drift_event_round_trip() -> None:
    d = DriftEvent(
        kind=DriftKind.NEW_WORK_DISCOVERED,
        severity=DriftSeverity.CRITICAL,
        detail="discovered a new task",
        current_task_id="t1",
        current_agent_id="researcher",
        raw={"opaque": "payload"},
    )
    recovered = from_pb_drift_event(to_pb_drift_event(d))
    # raw is not carried over the wire by design.
    assert recovered.kind == d.kind
    assert recovered.severity == d.severity
    assert recovered.detail == d.detail
    assert recovered.current_task_id == d.current_task_id
    assert recovered.current_agent_id == d.current_agent_id
    assert recovered.raw is None


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.BLOCKED,
    ],
)
def test_all_task_statuses_round_trip(status: TaskStatus) -> None:
    t = Task(id="t", title="A", status=status)
    assert from_pb_task(to_pb_task(t)).status == status


@pytest.mark.parametrize(
    "severity",
    [DriftSeverity.INFO, DriftSeverity.WARNING, DriftSeverity.CRITICAL],
)
def test_all_drift_severities_round_trip(severity: DriftSeverity) -> None:
    d = DriftEvent(kind=DriftKind.TOOL_ERROR, severity=severity)
    assert from_pb_drift_event(to_pb_drift_event(d)).severity == severity

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
    SupersessionKind,
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


def test_task_supersedes_string_round_trip_no_char_split() -> None:
    """``Task.supersedes`` survives proto round-trip as a single string.

    Regression guard for the char-split hypothesis from the goldfive#251
    audit. The proto field is a singular ``string supersedes = 9``; if a
    producer-side bug ever wired a list-iterable into it (e.g. via
    ``list.extend(str)`` instead of ``list.append(str)``), the round-trip
    would either raise on serialise or silently truncate. Pin both
    invariants:

    * ``round_trip(t).supersedes`` is the SAME str instance value, NOT a
      list of characters and NOT the empty string.
    * The pb-side field's Python type is ``str`` after parse.
    """
    t = Task(
        id="research_solar_flares_v2",
        title="Research solar flares (v2)",
        supersedes="research_solar_flares",
        supersedes_kind=SupersessionKind.CORRECT,
    )
    pb = to_pb_task(t)
    # Pb singular-string fields surface as Python str, not list[str] /
    # list[int] — verify we did not accidentally encode the field via a
    # repeated-string path that would char-iterate the value.
    assert isinstance(pb.supersedes, str)
    assert pb.supersedes == "research_solar_flares"
    recovered = from_pb_task(pb)
    assert isinstance(recovered.supersedes, str)
    assert recovered.supersedes == "research_solar_flares"
    # And: the dataclass round-trip preserves equality on the field
    # AND on supersedes_kind.
    assert recovered == t


def test_plan_supersedes_round_trip_via_plan_revised_event() -> None:
    """``Task.supersedes`` survives the full PlanRevised envelope path.

    Goes one level above ``test_task_supersedes_string_round_trip``: pins
    the field through the same pipeline a real refine takes — the steerer
    builds ``PlanRevised(plan=...)`` via ``to_pb_plan(plan)`` and emits
    it, sinks parse the bytes back. This is the path the goldfive#251
    correction-injection write-side relies on; if the round-trip char-
    splits, every dynainst correction-block render would render the
    superseded id as ``r, e, s, e, ...`` instead of the ``research_...``
    string the LLM should see.
    """
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="research_solar_flares", title="research", status=TaskStatus.COMPLETED),
            Task(
                id="correct_research_solar_flares",
                title="research (corrected)",
                supersedes="research_solar_flares",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
    )
    recovered = from_pb_plan(to_pb_plan(plan))
    assert recovered == plan
    # Spot-check the specific field: it must be a single string, never
    # a list of chars.
    target = next(t for t in recovered.tasks if t.id == "correct_research_solar_flares")
    assert isinstance(target.supersedes, str)
    assert target.supersedes == "research_solar_flares"
    assert target.supersedes_kind is SupersessionKind.CORRECT


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

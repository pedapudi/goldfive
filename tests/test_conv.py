"""Round-trip tests for goldfive.conv.

These tests require the generated proto stubs under ``goldfive.pb``
(produced by issue #3). When those stubs are not present the whole module
is skipped so ``pytest`` stays green on branches where only the Python-side
dataclasses have landed.
"""

from __future__ import annotations

import importlib.util
from typing import Any

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


def test_forecast_task_omits_kind_and_contributes_to_in_json() -> None:
    """A FORECAST task emits no ``kind`` / ``contributesTo`` JSON keys.

    AGENCY-PRESERVATION.md Stage 3 PR 10/11 carry-over — pins the zicato
    JSON-emission contract directly (rather than relying on sink-snapshot
    suites indirectly): in the default forecast plan mode the ledger
    taxonomy fields are at their proto3 value-0 / empty defaults, so
    ``MessageToJson`` omits them and forecast-mode JSON output is
    byte-identical to pre-PR-10. This is the property the cardinal
    "forecast bit-identical" rule protects.
    """
    from google.protobuf.json_format import MessageToJson

    forecast_task = Task(id="t1", title="forecast task")
    js = MessageToJson(to_pb_task(forecast_task))
    assert "kind" not in js
    assert "contributesTo" not in js


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


# ---------------------------------------------------------------------------
# Plan-descriptive-growth overlay (goldfive#423 PR 1; design doc
# ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md`` §4.1, §4.4). Round-trip the
# new ``Task.discovered`` and ``Task.discovery_identity_hash`` fields
# through the proto and pin back-compat for the missing-field path.
# ---------------------------------------------------------------------------


def test_task_discovered_field_round_trip() -> None:
    # The new fields survive the to/from proto path so sinks (PR 3:
    # harmonograf) can render discovered tasks distinctly without
    # re-reading the dataclass.
    t = Task(
        id="discovered-1",
        title="debugger_agent: locate files",
        discovered=True,
        discovery_identity_hash="abc123def4567890",
    )
    recovered = from_pb_task(to_pb_task(t))
    assert recovered == t
    assert recovered.discovered is True
    assert recovered.discovery_identity_hash == "abc123def4567890"


def test_task_discovered_default_round_trip() -> None:
    # A forecast task (discovered=False, hash="") round-trips with the
    # defaults unchanged. This is the back-compat path: every legacy
    # call site that builds a Task with no kwargs lands here.
    t = Task(id="t1", title="forecast")
    recovered = from_pb_task(to_pb_task(t))
    assert recovered.discovered is False
    assert recovered.discovery_identity_hash == ""
    assert recovered == t


def test_task_from_old_proto_without_discovered_field() -> None:
    """Old serialised events (pre-PR-1 wire format) must deserialize.

    Critical UI-safety guarantee from the brief: old harmonograf builds
    that produced events without ``discovered`` / ``discovery_identity_hash``
    must still be readable. We simulate the pre-PR-1 wire format by
    constructing a proto Task and clearing the new fields, then
    parsing the bytes back through a fresh Task message and feeding
    that to ``from_pb_task``.
    """
    pb = _pb_module_for_test()
    # Build a current-shape proto then strip the new fields by
    # round-tripping through a wire format that omits them — protobuf
    # default-fills missing scalars on parse.
    msg = pb.Task(id="t1", title="legacy", status=pb.TASK_STATUS_PENDING)
    wire = msg.SerializeToString()
    parsed = pb.Task()
    parsed.ParseFromString(wire)
    recovered = from_pb_task(parsed)
    # The defaults must light up — never accidentally discovered.
    assert recovered.discovered is False
    assert recovered.discovery_identity_hash == ""


def _pb_module_for_test() -> Any:
    """Test-local pb accessor — same shape as ``goldfive.conv._pb_module``."""
    from goldfive.pb.goldfive.v1 import types_pb2

    return types_pb2


def test_plan_with_discovered_task_round_trip() -> None:
    # End-to-end: a Plan carrying both forecast and discovered tasks
    # survives the full envelope path used by PlanRevised emit (the
    # PR 2 install-path observability story).
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="planned", title="forecast task"),
            Task(
                id="discovered-1",
                title="debugger_agent: locate files",
                discovered=True,
                discovery_identity_hash="hashabcdef123456",
            ),
        ],
        edges=[],
    )
    recovered = from_pb_plan(to_pb_plan(plan))
    assert recovered == plan
    discovered_recovered = next(t for t in recovered.tasks if t.id == "discovered-1")
    assert discovered_recovered.discovered is True
    assert discovered_recovered.discovery_identity_hash == "hashabcdef123456"
    planned_recovered = next(t for t in recovered.tasks if t.id == "planned")
    assert planned_recovered.discovered is False
    assert planned_recovered.discovery_identity_hash == ""

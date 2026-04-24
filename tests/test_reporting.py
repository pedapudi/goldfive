"""Unit tests for goldfive.reporting.

Covers:
  * ``BUILTIN_REPORTING_TOOLS`` exposes all seven canonical tools by
    name and each spec has a JSON-schema parameters block.
  * Each tool's handler drives the expected Steerer transition /
    drift hook when invoked with typical arguments.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    REPORTING_TOOL_NAMES,
    ReportingToolSpec,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
    DriftKind,
    Goal,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass

    @property
    def proto_events(self) -> list[Any]:
        """Filter out goldfive a4 dict-envelope events (refine_attempted /
        refine_failed / correlation plan_revised) so legacy assertions
        on proto-event order still hold."""
        return [e for e in self.events if hasattr(e, "WhichOneof")]


class StubPlanner:
    def __init__(self, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return self.revised

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised


def _plan() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[Task(id="t1", title="A"), Task(id="t2", title="B")],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )


def _fresh() -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=_plan(),
    )
    sink = ListSink()
    planner = StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    return steerer, session, sink, planner


def _tool(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"builtin tool {name!r} missing")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_builtin_tools_match_canonical_names() -> None:
    names = [t.name for t in BUILTIN_REPORTING_TOOLS]
    assert set(names) == set(REPORTING_TOOL_NAMES)
    assert len(names) == len(REPORTING_TOOL_NAMES)


def test_every_spec_has_schema_and_handler() -> None:
    for t in BUILTIN_REPORTING_TOOLS:
        assert isinstance(t, ReportingToolSpec)
        assert t.description
        assert t.parameters.get("type") == "object"
        assert "properties" in t.parameters
        assert callable(t.handler)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def test_report_task_started_marks_running() -> None:
    steerer, session, sink, _ = _fresh()
    out = await _tool("report_task_started").handler(
        {"task_id": "t1", "detail": "starting"}, session, steerer
    )
    assert out == {"acknowledged": True}
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    # task_started lands in the stream (followed by goldfive#251 R4
    # TaskTransitioned envelope; see test_task_transitioned_events.py).
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert "task_started" in kinds


async def test_report_task_progress_records_progress() -> None:
    steerer, session, sink, _ = _fresh()
    # goldfive#201: progress ticks are only valid on RUNNING tasks —
    # transition t1 first so the handler actually records progress
    # instead of returning invalid_transition.
    session.plan.tasks[0].status = TaskStatus.RUNNING
    await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.75, "detail": "three of four done"},
        session,
        steerer,
    )
    assert session.task_progress["t1"] == pytest.approx(0.75)
    kinds = [e.WhichOneof("payload") for e in sink.events]
    assert "task_progress" in kinds


async def test_report_task_completed_transitions_and_stores_summary() -> None:
    steerer, session, sink, _ = _fresh()
    await _tool("report_task_completed").handler(
        {
            "task_id": "t1",
            "summary": "all done",
            "artifacts": {"result": "42"},
        },
        session,
        steerer,
    )
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    assert session.completed_results["t1"] == "all done"
    completed = [
        e for e in sink.events if e.WhichOneof("payload") == "task_completed"
    ]
    assert len(completed) == 1
    assert dict(completed[0].task_completed.artifacts) == {"result": "42"}


async def test_report_task_failed_transitions_and_refines() -> None:
    steerer, session, sink, planner = _fresh()
    await _tool("report_task_failed").handler(
        {"task_id": "t1", "reason": "oops", "recoverable": True},
        session,
        steerer,
    )
    assert session.plan.tasks[0].status is TaskStatus.FAILED
    # Sequence: TaskFailed, DriftDetected. Planner.refine invoked once.
    kinds = [e.WhichOneof("payload") for e in sink.proto_events]
    assert "task_failed" in kinds and "drift_detected" in kinds
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.TASK_FAILED_RECOVERABLE


async def test_report_task_blocked_transitions_and_refines() -> None:
    steerer, session, sink, planner = _fresh()
    await _tool("report_task_blocked").handler(
        {
            "task_id": "t1",
            "blocker": "missing input",
            "needed": "CSV file",
        },
        session,
        steerer,
    )
    assert session.plan.tasks[0].status is TaskStatus.BLOCKED
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.BLOCKED


async def test_report_new_work_discovered_fires_refine() -> None:
    steerer, session, _sink, planner = _fresh()
    await _tool("report_new_work_discovered").handler(
        {
            "parent_task_id": "t1",
            "title": "dig deeper",
            "description": "validate source freshness",
            "assignee": "analyst",
        },
        session,
        steerer,
    )
    assert len(planner.refine_calls) == 1
    assert planner.refine_calls[0]["drift"].kind is DriftKind.NEW_WORK_DISCOVERED


async def test_report_plan_divergence_fires_refine_and_sets_flag() -> None:
    steerer, session, _sink, planner = _fresh()
    await _tool("report_plan_divergence").handler(
        {
            "note": "plan is stale",
            "suggested_action": "start from scratch",
        },
        session,
        steerer,
    )
    assert session.divergence_flag is True
    assert planner.refine_calls[0]["drift"].kind is DriftKind.PLAN_DIVERGENCE


async def test_handler_tolerates_missing_optional_fields() -> None:
    steerer, session, sink, _ = _fresh()
    # Only required fields — should still transition.
    await _tool("report_task_started").handler({"task_id": "t1"}, session, steerer)
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    started = [e for e in sink.events if e.WhichOneof("payload") == "task_started"]
    assert started and started[-1].task_started.detail == ""


async def test_handler_ignores_unknown_task_id_gracefully() -> None:
    steerer, session, sink, _ = _fresh()
    # Unknown task_id — the handler ack's but no event is emitted and
    # no existing task is mutated.
    out = await _tool("report_task_started").handler({"task_id": "bogus"}, session, steerer)
    assert out == {"acknowledged": True}
    assert sink.events == []
    assert all(t.status is TaskStatus.PENDING for t in session.plan.tasks)


async def test_handler_recoverable_accepts_string_bool() -> None:
    steerer, session, _sink, planner = _fresh()
    await _tool("report_task_failed").handler(
        {"task_id": "t1", "reason": "nope", "recoverable": "false"},
        session,
        steerer,
    )
    # "false" string → recoverable=False → TASK_FAILED_FATAL
    assert planner.refine_calls[0]["drift"].kind is DriftKind.TASK_FAILED_FATAL


# ---------------------------------------------------------------------------
# goldfive#237 — reroute-on-supersedes coverage
# ---------------------------------------------------------------------------


async def test_report_task_started_reroutes_to_replacement() -> None:
    """A report_task_started against a terminal+superseded id lands on the replacement."""
    steerer, session, sink, _ = _fresh()
    # Mark t1 FAILED and add a replacement superseding it.
    session.plan.tasks[0].status = TaskStatus.FAILED
    session.plan.tasks.append(
        Task(id="t1_retry", title="A retry", supersedes="t1"),
    )
    out = await _tool("report_task_started").handler(
        {"task_id": "t1", "detail": "starting retry"}, session, steerer
    )
    assert out == {"acknowledged": True}
    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["t1_retry"].status is TaskStatus.RUNNING
    assert by_id["t1"].status is TaskStatus.FAILED
    # The emitted TaskStarted event carries the replacement id.
    started = [e for e in sink.events if e.WhichOneof("payload") == "task_started"]
    assert started and started[-1].task_started.task_id == "t1_retry"

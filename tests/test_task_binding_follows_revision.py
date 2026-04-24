"""Tests for goldfive#237: task binding follows plan revisions.

Three mechanisms were mutually to blame for the "task panel shows
PENDING but the Gantt shows real activity" contradiction operators
observed in live sessions:

1. ``_emit_plan_revised`` didn't re-pin ``current_task_id`` when the
   revised plan superseded the currently-pinned task with a
   replacement. Agents kept reporting on the FAILED/CANCELLED
   original.
2. Reporting-tool handlers rejected terminal-state calls (correct)
   but didn't route them to the replacement (incorrect).
3. No machine-readable supersession link existed on ``Task`` —
   callers relied on ``_corrected`` / ``_v2`` id-suffix heuristics.

These tests exercise the fixes together: the explicit
``Task.supersedes`` link drives both re-pinning (via
``_repin_current_task_on_supersedes``) and reporting-tool rerouting
(via ``_resolve_effective_task_id``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive import orchestration_state as _ostate  # noqa: E402
from goldfive.reporting import (  # noqa: E402
    BUILTIN_REPORTING_TOOLS,
    _resolve_effective_task_id,
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


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        return None


class _StubPlanner:
    """Planner stub whose ``refine`` returns a pre-built revised plan."""

    def __init__(self, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised


def _tool(name: str):
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"builtin tool {name!r} missing")


def _plan_with_failed_research() -> Plan:
    """Mirror the live-session shape: research_solar has already FAILED."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.FAILED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="write_report")],
        revision_index=0,
    )


def _revised_with_replacement() -> Plan:
    """research_solar stays FAILED; research_solar_corrected supersedes it."""
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.FAILED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        edges=[
            TaskEdge(from_task_id="research_solar_corrected", to_task_id="write_report"),
        ],
        revision_index=1,
    )


def _drift() -> DriftEvent:
    return DriftEvent(
        kind=DriftKind.TASK_FAILED_RECOVERABLE,
        severity=DriftSeverity.WARNING,
        detail="research agent hit a dead end",
        current_task_id="research_solar",
    )


async def test_current_task_id_repins_on_revision() -> None:
    """session.current_task_id must follow a supersedes link across revision."""
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="write a report on solar")],
        plan=_plan_with_failed_research(),
    )
    # Simulate the pre-revision state: research_solar is still what the
    # agent is pinned on even though its status flipped to FAILED.
    session.current_task_id = "research_solar"

    revised = _revised_with_replacement()
    planner = _StubPlanner(revised=revised)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=planner)
    steerer._apply_revision(session, revised, _drift())
    await steerer._emit_plan_revised(session, revised, _drift(), prev_plan=None)

    assert session.current_task_id == "research_solar_corrected"


async def test_report_task_progress_routes_to_replacement() -> None:
    """report_task_progress on a terminal+superseded task lands on the replacement."""
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar report")],
        plan=_revised_with_replacement(),
    )
    # Transition the replacement into RUNNING so the progress tick is
    # a legal transition target.
    for t in session.plan.tasks:
        if t.id == "research_solar_corrected":
            t.status = TaskStatus.RUNNING

    sink = ListSink()
    planner = _StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    out = await _tool("report_task_progress").handler(
        {"task_id": "research_solar", "fraction": 0.42, "detail": "partway"},
        session,
        steerer,
    )
    assert out == {"acknowledged": True}
    assert session.task_progress["research_solar_corrected"] == pytest.approx(0.42)
    assert "research_solar" not in session.task_progress


async def test_report_task_completed_routes_to_replacement() -> None:
    """report_task_completed on a terminal+superseded task completes the replacement."""
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar report")],
        plan=_revised_with_replacement(),
    )
    sink = ListSink()
    planner = _StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    await _tool("report_task_completed").handler(
        {
            "task_id": "research_solar",
            "summary": "solar options landed",
            "artifacts": {"report.md": "s3://bucket/solar"},
        },
        session,
        steerer,
    )
    # The replacement is what actually transitioned; the superseded
    # task stays FAILED.
    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["research_solar_corrected"].status is TaskStatus.COMPLETED
    assert by_id["research_solar"].status is TaskStatus.FAILED
    assert session.completed_results["research_solar_corrected"] == "solar options landed"


async def test_report_rejected_when_terminal_but_no_supersession() -> None:
    """Without a supersedes link, the legacy invalid_transition rejection stands."""
    # Plan with a FAILED task but NO replacement — the rerouter must
    # leave the id alone and the handler must return the pre-#237
    # ``invalid_transition`` shape.
    plan = _plan_with_failed_research()
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar report")],
        plan=plan,
    )
    sink = ListSink()
    planner = _StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    out = await _tool("report_task_progress").handler(
        {"task_id": "research_solar", "fraction": 0.5},
        session,
        steerer,
    )
    assert out.get("acknowledged") is False
    assert out.get("error") == "invalid_transition"
    assert out.get("current_status") == TaskStatus.FAILED.value


async def test_multiple_supersessions_all_repin() -> None:
    """Two pinned task_ids, both superseded → state pin follows."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.FAILED),
            Task(id="a2", title="A retry", status=TaskStatus.PENDING, supersedes="a"),
            Task(id="b", title="B", status=TaskStatus.FAILED),
            Task(id="b2", title="B retry", status=TaskStatus.PENDING, supersedes="b"),
        ],
        edges=[],
        revision_index=1,
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="two tracks")],
        plan=plan,
    )
    session.current_task_id = "a"
    # orchestration-state key also pins the old id (mirrors adapter behaviour).
    session.state[_ostate.KEY_CURRENT_TASK_ID] = "b"

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())
    steerer._repin_current_task_on_supersedes(session, plan)

    assert session.current_task_id == "a2"
    assert session.state[_ostate.KEY_CURRENT_TASK_ID] == "b2"


async def test_no_supersedes_field_no_change() -> None:
    """Revisions with no supersedes link leave the pin alone."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="T1", status=TaskStatus.RUNNING),
            Task(id="t2", title="T2", status=TaskStatus.PENDING),
        ],
        edges=[],
        revision_index=1,
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="one track")],
        plan=plan,
    )
    session.current_task_id = "t1"
    session.state[_ostate.KEY_CURRENT_TASK_ID] = "t1"

    steerer = DefaultSteerer()
    steerer.bind(sinks=[ListSink()], planner=_StubPlanner())
    steerer._repin_current_task_on_supersedes(session, plan)

    assert session.current_task_id == "t1"
    assert session.state[_ostate.KEY_CURRENT_TASK_ID] == "t1"


async def test_resolve_effective_task_id_chain() -> None:
    """A → B → C supersession chain collapses to C."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.FAILED),
            Task(id="b", title="B", status=TaskStatus.FAILED, supersedes="a"),
            Task(id="c", title="C", status=TaskStatus.PENDING, supersedes="b"),
        ],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="chain")],
        plan=plan,
    )
    assert _resolve_effective_task_id(session, "a") == "c"
    # b is also terminal+superseded, lands on c.
    assert _resolve_effective_task_id(session, "b") == "c"
    # c is live, returns itself.
    assert _resolve_effective_task_id(session, "c") == "c"
    # unknown id returned unchanged (handler will see "unknown task").
    assert _resolve_effective_task_id(session, "unknown") == "unknown"


async def test_supersedes_field_flows_through_plan_submission() -> None:
    """Task.supersedes round-trips through the proto conv / plan validator."""
    from goldfive.conv import from_pb_plan, to_pb_plan

    plan = _revised_with_replacement()
    pb_plan = to_pb_plan(plan)
    # Proto serialisation keeps the link.
    for pb_task in pb_plan.tasks:
        if pb_task.id == "research_solar_corrected":
            assert pb_task.supersedes == "research_solar"
    # Round-trip dataclass -> pb -> dataclass preserves the field.
    roundtripped = from_pb_plan(pb_plan)
    by_id = {t.id: t for t in roundtripped.tasks}
    assert by_id["research_solar_corrected"].supersedes == "research_solar"
    assert by_id["research_solar"].supersedes == ""
    # Plan.validate on a revised plan carrying supersedes still passes.
    plan.validate(for_revision=True, prior=_plan_with_failed_research())

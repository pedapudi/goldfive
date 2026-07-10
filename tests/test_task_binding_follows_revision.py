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

from goldfive import state_store as _ostate  # noqa: E402
from goldfive.config import SteeringConfig  # noqa: E402
from goldfive.plan_reviser import PlanReviser  # noqa: E402
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
    SupersessionKind,
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
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    steerer.bind(sinks=[ListSink()], planner=planner)
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer.plans._apply_revision(session, revised, _drift())
    await steerer.plans._emit_plan_revised(session, revised, _drift(), prev_plan=None)

    assert session.current_task_id == "research_solar_corrected"


async def test_report_task_progress_routes_to_replacement() -> None:
    """report_task_progress on a terminal+superseded task lands on the replacement."""
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar report")],
        plan=_revised_with_replacement(),
    )
    # Transition the replacement into RUNNING so the progress tick is
    # a legal transition target. goldfive#247: derive via helper.
    from tests._immutable_plan_helpers import force_task_status

    force_task_status(session, "research_solar_corrected", TaskStatus.RUNNING)

    sink = ListSink()
    planner = _StubPlanner()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)

    out = await _tool("report_task_progress").handler(
        {"task_id": "research_solar", "fraction": 0.42, "detail": "partway"},
        session,
        steerer,
    )
    # F1 directive ack on the rerouted progress tick.
    assert out["acknowledged"] is True
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
    steerer.plans._repin_current_task_on_supersedes(session, plan)

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
    steerer.plans._repin_current_task_on_supersedes(session, plan)

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


# ---------------------------------------------------------------------------
# goldfive#251: SupersessionKind + Option B append-as-correction
# ---------------------------------------------------------------------------


def _plan_with_completed_research() -> Plan:
    """Like _plan_with_failed_research but the old task already COMPLETED.

    This is the live-evidence shape: refine-superseded a task that was
    legitimately done (agent signalled completion) but the output was
    drift-contaminated and a correction re-does the work.
    """
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
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


async def test_integrate_correction_supersedes_keeps_old_task() -> None:
    """CORRECT-kind supersedes: old task stays in the plan as a COMPLETED node."""
    prior = _plan_with_completed_research()
    # Refine output: new correction-task supersedes the COMPLETED old task.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        # Raw refine edges: downstream still points at old. Integration
        # should rewire write_report to hang off the correction.
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="write_report")],
        revision_index=1,
    )

    revised = PlanReviser._integrate_correction_supersedes(revised)

    by_id = {t.id: t for t in revised.tasks}
    # Old task preserved with its COMPLETED status intact.
    assert by_id["research_solar"].status is TaskStatus.COMPLETED
    # New correction-task added; the old is its upstream.
    edges_set = {(e.from_task_id, e.to_task_id) for e in revised.edges}
    assert ("research_solar", "research_solar_corrected") in edges_set
    # write_report now flows through the correction, not directly from old.
    assert ("research_solar_corrected", "write_report") in edges_set
    assert ("research_solar", "write_report") not in edges_set
    # Plan still validates as a revision.
    revised.validate(for_revision=True, prior=prior)


async def test_integrate_correction_supersedes_replace_kind_no_dag_rewrite() -> None:
    """REPLACE-kind supersedes: _integrate_correction_supersedes does NOTHING.

    The pre-#251 REPLACE behaviour (refiner produces a FAILED old task
    and a PENDING replacement; downstream edges already rewritten by
    the refiner) is preserved verbatim — this method is a CORRECT-only
    topology hook.
    """
    revised = _revised_with_replacement()
    original_edges = [(e.from_task_id, e.to_task_id) for e in revised.edges]
    revised = PlanReviser._integrate_correction_supersedes(revised)
    # No edge mutations.
    assert [(e.from_task_id, e.to_task_id) for e in revised.edges] == original_edges


async def test_validator_coerces_replace_to_correct_on_completed() -> None:
    """LLM says REPLACE but the old task is COMPLETED → coerced to CORRECT."""
    from goldfive.planner import _normalize_supersession_kinds

    prior = _plan_with_completed_research()
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                # LLM got it wrong — old task is COMPLETED, not retryable.
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    revised = _normalize_supersession_kinds(revised, prior=prior)
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["research_solar_corrected"].supersedes_kind is SupersessionKind.CORRECT


async def test_validator_coerces_correct_to_replace_on_pending() -> None:
    """LLM says CORRECT but the old task is PENDING → coerced to REPLACE."""
    from goldfive.planner import _normalize_supersession_kinds

    prior = _plan_with_failed_research()
    # Swap research_solar back to PENDING for this scenario.
    # goldfive#247: Plan is frozen — derive a new prior via with_task_status.
    from goldfive.types import with_task_status as _wts

    prior = _wts(prior, prior.tasks[0].id, TaskStatus.PENDING)
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_v2",
                title="Research solar options (reshaped)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,  # wrong for PENDING
            ),
        ],
        edges=[],
        revision_index=1,
    )
    revised = _normalize_supersession_kinds(revised, prior=prior)
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["research_solar_v2"].supersedes_kind is SupersessionKind.REPLACE


async def test_validator_clears_kind_with_empty_target() -> None:
    """supersedes_kind set but supersedes is empty → kind cleared."""
    from goldfive.planner import _normalize_supersession_kinds

    prior = _plan_with_failed_research()
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="new_task",
                title="Fresh work",
                status=TaskStatus.PENDING,
                supersedes="",  # empty
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    revised = _normalize_supersession_kinds(revised, prior=prior)
    assert revised.tasks[0].supersedes_kind is SupersessionKind.UNSPECIFIED


async def test_validator_unspecified_resolves_from_status() -> None:
    """supersedes set, kind UNSPECIFIED → filled from old task's status."""
    from goldfive.planner import _normalize_supersession_kinds

    prior = _plan_with_completed_research()
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="old",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="research_solar_corrected",
                title="new",
                status=TaskStatus.PENDING,
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.UNSPECIFIED,
            ),
        ],
        edges=[],
        revision_index=1,
    )
    revised = _normalize_supersession_kinds(revised, prior=prior)
    by_id = {t.id: t for t in revised.tasks}
    assert by_id["research_solar_corrected"].supersedes_kind is SupersessionKind.CORRECT


async def test_downstream_edge_rewrite_on_correct_chain() -> None:
    """A -> B -> C; refine adds A' supersedes A kind=CORRECT.

    Post-integration: edges become A -> A' -> B -> C (B now depends on
    A', not A directly). The old A stays COMPLETED.
    """
    prior = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.COMPLETED),
            Task(id="b", title="B", status=TaskStatus.PENDING),
            Task(id="c", title="C", status=TaskStatus.PENDING),
        ],
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="b", to_task_id="c"),
        ],
        revision_index=0,
    )
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.COMPLETED),
            Task(
                id="a_prime",
                title="A corrected",
                status=TaskStatus.PENDING,
                supersedes="a",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(id="b", title="B", status=TaskStatus.PENDING),
            Task(id="c", title="C", status=TaskStatus.PENDING),
        ],
        # Refiner left edges pointing at the old A.
        edges=[
            TaskEdge(from_task_id="a", to_task_id="b"),
            TaskEdge(from_task_id="b", to_task_id="c"),
        ],
        revision_index=1,
    )

    revised = PlanReviser._integrate_correction_supersedes(revised)

    edges_set = {(e.from_task_id, e.to_task_id) for e in revised.edges}
    # Chain is now a -> a_prime -> b -> c.
    assert ("a", "a_prime") in edges_set
    assert ("a_prime", "b") in edges_set
    assert ("b", "c") in edges_set
    # Old direct a -> b edge is rewritten away.
    assert ("a", "b") not in edges_set
    # Plan structurally valid as a revision.
    revised.validate(for_revision=True, prior=prior)


async def test_report_on_completed_old_task_under_correct_does_not_route() -> None:
    """A report on a COMPLETED old task under CORRECT supersedes does NOT route.

    Contrast with the REPLACE-kind test above where a report on a
    FAILED old task reroutes to the replacement. Under CORRECT the old
    task's COMPLETED status is historical fact — the handler should
    see the old id unchanged and the existing idempotent completed-
    task path takes over.
    """
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="research_solar_corrected")],
        revision_index=1,
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar")],
        plan=plan,
    )
    # _resolve_effective_task_id must NOT route research_solar to the correction.
    assert _resolve_effective_task_id(session, "research_solar") == "research_solar"


async def test_report_on_failed_old_task_under_replace_still_routes() -> None:
    """REPLACE-kind keeps the pre-#251 routing behaviour intact."""
    plan = Plan(
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
                id="research_solar_v2",
                title="Research solar options (retry)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
        ],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar")],
        plan=plan,
    )
    # REPLACE continues to reroute — old behaviour preserved.
    assert _resolve_effective_task_id(session, "research_solar") == "research_solar_v2"


async def test_supersedes_kind_roundtrips_through_proto() -> None:
    """Task.supersedes_kind survives to-pb / from-pb round-trip."""
    from goldfive.conv import from_pb_plan, to_pb_plan

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="a", title="A", status=TaskStatus.COMPLETED),
            Task(
                id="a_prime",
                title="A corrected",
                status=TaskStatus.PENDING,
                supersedes="a",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="b",
                title="B",
                status=TaskStatus.PENDING,
                supersedes="a",
                supersedes_kind=SupersessionKind.REPLACE,
            ),
            Task(id="c", title="C", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    pb_plan = to_pb_plan(plan)
    # Proto side carries the values.
    by_pb = {t.id: t for t in pb_plan.tasks}
    from goldfive.pb.goldfive.v1 import types_pb2

    assert by_pb["a_prime"].supersedes_kind == types_pb2.SUPERSESSION_KIND_CORRECT
    assert by_pb["b"].supersedes_kind == types_pb2.SUPERSESSION_KIND_REPLACE
    assert by_pb["c"].supersedes_kind == types_pb2.SUPERSESSION_KIND_UNSPECIFIED
    # Round-trip preserves the dataclass enum.
    rt = from_pb_plan(pb_plan)
    by_id = {t.id: t for t in rt.tasks}
    assert by_id["a_prime"].supersedes_kind is SupersessionKind.CORRECT
    assert by_id["b"].supersedes_kind is SupersessionKind.REPLACE
    assert by_id["c"].supersedes_kind is SupersessionKind.UNSPECIFIED


async def test_correct_integration_via_emit_plan_revised_end_to_end() -> None:
    """End-to-end: _emit_plan_revised wires the CORRECT topology in.

    Feeds the steerer a refine output with a CORRECT-kind supersedes
    link whose downstream edges still point at the old task. After
    emit the plan has been rewired and the PlanRevised event carries
    the corrected topology on the wire.
    """
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="solar report")],
        plan=_plan_with_completed_research(),
    )
    # Revised plan as the LLM would emit it pre-integration: old
    # downstream edge (research_solar -> write_report) still present.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research_solar",
                title="Research solar options",
                status=TaskStatus.COMPLETED,
                assignee_agent_id="research_agent",
            ),
            Task(
                id="research_solar_corrected",
                title="Research solar options (corrected)",
                status=TaskStatus.PENDING,
                assignee_agent_id="research_agent",
                supersedes="research_solar",
                supersedes_kind=SupersessionKind.CORRECT,
            ),
            Task(
                id="write_report",
                title="Write final report",
                status=TaskStatus.PENDING,
                assignee_agent_id="writer_agent",
            ),
        ],
        edges=[TaskEdge(from_task_id="research_solar", to_task_id="write_report")],
        revision_index=1,
    )

    sink = ListSink()
    planner = _StubPlanner(revised=revised)
    steerer = DefaultSteerer(steering_config=SteeringConfig(observation_only=False))
    steerer.bind(sinks=[sink], planner=planner)
    # goldfive#247: rebind to the stamped instance.
    # goldfive#255: _apply_revision now returns ``(revised, was_installed)``.
    revised, _was_installed = steerer.plans._apply_revision(session, revised, _drift())
    await steerer.plans._emit_plan_revised(session, revised, _drift(), prev_plan=None)

    # Session plan now has the corrected topology.
    edges_set = {(e.from_task_id, e.to_task_id) for e in session.plan.edges}
    assert ("research_solar", "research_solar_corrected") in edges_set
    assert ("research_solar_corrected", "write_report") in edges_set
    assert ("research_solar", "write_report") not in edges_set
    # Old task retained with COMPLETED status.
    by_id = {t.id: t for t in session.plan.tasks}
    assert by_id["research_solar"].status is TaskStatus.COMPLETED

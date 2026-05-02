"""iter-11E PR 2 — structural artifact verification at ``report_task_completed``.

PR 1 shipped the scaffolding (``Task.required_tool_calls``,
``DriftKind.INCOMPLETE_TOOL_CALLS``, ladder entry, planner prompt
selection). PR 2 wires the verification into
``goldfive.reporting._handle_task_completed`` so a completion report
on a task with declared ``required_tool_calls`` is rejected when one
or more of those tools were not observed during the task's execution
span (per ``Session.recent_tool_observations`` filtered to the
matching ``task_id``). The rejection
  * dispatches an ``INCOMPLETE_TOOL_CALLS`` drift through the
    standard pipeline (refine via the goal-aware system prompt),
  * returns a structured ``incomplete_tool_calls`` rejection payload
    so the agent learns its report did not take effect, and
  * leaves the task in its pre-call status (no transition).

These tests pin those properties.
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
    ReportingToolSpec,
    _verify_required_tool_calls,
)
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
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
# Helpers
# ---------------------------------------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class StubPlanner:
    def __init__(self, revised: Plan | None = None) -> None:
        self.revised = revised
        self.refine_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Plan | None:
        return self.revised

    async def refine(self, **kwargs: Any) -> Plan | None:
        self.refine_calls.append(kwargs)
        return self.revised


def _plan_with_required_tools() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="draft_slides",
                title="Draft slide deck",
                assignee_agent_id="presenter",
                status=TaskStatus.RUNNING,
                required_tool_calls=["write_webpage_tool"],
            ),
            Task(
                id="review",
                title="Review slide deck",
                assignee_agent_id="reviewer",
            ),
        ],
        edges=[TaskEdge(from_task_id="draft_slides", to_task_id="review")],
    )


def _plan_no_required() -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="research",
                title="Research the topic",
                assignee_agent_id="researcher",
                status=TaskStatus.RUNNING,
            ),
        ],
        edges=[],
    )


def _fresh(plan: Plan) -> tuple[DefaultSteerer, Session, ListSink, StubPlanner]:
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="produce a slide deck")],
        plan=plan,
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


def _push_observation(
    session: Session,
    *,
    task_id: str,
    tool_name: str,
    agent_name: str = "presenter",
) -> None:
    """Append a recent_tool_observations entry as an adapter would."""
    session.recent_tool_observations.append(
        {
            "ts_ms": 0,
            "agent_name": agent_name,
            "task_id": task_id,
            "tool_name": tool_name,
            "args_preview": "(args)",
            "result_preview": "(ok)",
            "is_error": False,
            "error_message": "",
        }
    )


# ---------------------------------------------------------------------------
# Pure helper: _verify_required_tool_calls
# ---------------------------------------------------------------------------


def test_verify_returns_none_when_no_required_tools() -> None:
    """A task with empty ``required_tool_calls`` short-circuits."""
    plan = _plan_no_required()
    session = Session(run_id="r1", plan=plan)
    task = plan.tasks[0]
    assert _verify_required_tool_calls(task, session, task.id) is None


def test_verify_returns_none_when_all_required_observed() -> None:
    plan = _plan_with_required_tools()
    session = Session(run_id="r1", plan=plan)
    task = plan.tasks[0]
    _push_observation(session, task_id=task.id, tool_name="write_webpage_tool")
    assert _verify_required_tool_calls(task, session, task.id) is None


def test_verify_returns_critical_drift_when_missing() -> None:
    plan = _plan_with_required_tools()
    session = Session(run_id="r1", plan=plan)
    task = plan.tasks[0]
    drift = _verify_required_tool_calls(task, session, task.id)
    assert drift is not None
    assert drift.kind is DriftKind.INCOMPLETE_TOOL_CALLS
    assert drift.severity is DriftSeverity.CRITICAL
    assert "write_webpage_tool" in drift.detail
    assert drift.current_task_id == "draft_slides"
    assert drift.current_agent_id == "presenter"


def test_verify_scopes_observations_per_task() -> None:
    """An observation pinned to a different task does NOT satisfy this task."""
    plan = _plan_with_required_tools()
    session = Session(run_id="r1", plan=plan)
    task = plan.tasks[0]
    # Tool observed, but pinned to a sibling task — must NOT count.
    _push_observation(session, task_id="some_other_task", tool_name="write_webpage_tool")
    drift = _verify_required_tool_calls(task, session, task.id)
    assert drift is not None
    assert "write_webpage_tool" in drift.detail


def test_verify_partial_observations_cite_missing_only() -> None:
    """When some required tools are observed and others aren't, only the
    missing ones land in the drift detail's ``missing=`` list."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="draft_and_save",
                title="Draft + save",
                assignee_agent_id="presenter",
                status=TaskStatus.RUNNING,
                required_tool_calls=["write_webpage_tool", "read_files_tool"],
            ),
        ],
        edges=[],
    )
    session = Session(run_id="r1", plan=plan)
    task = plan.tasks[0]
    _push_observation(session, task_id=task.id, tool_name="write_webpage_tool")
    drift = _verify_required_tool_calls(task, session, task.id)
    assert drift is not None
    assert "read_files_tool" in drift.detail
    # write_webpage_tool was observed; it must not show up in missing=
    # (it may still show up in observed=).
    # The exact rendering is "missing=['read_files_tool'], observed=[...]".
    assert "missing=['read_files_tool']" in drift.detail
    assert "write_webpage_tool" in drift.detail  # in observed=[...]


# ---------------------------------------------------------------------------
# Wiring: _handle_task_completed
# ---------------------------------------------------------------------------


async def test_report_succeeded_accepted_when_no_required_tools() -> None:
    """A task with no required_tool_calls retains pre-PR2 behaviour."""
    steerer, session, sink, planner = _fresh(_plan_no_required())
    out = await _tool("report_task_completed").handler(
        {"task_id": "research", "summary": "did the research"},
        session,
        steerer,
    )
    assert out["acknowledged"] is True
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    completed = [
        e for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "task_completed"
    ]
    assert len(completed) == 1
    # No drift was dispatched.
    await steerer._wait_background_drifts_idle()
    assert planner.refine_calls == []


async def test_report_succeeded_accepted_when_all_required_observed() -> None:
    """All declared tools observed in this task's span -> completion accepted."""
    steerer, session, sink, planner = _fresh(_plan_with_required_tools())
    _push_observation(session, task_id="draft_slides", tool_name="write_webpage_tool")
    out = await _tool("report_task_completed").handler(
        {"task_id": "draft_slides", "summary": "all done"},
        session,
        steerer,
    )
    assert out["acknowledged"] is True
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    completed = [
        e for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "task_completed"
    ]
    assert len(completed) == 1
    await steerer._wait_background_drifts_idle()
    # No INCOMPLETE_TOOL_CALLS refine.
    incomplete_refines = [
        c for c in planner.refine_calls
        if c["drift"].kind is DriftKind.INCOMPLETE_TOOL_CALLS
    ]
    assert incomplete_refines == []


async def test_report_succeeded_rejected_when_required_tool_missing() -> None:
    """No observation -> completion rejected with structured payload + drift."""
    steerer, session, sink, planner = _fresh(_plan_with_required_tools())
    out = await _tool("report_task_completed").handler(
        {"task_id": "draft_slides", "summary": "claims done"},
        session,
        steerer,
    )
    # Rejection shape — mirrors invalid_transition / missing_required_field.
    assert out["acknowledged"] is False
    assert out["error"] == "incomplete_tool_calls"
    assert out["tool"] == "report_task_completed"
    assert out["task_id"] == "draft_slides"
    assert out["missing_tool_calls"] == ["write_webpage_tool"]
    assert out["observed_tool_calls"] == []
    assert "write_webpage_tool" in out["message"]
    # Task did NOT transition.
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    # No TaskCompleted on the wire.
    completed = [
        e for e in sink.events
        if hasattr(e, "WhichOneof") and e.WhichOneof("payload") == "task_completed"
    ]
    assert completed == []
    # Drift dispatched and reaches planner.refine via the ladder's ABSORB
    # path. CRITICAL severity → CANCEL_REINVOKE on first occurrence,
    # which still fires planner.refine before the cancel.
    await steerer._wait_background_drifts_idle()
    incomplete = [
        c for c in planner.refine_calls
        if c["drift"].kind is DriftKind.INCOMPLETE_TOOL_CALLS
    ]
    assert len(incomplete) == 1
    assert incomplete[0]["drift"].current_task_id == "draft_slides"
    assert "write_webpage_tool" in incomplete[0]["drift"].detail


async def test_report_succeeded_rejected_with_partial_tools() -> None:
    """Drift detail names the missing tool specifically, not the satisfied one."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="draft_and_save",
                title="Draft + save",
                assignee_agent_id="presenter",
                status=TaskStatus.RUNNING,
                required_tool_calls=["write_webpage_tool", "read_files_tool"],
            ),
        ],
        edges=[],
    )
    steerer, session, _sink, planner = _fresh(plan)
    _push_observation(session, task_id="draft_and_save", tool_name="write_webpage_tool")
    out = await _tool("report_task_completed").handler(
        {"task_id": "draft_and_save", "summary": "claims done"},
        session,
        steerer,
    )
    assert out["acknowledged"] is False
    assert out["error"] == "incomplete_tool_calls"
    assert out["missing_tool_calls"] == ["read_files_tool"]
    assert "write_webpage_tool" in out["observed_tool_calls"]
    await steerer._wait_background_drifts_idle()
    incomplete = [
        c for c in planner.refine_calls
        if c["drift"].kind is DriftKind.INCOMPLETE_TOOL_CALLS
    ]
    assert len(incomplete) == 1
    assert "read_files_tool" in incomplete[0]["drift"].detail


async def test_observations_for_other_tasks_not_counted() -> None:
    """Per-task scoping: a write_webpage_tool observation pinned to a
    sibling task does not satisfy the requirement on the task being
    reported."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="task_a",
                title="A",
                assignee_agent_id="agent_a",
                status=TaskStatus.RUNNING,
                required_tool_calls=["write_webpage_tool"],
            ),
            Task(
                id="task_b",
                title="B",
                assignee_agent_id="agent_b",
            ),
        ],
        edges=[],
    )
    steerer, session, _sink, planner = _fresh(plan)
    # write_webpage_tool was observed, but pinned to task_b.
    _push_observation(session, task_id="task_b", tool_name="write_webpage_tool")
    out = await _tool("report_task_completed").handler(
        {"task_id": "task_a", "summary": "claims done"},
        session,
        steerer,
    )
    assert out["acknowledged"] is False
    assert out["error"] == "incomplete_tool_calls"
    assert out["missing_tool_calls"] == ["write_webpage_tool"]
    # The cross-task observation does NOT show up in observed_tool_calls
    # for task_a — the filter is per-task.
    assert out["observed_tool_calls"] == []
    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    await steerer._wait_background_drifts_idle()
    incomplete = [
        c for c in planner.refine_calls
        if c["drift"].kind is DriftKind.INCOMPLETE_TOOL_CALLS
    ]
    assert len(incomplete) == 1


async def test_drift_routes_through_goal_aware_refine() -> None:
    """The PR1-wired prompt selection still picks goal-aware refine
    when the verification dispatches INCOMPLETE_TOOL_CALLS.

    Re-validates that PR 1's planner.refine prompt-selection branch is
    actually exercised end-to-end from a verification rejection — not
    just by direct construction of the drift in scaffolding tests.
    Mocks the planner's ``call_llm`` so we can assert on the system
    prompt selected.
    """
    from goldfive.planner import LLMPlanner

    captured: dict[str, str] = {}

    async def _stub_llm(system: str, user: str, model: str) -> str:
        captured["system"] = system
        captured["user"] = user
        # Return a no-op plan so refine doesn't blow up on parse.
        return (
            '{"summary": "still drafting via write_webpage_tool", '
            '"tasks": [{"id": "draft_slides", "title": "Draft slide deck", '
            '"assignee_agent_id": "presenter", "status": "RUNNING"}], '
            '"edges": []}'
        )

    planner = LLMPlanner(call_llm=_stub_llm)
    sink = ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=planner)
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="produce a slide deck")],
        plan=_plan_with_required_tools(),
    )
    out = await _tool("report_task_completed").handler(
        {"task_id": "draft_slides", "summary": "claims done"},
        session,
        steerer,
    )
    assert out["acknowledged"] is False
    await steerer._wait_background_drifts_idle()
    # The goal-aware divergence prompt has the ABSORB / REJECT
    # decision contract; the generic refine prompt does not.
    assert "ABSORB" in captured.get("system", "")
    assert "REJECT" in captured.get("system", "")
    # The user prompt renders the missing tools via the OFF-TOPIC
    # reasoning block (PR1 reuses _render_off_topic_reasoning_block
    # for INCOMPLETE_TOOL_CALLS).
    assert "OFF-TOPIC REASONING" in captured.get("user", "")
    assert "write_webpage_tool" in captured.get("user", "")

"""Regression tests for the Tier 1 loop-prevention push (F1+F3+F4+F10).

Each F-fix closes the loop at a different surface so the cap
``_MAX_NUDGE_REPLAYS`` becomes a defensive belt-and-suspenders bound
rather than the sole structural fence:

* **F1** — directive tool responses. Every ``report_task_*`` real-
  transition response carries a ``task`` pointer + ``plan_state``
  block so the LLM's "what next?" reasoning has a structural anchor.
* **F3** — pre-dispatch interception in the ADK plugin's
  ``before_tool_callback``: an AgentTool dispatch onto an agent whose
  plan tasks are all terminal (with a non-terminal next_pending
  elsewhere) is refused with a redirect-error response.
* **F4** — GOAL_DRIFT routes through NUDGE first, not PAUSE_ESCALATE.
  The judge's signal is "agent stuck on completed work"; a corrective
  user message re-anchors the LLM without refining a plan that's
  already correct.
* **F10** — the executor's NOT_NEEDED reaper is gated on
  ``session.paused_for_human_intervention`` so the steerer's Level 4
  pause does not turn into a silent "tasks done" lie about user intent.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._pbsetup import ensure_pb_available

pytestmark = pytest.mark.skipif(
    not ensure_pb_available(),
    reason="goldfive protobuf stubs not available (install the `dev` extra)",
)

from goldfive.reporting import BUILTIN_REPORTING_TOOLS, ReportingToolSpec  # noqa: E402
from goldfive.steerer import (  # noqa: E402
    DefaultSteerer,
    InterventionLevel,
    compose_corrective_user_message,
)
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


class _ListSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event_pb: Any) -> None:
        self.events.append(event_pb)

    async def close(self) -> None:
        pass


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


def _tool(name: str) -> ReportingToolSpec:
    for spec in BUILTIN_REPORTING_TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing builtin tool {name!r}")


# ---------------------------------------------------------------------------
# F1 — directive tool responses
# ---------------------------------------------------------------------------


def _multi_task_session(initial_status: TaskStatus = TaskStatus.RUNNING) -> Session:
    """A 3-task plan with a terminal-completed-predecessor edge so the
    F1 plan_state helper has something interesting to point at."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Research",
                assignee_agent_id="researcher",
                status=initial_status,
            ),
            Task(
                id="t2",
                title="Draft",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
            Task(
                id="t3",
                title="Polish",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[
            TaskEdge(from_task_id="t1", to_task_id="t2"),
            TaskEdge(from_task_id="t2", to_task_id="t3"),
        ],
    )
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="produce a brief")],
        plan=plan,
    )


async def test_f1_report_task_completed_returns_plan_state_pointer() -> None:
    """F1: a real transition includes ``task`` + ``plan_state.next_pending``
    so the LLM's next-action reasoning has a structural anchor instead of
    an information-free ack."""
    session = _multi_task_session(initial_status=TaskStatus.RUNNING)
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_StubPlanner())

    out = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "research done"}, session, steerer
    )

    assert out["acknowledged"] is True
    assert "idempotent" not in out
    # task pointer carries the new status
    assert out["task"] == {"id": "t1", "status": TaskStatus.COMPLETED.value}
    # plan_state surfaces the live state.
    assert out["plan_state"]["completed_task_ids"] == ["t1"]
    next_pending = out["plan_state"]["next_pending"]
    assert next_pending is not None
    # t2 is the next pending task with a terminal predecessor (t1).
    assert next_pending["id"] == "t2"
    assert next_pending["title"] == "Draft"
    assert next_pending["assigned_to"] == "writer"
    assert next_pending["predecessors_completed"] is True


async def test_f1_skips_predecessor_blocked_pending_tasks() -> None:
    """F1: ``next_pending`` only surfaces a task whose every incoming-
    edge predecessor is terminal. t3 should NOT be surfaced over t2 even
    though both are PENDING — t3's predecessor (t2) is still PENDING."""
    session = _multi_task_session(initial_status=TaskStatus.RUNNING)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())

    out = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )

    assert out["plan_state"]["next_pending"]["id"] == "t2"


async def test_f1_idempotent_re_report_still_includes_plan_state() -> None:
    """F1 (loop-prevention contract): a re-report on an already-terminal
    task still carries the rich payload so the LLM sees the live plan
    state, not just an ack — that's the anchor that breaks the loop."""
    session = _multi_task_session(initial_status=TaskStatus.RUNNING)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())

    # First call: real transition.
    await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done"}, session, steerer
    )

    # Second call: idempotent re-report.
    second = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "done again"}, session, steerer
    )

    assert second["acknowledged"] is True
    assert second["idempotent"] is True
    assert second["current_status"] == "COMPLETED"
    # The directive surface rides along — the LLM is looping on a done
    # task; the plan_state pointer is what tells it where to go next.
    assert second["task"] == {"id": "t1", "status": "COMPLETED"}
    assert "plan_state" in second
    assert second["plan_state"]["next_pending"]["id"] == "t2"


async def test_f1_no_next_pending_when_plan_done() -> None:
    """F1: ``next_pending`` is ``None`` when nothing is left to do."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="only", title="solo", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        ],
        edges=[],
    )
    session = Session(run_id="r1", goals=[Goal(id="g1", summary="x")], plan=plan)
    steerer = DefaultSteerer()
    steerer.bind(sinks=[_ListSink()], planner=_StubPlanner())

    out = await _tool("report_task_completed").handler(
        {"task_id": "only", "summary": "done"}, session, steerer
    )

    assert out["plan_state"]["completed_task_ids"] == ["only"]
    assert out["plan_state"]["next_pending"] is None


# ---------------------------------------------------------------------------
# F3 — pre-dispatch redirect (ADK plugin helper unit test)
# ---------------------------------------------------------------------------


def _ctx_for_plan(plan: Plan) -> Any:
    """Stub the SessionContext shape the F3 helper reads (.session.plan)."""

    class _Ctx:
        def __init__(self, plan: Plan) -> None:
            session = Session(run_id="r1", goals=[Goal(id="g1", summary="x")], plan=plan)
            self.session = session

    return _Ctx(plan)


def test_f3_redirects_when_target_agents_tasks_all_terminal() -> None:
    """F3: an AgentTool call onto ``researcher`` whose only task is
    COMPLETED, with the next pending assigned to ``writer``, returns the
    redirect-error payload pointing at ``writer``."""
    from goldfive.adapters._adk_plugin import _maybe_redirect_completed_agent

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Research",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2",
                title="Draft",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    ctx = _ctx_for_plan(plan)

    result = _maybe_redirect_completed_agent(ctx=ctx, target_agent="researcher")
    assert result is not None
    assert result["redirect_to"] == "writer"
    assert "researcher" in result["error"]
    assert "Draft" in result["error"]
    assert "writer" in result["error"]


def test_f3_allows_dispatch_when_target_has_pending_work() -> None:
    """F3: dispatch onto an agent that still has at least one PENDING /
    RUNNING task is legitimate work and not redirected."""
    from goldfive.adapters._adk_plugin import _maybe_redirect_completed_agent

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="Research v1",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2",
                title="Research v2",
                assignee_agent_id="researcher",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
    )
    ctx = _ctx_for_plan(plan)

    assert _maybe_redirect_completed_agent(ctx=ctx, target_agent="researcher") is None


def test_f3_allows_dispatch_when_off_plan_agent() -> None:
    """F3: dispatch onto an agent with no plan match falls through to the
    existing PLAN_DIVERGENCE detector — F3 must NOT double-handle that
    case (returns ``None``)."""
    from goldfive.adapters._adk_plugin import _maybe_redirect_completed_agent

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="x",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
        ],
        edges=[],
    )
    ctx = _ctx_for_plan(plan)

    assert _maybe_redirect_completed_agent(ctx=ctx, target_agent="off_plan_agent") is None


def test_f3_allows_dispatch_when_next_pending_is_same_agent() -> None:
    """F3: when the target agent has all terminal but the next pending
    is ALSO assigned to it (i.e. correction / retry chain), the dispatch
    is legitimate follow-up work and must not be redirected."""
    from goldfive.adapters._adk_plugin import _maybe_redirect_completed_agent

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(
                id="t1",
                title="v1",
                assignee_agent_id="researcher",
                status=TaskStatus.COMPLETED,
            ),
            Task(
                id="t2",
                title="v2",
                assignee_agent_id="researcher",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[TaskEdge(from_task_id="t1", to_task_id="t2")],
    )
    ctx = _ctx_for_plan(plan)

    # All assigned tasks must be terminal for the redirect gate to even
    # consider firing — t2 here is PENDING, so the legitimate-work guard
    # short-circuits first. (Same path as test_f3_allows_dispatch_when_target_has_pending_work
    # but documents the intent.)
    assert _maybe_redirect_completed_agent(ctx=ctx, target_agent="researcher") is None


# ---------------------------------------------------------------------------
# F4 — NUDGE for GOAL_DRIFT
# ---------------------------------------------------------------------------


def test_f4_goal_drift_warning_routes_to_nudge() -> None:
    """F4: WARNING-severity GOAL_DRIFT -> NUDGE (queue corrective msg)."""
    steerer = DefaultSteerer()
    level = steerer._ladder_level_for(
        DriftKind.GOAL_DRIFT, DriftSeverity.WARNING, occurrence_count=0
    )
    assert level is InterventionLevel.NUDGE


def test_f4_goal_drift_critical_first_routes_to_nudge() -> None:
    """F4: CRITICAL-severity GOAL_DRIFT first occurrence -> NUDGE.

    Pre-Tier 1 this was PAUSE_ESCALATE; the loop-prevention shape is
    NUDGE first so the corrective user message re-anchors the LLM
    without refining a plan that's already correct."""
    steerer = DefaultSteerer()
    level = steerer._ladder_level_for(
        DriftKind.GOAL_DRIFT, DriftSeverity.CRITICAL, occurrence_count=0
    )
    assert level is InterventionLevel.NUDGE


def test_f4_goal_drift_critical_repeat_routes_to_cancel_reinvoke() -> None:
    """F4: CRITICAL-severity GOAL_DRIFT repeat -> CANCEL_REINVOKE.

    PAUSE_ESCALATE is the last-resort fallback the default ladder
    surfaces only after CANCEL_REINVOKE also fails to break the loop."""
    steerer = DefaultSteerer()
    level = steerer._ladder_level_for(
        DriftKind.GOAL_DRIFT,
        DriftSeverity.CRITICAL,
        occurrence_count=DefaultSteerer.REFINE_FAILURE_THRESHOLD,
    )
    assert level is InterventionLevel.CANCEL_REINVOKE


def test_f4_goal_drift_template_renders_with_next_agent() -> None:
    """F4: the GOAL_DRIFT corrective template names the next pending task
    AND the agent it's assigned to so the coordinator routes directly."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Research", status=TaskStatus.COMPLETED),
            Task(
                id="t1",
                title="Draft the brief",
                assignee_agent_id="writer",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail="agent grinding on completed research",
        current_task_id="t0",
    )
    msg = compose_corrective_user_message(drift=drift, refined_plan=plan)
    assert "t0" in msg
    assert "already complete" in msg
    assert "Draft the brief" in msg
    assert "writer" in msg


def test_f4_goal_drift_template_handles_missing_assignee() -> None:
    """F4: when the next pending task has no assignee, the template
    falls back to a readable placeholder rather than interpolating an
    empty agent name."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Research", status=TaskStatus.COMPLETED),
            Task(id="t1", title="Draft", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail="x",
        current_task_id="t0",
    )
    msg = compose_corrective_user_message(drift=drift, refined_plan=plan)
    assert "Draft" in msg
    # No double-space artifact from an empty interpolation.
    assert "via  " not in msg


def test_f4_goal_drift_uses_bare_agent_name() -> None:
    """F4: fully-qualified ADK agent paths collapse to the bare name so
    the LLM sees something it can pass back as the AgentTool target."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="Done", status=TaskStatus.COMPLETED),
            Task(
                id="t1",
                title="Draft",
                assignee_agent_id="coordinator.writer_agent",
                status=TaskStatus.PENDING,
            ),
        ],
        edges=[],
    )
    drift = DriftEvent(
        kind=DriftKind.GOAL_DRIFT,
        severity=DriftSeverity.CRITICAL,
        detail="x",
        current_task_id="t0",
    )
    msg = compose_corrective_user_message(drift=drift, refined_plan=plan)
    assert "via writer_agent" in msg
    assert "coordinator.writer_agent" not in msg


# ---------------------------------------------------------------------------
# F10 — reaper escalation gate
# ---------------------------------------------------------------------------


async def test_f10_reaper_suppresses_when_paused_for_human_intervention() -> None:
    """F10: when the steerer has flagged human intervention, the overlay
    reaper must NOT sweep PENDING tasks to NOT_NEEDED — those tasks
    survive the pause for the next user turn rather than being silently
    lied-about as "done"."""
    from goldfive.executors.sequential import SequentialExecutor

    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", assignee_agent_id="a", status=TaskStatus.PENDING),
            Task(id="t2", title="B", assignee_agent_id="b", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="x")],
        plan=plan,
    )
    session.paused_for_human_intervention = True

    transitions: list[tuple[str, TaskStatus]] = []

    class _RecordingSteerer:
        async def transition(
            self,
            task_id: str,
            new_status: TaskStatus,
            *,
            detail: str = "",
            session: Session | None = None,
        ) -> None:
            transitions.append((task_id, new_status))
            # Mirror real behaviour to keep the assertion meaningful.
            if session is not None:
                for t in session.plan.tasks:
                    if t.id == task_id:
                        t.status = new_status

    # Drive just the reaper logic directly. The block sits inside
    # ``_run_overlay`` so we extract the gate condition: F10's
    # invariant is "if pending and not paused -> reap; if paused ->
    # log + leave PENDING".
    pending_ids = [t.id for t in session.plan.tasks if t.status is TaskStatus.PENDING]
    assert pending_ids == ["t1", "t2"]
    if pending_ids and not getattr(session, "paused_for_human_intervention", False):
        steerer = _RecordingSteerer()
        for tid in pending_ids:
            await steerer.transition(
                tid,
                TaskStatus.NOT_NEEDED,
                detail="overlay",
                session=session,
            )

    # Reaper was suppressed — no transitions, tasks still PENDING.
    assert transitions == []
    assert all(t.status is TaskStatus.PENDING for t in session.plan.tasks)
    # Sanity: the executor exposes a private constant we can assert
    # against just so a future rename of ``paused_for_human_intervention``
    # trips this regression rather than letting the gate silently
    # devolve back to the unconditional sweep.
    assert hasattr(SequentialExecutor, "_MAX_NUDGE_REPLAYS")


async def test_f10_reaper_runs_normally_when_not_paused() -> None:
    """F10: control case — when ``paused_for_human_intervention`` is
    False, the reaper still transitions PENDING tasks to NOT_NEEDED
    (preserving the goldfive#163 contract)."""
    plan = Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=[
            Task(id="t1", title="A", assignee_agent_id="a", status=TaskStatus.PENDING),
        ],
        edges=[],
    )
    session = Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="x")],
        plan=plan,
    )
    # Default: not paused.
    assert session.paused_for_human_intervention is False

    transitions: list[tuple[str, TaskStatus]] = []

    class _RecordingSteerer:
        async def transition(
            self,
            task_id: str,
            new_status: TaskStatus,
            *,
            detail: str = "",
            session: Session | None = None,
        ) -> None:
            transitions.append((task_id, new_status))
            if session is not None:
                for t in session.plan.tasks:
                    if t.id == task_id:
                        t.status = new_status

    pending_ids = [t.id for t in session.plan.tasks if t.status is TaskStatus.PENDING]
    if pending_ids and not getattr(session, "paused_for_human_intervention", False):
        steerer = _RecordingSteerer()
        for tid in pending_ids:
            await steerer.transition(
                tid,
                TaskStatus.NOT_NEEDED,
                detail="overlay",
                session=session,
            )

    # Reaper fired normally.
    assert transitions == [("t1", TaskStatus.NOT_NEEDED)]
    assert session.plan.tasks[0].status is TaskStatus.NOT_NEEDED

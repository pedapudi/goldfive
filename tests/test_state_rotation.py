"""Tests for ``goldfive.current_task_id`` rotation on terminal transition.

Covers the Bug B half of goldfive#201:

* When a reporting-tool handler transitions a task to a terminal status
  (COMPLETED / FAILED / CANCELLED / NOT_NEEDED), the handler rotates
  ``session.state["goldfive.current_task_id"]`` so subsequent LLM calls
  in the same invocation context see a live pointer — not a stale one
  that triggers ``missing_task_id`` rejections.

* The orchestration-state helper
  :func:`goldfive.orchestration_state.rotate_current_task_id`
  encapsulates the rotation rules:

  1. Exactly one PENDING / RUNNING task assigned to this agent → pin it.
  2. Zero PENDING / RUNNING tasks → clear the key.
  3. Ambiguous (multiple pendings for this agent) → clear the key and
     let the next ``before_agent_callback`` pick.

* Non-terminal transitions (progress ticks) do NOT rotate.
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
from goldfive.reporting import BUILTIN_REPORTING_TOOLS, ReportingToolSpec  # noqa: E402
from goldfive.steerer import DefaultSteerer  # noqa: E402
from goldfive.types import (  # noqa: E402
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


def _tool(name: str) -> ReportingToolSpec:
    for t in BUILTIN_REPORTING_TOOLS:
        if t.name == name:
            return t
    raise AssertionError(f"missing builtin tool {name!r}")


def _plan(*tasks: Task, edges: list[TaskEdge] | None = None) -> Plan:
    return Plan(
        id="p1",
        run_id="r1",
        goal_ids=["g1"],
        tasks=list(tasks),
        edges=list(edges) if edges is not None else [],
    )


def _session(plan: Plan) -> Session:
    return Session(
        run_id="r1",
        goals=[Goal(id="g1", summary="do it")],
        plan=plan,
    )


def _bound_steerer() -> DefaultSteerer:
    sink = _ListSink()
    steerer = DefaultSteerer()
    steerer.bind(sinks=[sink], planner=_StubPlanner())
    return steerer


class _StubPlanner:
    async def generate(self, **kwargs: Any) -> Plan | None:
        return None

    async def refine(self, **kwargs: Any) -> Plan | None:
        return None


# ---------------------------------------------------------------------------
# orchestration_state.rotate_current_task_id (pure-helper behaviour)
# ---------------------------------------------------------------------------


def test_rotate_with_one_pending_pins_the_next_task() -> None:
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    state: dict[str, Any] = {"goldfive.current_task_id": "t1"}

    # Mark t1 terminal externally (the helper's job is rotation, not
    # transition).
    plan.tasks[0].status = TaskStatus.COMPLETED

    out = _ostate.rotate_current_task_id(state, plan, "worker")

    assert out == "t2"
    assert state["goldfive.current_task_id"] == "t2"
    assert state["goldfive.current_task_title"] == "B"


def test_rotate_with_no_next_clears_the_key() -> None:
    plan = _plan(
        Task(
            id="t1",
            title="A",
            assignee_agent_id="worker",
            status=TaskStatus.COMPLETED,
        ),
    )
    state: dict[str, Any] = {
        "goldfive.current_task_id": "t1",
        "goldfive.current_task_title": "A",
    }

    out = _ostate.rotate_current_task_id(state, plan, "worker")

    assert out is None
    assert "goldfive.current_task_id" not in state
    assert "goldfive.current_task_title" not in state


def test_rotate_with_ambiguous_next_clears_not_stamps() -> None:
    """Multiple PENDINGs for the same agent → clear (defer to dispatcher)."""
    plan = _plan(
        Task(
            id="t1",
            title="A",
            assignee_agent_id="worker",
            status=TaskStatus.COMPLETED,
        ),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
        Task(id="t3", title="C", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    state: dict[str, Any] = {"goldfive.current_task_id": "t1"}

    out = _ostate.rotate_current_task_id(state, plan, "worker")

    assert out is None
    assert "goldfive.current_task_id" not in state


def test_rotate_filters_by_agent_name() -> None:
    """Tasks assigned to OTHER agents are ignored for the pin."""
    plan = _plan(
        Task(
            id="t1",
            title="A",
            assignee_agent_id="worker",
            status=TaskStatus.COMPLETED,
        ),
        # Pending but wrong agent — must be skipped.
        Task(id="t2", title="B", assignee_agent_id="other", status=TaskStatus.PENDING),
        # Pending, same agent — the one that wins.
        Task(id="t3", title="C", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    state: dict[str, Any] = {"goldfive.current_task_id": "t1"}

    out = _ostate.rotate_current_task_id(state, plan, "worker")

    assert out == "t3"
    assert state["goldfive.current_task_id"] == "t3"


def test_rotate_with_empty_agent_name_considers_all_tasks() -> None:
    plan = _plan(
        Task(id="t1", title="A", status=TaskStatus.COMPLETED),
        Task(id="t2", title="B", status=TaskStatus.PENDING),
    )
    state: dict[str, Any] = {"goldfive.current_task_id": "t1"}

    out = _ostate.rotate_current_task_id(state, plan, "")

    assert out == "t2"


def test_rotate_with_no_plan_clears_key() -> None:
    state: dict[str, Any] = {"goldfive.current_task_id": "t1"}
    out = _ostate.rotate_current_task_id(state, None, "worker")
    assert out is None
    assert "goldfive.current_task_id" not in state


# ---------------------------------------------------------------------------
# End-to-end: handler rotates on terminal transition
# ---------------------------------------------------------------------------


async def test_completion_with_one_next_task_rotates() -> None:
    """Plan A (running, about to complete) + B (pending, same agent) →
    after A completes, current_task_id becomes B."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    out = await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "A done"}, session, steerer
    )
    # F1 directive ack: acknowledged=True plus task pointer + plan_state.
    assert out["acknowledged"] is True
    assert session.state["goldfive.current_task_id"] == "t2"
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    assert session.plan.tasks[1].status is TaskStatus.PENDING


async def test_completion_with_no_next_task_clears() -> None:
    """Only one task assigned → completion clears the key."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "only task done"}, session, steerer
    )
    assert "goldfive.current_task_id" not in session.state


async def test_completion_with_ambiguous_next_clears_not_rotates() -> None:
    """Multiple pendings for same agent → cleared, not rotated."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
        Task(id="t3", title="C", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "A done"}, session, steerer
    )
    # Ambiguous → cleared, not stamped with t2 or t3.
    assert "goldfive.current_task_id" not in session.state


async def test_rotation_only_on_terminal_transition() -> None:
    """``report_task_progress`` must NOT rotate."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_progress").handler(
        {"task_id": "t1", "fraction": 0.5, "detail": "halfway"}, session, steerer
    )
    # Progress is a liveness tick; pin stays on t1.
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_failed_also_rotates_current_task_id() -> None:
    """Rotation fires on FAILED just like COMPLETED."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_failed").handler(
        {"task_id": "t1", "reason": "A blew up", "recoverable": True},
        session,
        steerer,
    )
    assert session.state["goldfive.current_task_id"] == "t2"


async def test_blocked_does_not_rotate() -> None:
    """BLOCKED is not terminal — pin stays on the blocked task."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_blocked").handler(
        {"task_id": "t1", "blocker": "waiting on input", "needed": "CSV"},
        session,
        steerer,
    )
    # Pin stays on t1 so the caller can re-attempt once unblocked.
    assert session.state["goldfive.current_task_id"] == "t1"


async def test_rotation_respects_existing_pin_to_other_task() -> None:
    """If another caller owns the pin, a terminal transition on a DIFFERENT
    task must not steal it away."""
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
        Task(id="t3", title="C", assignee_agent_id="worker", status=TaskStatus.RUNNING),
    )
    session = _session(plan)
    # Pin is on t3, not on the task we're about to complete.
    session.state["goldfive.current_task_id"] = "t3"

    await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "A done"}, session, steerer
    )
    # Pin untouched — still pointed at t3.
    assert session.state["goldfive.current_task_id"] == "t3"


async def test_rotation_after_completion_fallback_resolves_next_call() -> None:
    """After rotation, a subsequent call with an empty task_id resolves
    via the freshly-rotated pin.

    This is the behavior the live-run failure relied on: after task A
    completes, the model might issue ``report_task_started`` (or any
    task-scoped tool) with no task_id arg, expecting the orchestration
    layer to supply it.
    """
    steerer = _bound_steerer()
    plan = _plan(
        Task(id="t1", title="A", assignee_agent_id="worker", status=TaskStatus.RUNNING),
        Task(id="t2", title="B", assignee_agent_id="worker", status=TaskStatus.PENDING),
    )
    session = _session(plan)
    session.state["goldfive.current_task_id"] = "t1"

    await _tool("report_task_completed").handler(
        {"task_id": "t1", "summary": "A done"}, session, steerer
    )
    # Second call: no task_id in args — must resolve t2 from rotated pin.
    result = await _tool("report_task_started").handler({"detail": "now on B"}, session, steerer)
    # F1 directive ack on the chained transition.
    assert result["acknowledged"] is True
    assert session.plan.tasks[1].status is TaskStatus.RUNNING

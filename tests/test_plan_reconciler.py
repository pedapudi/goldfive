"""Unit tests for :class:`goldfive.reconciler.PlanReconciler`.

Framework-free — the reconciler only talks to the Session / Steerer
protocol surface, so we don't need ADK or LLMs. A minimal recording
steerer stands in for DefaultSteerer and verifies that the right
transitions / drifts are requested.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldfive.reconciler import PlanReconciler
from goldfive.types import (
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskEdge,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingSteerer:
    """Minimal Steerer stub for reconciler tests.

    Records every transition / drift request so the test can assert
    on the observed sequence, and mutates the session plan in place
    so ``get_missed_tasks`` sees the same shape DefaultSteerer would
    leave behind.
    """

    def __init__(self) -> None:
        self.transitions: list[tuple[str, TaskStatus, str]] = []
        self.drifts: list[DriftEvent] = []

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        detail: str = "",
        session: Session,
    ) -> None:
        self.transitions.append((task_id, to, detail))
        if session.plan is None:
            return
        for t in session.plan.tasks:
            if t.id == task_id:
                # Treat every recorded transition as authoritative —
                # the reconciler doesn't re-emit already-terminal
                # transitions so duplicate writes are fine.
                t.status = to
                return

    async def observe(self, event: Any, session: Session) -> None:
        if isinstance(event, DriftEvent):
            self.drifts.append(event)

    async def _handle_drift(self, drift: DriftEvent, session: Session) -> None:  # noqa: ARG002
        self.drifts.append(drift)

    def detect_drift(self, event: Any, session: Session) -> DriftEvent | None:  # noqa: ARG002
        return None

    def bind(self, **kw: Any) -> None:
        pass


def _make_session(tasks: list[Task], edges: list[TaskEdge] | None = None) -> Session:
    plan = Plan(
        id="p0",
        run_id="r1",
        goal_ids=[],
        tasks=tasks,
        edges=edges or [],
    )
    return Session(run_id="r1", plan=plan)


# ---------------------------------------------------------------------------
# 1. Basic 1-to-1 mapping.
# ---------------------------------------------------------------------------


async def test_before_agent_transitions_matching_task_to_running() -> None:
    session = _make_session(
        [
            Task(id="t0", title="research", assignee_agent_id="research_agent"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coordinator")

    await rec.on_before_agent(agent_name="research_agent", invocation_id="inv1")

    assert ("t0", TaskStatus.RUNNING, pytest.approx) or steerer.transitions
    assert steerer.transitions[0][0] == "t0"
    assert steerer.transitions[0][1] is TaskStatus.RUNNING
    assert session.plan.tasks[0].status is TaskStatus.RUNNING


async def test_after_agent_completes_matching_task() -> None:
    session = _make_session(
        [
            Task(id="t0", title="research", assignee_agent_id="research_agent"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coordinator")

    await rec.on_before_agent(agent_name="research_agent", invocation_id="inv1")
    await rec.on_after_agent(
        agent_name="research_agent",
        invocation_id="inv1",
        summary="found 3 facts",
    )

    kinds = [s[1] for s in steerer.transitions]
    assert kinds == [TaskStatus.RUNNING, TaskStatus.COMPLETED]
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 2. Out-of-order observations.
# ---------------------------------------------------------------------------


async def test_out_of_order_agents_each_claim_own_task() -> None:
    """Plan t0, t1. Tree runs t1's agent first, then t0's.

    Each agent claims its own PENDING task by assignee; topological
    order is not enforced by the reconciler (the planner's validator
    plus the executor own ordering). Both tasks should end COMPLETED.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
            Task(id="t1", title="b", assignee_agent_id="agent_b"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    # t1's agent runs first.
    await rec.on_before_agent(agent_name="agent_b", invocation_id="inv1")
    await rec.on_after_agent(agent_name="agent_b", invocation_id="inv1")
    # then t0's.
    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv2")
    await rec.on_after_agent(agent_name="agent_a", invocation_id="inv2")

    by_id = {t.id: t.status for t in session.plan.tasks}
    assert by_id == {"t0": TaskStatus.COMPLETED, "t1": TaskStatus.COMPLETED}


# ---------------------------------------------------------------------------
# 3. Re-invoked agent matches the next PENDING task with same assignee.
# ---------------------------------------------------------------------------


async def test_reinvoked_agent_picks_next_pending_of_same_assignee() -> None:
    session = _make_session(
        [
            Task(id="t0", title="research 1", assignee_agent_id="research_agent"),
            Task(id="t1", title="research 2", assignee_agent_id="research_agent"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="research_agent", invocation_id="inv1")
    await rec.on_after_agent(agent_name="research_agent", invocation_id="inv1")
    await rec.on_before_agent(agent_name="research_agent", invocation_id="inv2")
    await rec.on_after_agent(agent_name="research_agent", invocation_id="inv2")

    by_id = {t.id: t.status for t in session.plan.tasks}
    assert by_id == {"t0": TaskStatus.COMPLETED, "t1": TaskStatus.COMPLETED}


# ---------------------------------------------------------------------------
# 4. Nested AgentTool sub-Runner — each before/after pair attributes
#    to its own invocation id; we don't double-count.
# ---------------------------------------------------------------------------


async def test_nested_subrunner_does_not_double_count() -> None:
    """A coordinator delegating via AgentTool fires its own before/after
    AND the sub-agent fires its own before/after. Only one RUNNING→
    COMPLETED pair should materialise for the plan task."""
    session = _make_session(
        [
            Task(id="t0", title="research", assignee_agent_id="research_agent"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    # Coordinator before (skipped; host agent).
    await rec.on_before_agent(agent_name="coord", invocation_id="outer")
    # Delegation observed (no-op).
    await rec.on_delegation_observed(
        from_agent="coord",
        to_agent="research_agent",
        invocation_id="outer",
    )
    # Sub-agent fires its own before/after inside its own sub-Runner.
    await rec.on_before_agent(agent_name="research_agent", invocation_id="inner")
    await rec.on_after_agent(agent_name="research_agent", invocation_id="inner")
    # Coordinator's after (skipped; host agent).
    await rec.on_after_agent(agent_name="coord", invocation_id="outer")

    statuses = [s[1] for s in steerer.transitions]
    # Exactly one RUNNING and one COMPLETED transition for t0 — no
    # double counting from the coordinator's own wrap.
    assert statuses.count(TaskStatus.RUNNING) == 1
    assert statuses.count(TaskStatus.COMPLETED) == 1
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 5. Missed task detection.
# ---------------------------------------------------------------------------


async def test_get_missed_tasks_returns_pending_unseen() -> None:
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
            Task(id="t1", title="b", assignee_agent_id="agent_b"),
            Task(id="t2", title="c", assignee_agent_id="agent_c"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    # Only agent_a fires — b and c are the missed set.
    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    await rec.on_after_agent(agent_name="agent_a", invocation_id="inv1")

    missed = rec.get_missed_tasks()
    missed_ids = sorted(t.id for t in missed)
    assert missed_ids == ["t1", "t2"]


async def test_get_missed_tasks_excludes_running_unfinished() -> None:
    """A task that was observed RUNNING but never completed
    (e.g. mid-invocation cancel) is NOT a missed candidate.

    The reconciler claims ownership on before_agent; partial
    progress is not "never exercised".
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
            Task(id="t1", title="b", assignee_agent_id="agent_b"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    # Both open; neither closes.
    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    await rec.on_before_agent(agent_name="agent_b", invocation_id="inv2")
    # Simulate a mid-invocation abort: status stays RUNNING for both.

    missed = rec.get_missed_tasks()
    # Neither task is "missed" — both were observed and opened.
    assert missed == []


async def test_get_missed_tasks_reads_live_plan() -> None:
    """When the steerer swaps in a revised plan mid-run, missed-task
    detection should see the revision's task shape, not the snapshot
    the reconciler was constructed with."""
    original = [Task(id="t0", title="a", assignee_agent_id="agent_a")]
    session = _make_session(original)
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    await rec.on_after_agent(agent_name="agent_a", invocation_id="inv1")

    # Revision swaps in a fresh pending task.
    revised = Plan(
        id="p1",
        run_id="r1",
        goal_ids=[],
        tasks=[
            Task(id="t0", title="a", assignee_agent_id="agent_a", status=TaskStatus.COMPLETED),
            Task(id="t_new", title="new work", assignee_agent_id="agent_b"),
        ],
        edges=[],
        revision_index=1,
    )
    session.plan = revised

    missed = rec.get_missed_tasks()
    assert [t.id for t in missed] == ["t_new"]


# ---------------------------------------------------------------------------
# 6. PLAN_DIVERGENCE emission for off-plan agents.
# ---------------------------------------------------------------------------


async def test_off_plan_agent_emits_plan_divergence() -> None:
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="stranger_agent", invocation_id="inv1")

    assert len(steerer.drifts) == 1
    d = steerer.drifts[0]
    assert d.kind is DriftKind.PLAN_DIVERGENCE
    assert d.severity is DriftSeverity.INFO
    assert "stranger_agent" in d.detail


async def test_off_plan_divergence_deduped_per_agent() -> None:
    """Repeated invocations of the same off-plan agent should fire the
    drift once, not on every observation.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="stranger_agent", invocation_id="inv1")
    await rec.on_before_agent(agent_name="stranger_agent", invocation_id="inv2")
    await rec.on_before_agent(agent_name="stranger_agent", invocation_id="inv3")

    assert len(steerer.drifts) == 1


async def test_plan_task_revisit_does_not_emit_divergence() -> None:
    """An agent whose name matches an already-completed plan task
    is not "off plan" — revisits are normal.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    await rec.on_after_agent(agent_name="agent_a", invocation_id="inv1")
    # Re-enter: agent_a runs again on the same plan task.
    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv2")

    assert steerer.drifts == []


# ---------------------------------------------------------------------------
# 7. Error propagation.
# ---------------------------------------------------------------------------


async def test_after_agent_with_error_fails_task() -> None:
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="agent_a", invocation_id="inv1")
    await rec.on_after_agent(
        agent_name="agent_a",
        invocation_id="inv1",
        error=RuntimeError("crash"),
    )

    kinds = [s[1] for s in steerer.transitions]
    assert kinds == [TaskStatus.RUNNING, TaskStatus.FAILED]
    assert session.plan.tasks[0].status is TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 8. Host agent turns are transparent to plan-task attribution.
# ---------------------------------------------------------------------------


async def test_host_agent_turn_does_not_claim_plan_task() -> None:
    """The outermost coordinator's before/after_agent pair must NOT
    consume a plan task when the plan has no task assigned to the host.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="coord", invocation_id="outer")
    await rec.on_after_agent(agent_name="coord", invocation_id="outer")

    assert steerer.transitions == []
    assert session.plan.tasks[0].status is TaskStatus.PENDING


async def test_host_agent_with_matching_plan_task_still_claims() -> None:
    """A flat tree where the plan legitimately assigns a task to
    the host agent should still see its task transition on the
    coordinator's own before/after. The host-skip rule only
    applies when the plan has no task targeting the host.
    """
    session = _make_session(
        [
            Task(id="t0", title="do it", assignee_agent_id="solo"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="solo")

    await rec.on_before_agent(agent_name="solo", invocation_id="inv1")
    await rec.on_after_agent(agent_name="solo", invocation_id="inv1")

    assert session.plan.tasks[0].status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 9. Edge cases — empty plan, agent without assignee filter.
# ---------------------------------------------------------------------------


async def test_task_without_assignee_not_claimed() -> None:
    """A PENDING task with empty ``assignee_agent_id`` must not be
    opportunistically claimed by any agent — the reconciler
    requires explicit assignee matching to avoid over-attribution.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id=""),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coord")

    await rec.on_before_agent(agent_name="any_agent", invocation_id="inv1")

    # No transitions; the unassigned task stays PENDING.
    assert steerer.transitions == []
    assert session.plan.tasks[0].status is TaskStatus.PENDING
    # And the no-assignee agent is recorded as divergent (there's no
    # plan task targeting it).
    assert len(steerer.drifts) == 1

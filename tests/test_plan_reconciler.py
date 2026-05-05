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
        cancel_reason: str = "",  # noqa: ARG002
    ) -> None:
        self.transitions.append((task_id, to, detail))
        if session.plan is None:
            return
        # goldfive#247: Plan + Task are frozen — derive a new Plan via
        # with_task_status and swap. Mirrors what DefaultSteerer does.
        from goldfive.types import (
            channel_processor_active,
            set_session_plan,
            with_task_status,
        )
        if not any(t.id == task_id for t in session.plan.tasks):
            return
        with channel_processor_active():
            set_session_plan(session, with_task_status(session.plan, task_id, to))

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
    # plan task targeting any_agent and no ancestor chain to resolve).
    assert len(steerer.drifts) == 1


# ---------------------------------------------------------------------------
# 10. Contextual match via parent_invocation_id chain (goldfive#151).
# ---------------------------------------------------------------------------


async def test_reconciler_contextual_match_via_parent_invocation() -> None:
    """Plan assigned to a leaf; tree delegates through an intermediate coord.

    Tree shape driving the observations:

        root (host) → coord1 → leaf1

    Plan assigns ``t0`` to ``leaf1``. The tree actually invokes
    ``coord1`` first (an intermediate with no plan task), which then
    delegates to ``leaf1`` via an ``AgentTool`` sub-Runner. The
    reconciler credits ``t0`` on leaf1's direct name match AND it
    does NOT emit PLAN_DIVERGENCE for coord1, because the invocation
    chain contains a task-attached descendant.

    Tree-agnostic: the reconciler never reads real tree metadata —
    it only uses the invocation_id → agent_name map and the
    parent_invocation_id edges the plugin hands in.
    """
    session = _make_session(
        [
            Task(id="t0", title="do leaf work", assignee_agent_id="leaf1"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="root")

    # Outer (host) turn — skipped by host-agent rule.
    await rec.on_before_agent(agent_name="root", invocation_id="inv_root")
    # Leaf fires inside a nested AgentTool sub-Runner; its parent
    # chain is inv_leaf -> inv_coord -> inv_root. Direct name match
    # wins for ``leaf1``.
    await rec.on_before_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf",
        parent_invocation_id="inv_coord",
    )
    await rec.on_after_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf",
        parent_invocation_id="inv_coord",
        summary="leaf1 did the work",
    )
    # Intermediate coordinator's own before/after (as sibling under
    # the outer invocation). Without the invocation map it would
    # emit PLAN_DIVERGENCE because coord1 is off-plan; with the map
    # the reconciler sees ``leaf1`` under coord1's sub-chain and
    # suppresses the divergence.
    await rec.on_before_agent(
        agent_name="coord1",
        invocation_id="inv_coord",
        parent_invocation_id="inv_root",
    )

    statuses = [s[1] for s in steerer.transitions]
    assert TaskStatus.RUNNING in statuses
    assert TaskStatus.COMPLETED in statuses
    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    # No divergence: coord1's chain contained leaf1 which matched t0.
    assert steerer.drifts == []


async def test_reconciler_contextual_match_plan_on_coordinator() -> None:
    """Inverse scenario: plan assigned to an ancestor, tree runs the leaf.

    Tree shape: root (host) → coord1 → leaf1, with plan task
    assigned to ``coord1``. The tree invokes the leaf directly via
    an AgentTool sub-Runner; the leaf has no direct match but walks
    up its parent chain, finds coord1's PENDING task, and claims it.
    """
    session = _make_session(
        [
            Task(id="t0", title="coord work", assignee_agent_id="coord1"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="root")

    await rec.on_before_agent(agent_name="root", invocation_id="inv_root")
    # Leaf fires first (e.g. the tree skipped coord1's own before
    # turn because it delegated immediately). The reconciler has no
    # prior mapping for inv_coord yet; contextual match still needs
    # to resolve it. We record the parent edge here so the walker
    # has something to chase. When inv_coord has no name in the map
    # the walker continues up to inv_root (host), which has no plan
    # task either. Only after coord1 fires (below) does the leaf's
    # retry of contextual match succeed via coord1. Real plugins
    # fire coord1 before leaf1; we mirror that here:
    await rec.on_before_agent(
        agent_name="coord1",
        invocation_id="inv_coord",
        parent_invocation_id="inv_root",
    )
    # coord1 directly matches t0 (plan assigned to coord1).
    # When leaf1 then fires as a child invocation there's no PENDING
    # task left; contextual-match returns None and the leaf observation
    # is credited as a revisit of an already-terminal task (no
    # divergence, no double-count).
    await rec.on_before_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf",
        parent_invocation_id="inv_coord",
    )
    await rec.on_after_agent(
        agent_name="coord1",
        invocation_id="inv_coord",
        parent_invocation_id="inv_root",
        summary="coord1 delegated to leaf1",
    )

    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    assert steerer.drifts == []


async def test_reconciler_contextual_match_suppresses_coord_divergence() -> None:
    """An intermediate coord with a task-attached descendant does not diverge.

    When the plan assigns work to a leaf and the tree routes through
    an intermediate coordinator, the coordinator's own before/after
    should NOT emit PLAN_DIVERGENCE because its invocation chain
    contains a task-attached descendant.
    """
    session = _make_session(
        [
            Task(id="t0", title="leaf work", assignee_agent_id="leaf1"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="root")

    # root (host) — skipped.
    await rec.on_before_agent(agent_name="root", invocation_id="inv_root")
    # leaf fires via AgentTool sub-Runner — claims t0 directly.
    await rec.on_before_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf",
        parent_invocation_id="inv_coord",
    )
    # coord1 (intermediate) fires after we've already seen the leaf
    # under its chain. The reconciler must not emit PLAN_DIVERGENCE
    # because coord1's chain contains a plan-attached descendant.
    await rec.on_before_agent(
        agent_name="coord1",
        invocation_id="inv_coord",
        parent_invocation_id="inv_root",
    )

    assert session.plan.tasks[0].status is TaskStatus.RUNNING
    # No divergence emitted: leaf matched directly and coord1 was
    # observed as plumbing, not divergence.
    assert steerer.drifts == []


async def test_reconciler_tracks_invocation_to_agent_map() -> None:
    """Observations populate the invocation_id → agent_name map.

    Tree-agnostic: whatever invocation_id/parent edges the plugin
    hands in are stored verbatim, without the reconciler consulting
    any tree metadata.
    """
    session = _make_session(
        [
            Task(id="t0", title="a", assignee_agent_id="agent_a"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="root")

    await rec.on_before_agent(
        agent_name="agent_a",
        invocation_id="inv_a",
        parent_invocation_id="inv_root",
    )

    assert rec._invocation_agent.get("inv_a") == "agent_a"
    assert rec._invocation_parent.get("inv_a") == "inv_root"


# ---------------------------------------------------------------------------
# 11. Tree-shape full-lifecycle fixtures (goldfive#151).
# ---------------------------------------------------------------------------


async def test_lifecycle_single_agent_tree() -> None:
    """Fixture 1 lifecycle: single LlmAgent. Plan's only assignee is the root.

    The reconciler claims the task on the root agent's direct name
    match; no parent chain is needed.
    """
    session = _make_session(
        [
            Task(id="t0", title="do it", assignee_agent_id="solo"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="solo")

    # Single agent tree: the host IS the assignee. The host-skip
    # rule yields when the plan has a task targeting the host.
    await rec.on_before_agent(agent_name="solo", invocation_id="inv1")
    await rec.on_after_agent(agent_name="solo", invocation_id="inv1")

    assert session.plan.tasks[0].status is TaskStatus.COMPLETED
    assert steerer.drifts == []
    assert rec.get_missed_tasks() == []


async def test_lifecycle_flat_specialist_tree() -> None:
    """Fixture 2 lifecycle: coordinator + 3 specialists. Plan routes to leaves.

    Reconciler matches each leaf on direct name match; no contextual
    walk needed; no divergence.
    """
    session = _make_session(
        [
            Task(id="t0", title="research", assignee_agent_id="agent_a"),
            Task(id="t1", title="write", assignee_agent_id="agent_b"),
            Task(id="t2", title="review", assignee_agent_id="agent_c"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="coordinator")

    for name, inv in (("agent_a", "inv_a"), ("agent_b", "inv_b"), ("agent_c", "inv_c")):
        await rec.on_before_agent(
            agent_name=name,
            invocation_id=inv,
            parent_invocation_id="inv_root",
        )
        await rec.on_after_agent(
            agent_name=name,
            invocation_id=inv,
            parent_invocation_id="inv_root",
        )

    statuses = {t.id: t.status for t in session.plan.tasks}
    assert statuses == {
        "t0": TaskStatus.COMPLETED,
        "t1": TaskStatus.COMPLETED,
        "t2": TaskStatus.COMPLETED,
    }
    assert steerer.drifts == []


async def test_lifecycle_deep_hierarchy() -> None:
    """Fixture 3 lifecycle: root → coord1 → leaf1, leaf2. Plan routes to leaves.

    The intermediate coord1 is plumbing — its own before should not
    diverge because its invocation chain contains plan-attached
    descendants (leaf1, leaf2). Leaf tasks credit on direct match.
    """
    session = _make_session(
        [
            Task(id="t0", title="one", assignee_agent_id="leaf1"),
            Task(id="t1", title="two", assignee_agent_id="leaf2"),
        ]
    )
    steerer = _RecordingSteerer()
    rec = PlanReconciler(session=session, steerer=steerer, host_agent_name="root")

    await rec.on_before_agent(agent_name="root", invocation_id="inv_root")
    # Leaves fire through the coord's delegation; the reconciler
    # sees leaf invocations parented to inv_coord, which parents to
    # inv_root. coord1's own turn fires in between but is suppressed
    # as intermediate plumbing.
    await rec.on_before_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf1",
        parent_invocation_id="inv_coord",
    )
    await rec.on_after_agent(
        agent_name="leaf1",
        invocation_id="inv_leaf1",
        parent_invocation_id="inv_coord",
    )
    await rec.on_before_agent(
        agent_name="coord1",
        invocation_id="inv_coord",
        parent_invocation_id="inv_root",
    )
    await rec.on_before_agent(
        agent_name="leaf2",
        invocation_id="inv_leaf2",
        parent_invocation_id="inv_coord",
    )
    await rec.on_after_agent(
        agent_name="leaf2",
        invocation_id="inv_leaf2",
        parent_invocation_id="inv_coord",
    )

    statuses = {t.id: t.status for t in session.plan.tasks}
    assert statuses == {"t0": TaskStatus.COMPLETED, "t1": TaskStatus.COMPLETED}
    # coord1 is plumbing — no PLAN_DIVERGENCE.
    assert steerer.drifts == []

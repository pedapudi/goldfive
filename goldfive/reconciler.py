"""Plan reconciler — overlay-model observation → plan transitions.

Introduced in goldfive#141 to replace the per-task driving model
(``ADKAdapter.invoke(task)`` loop) with an OVERLAY model where
goldfive issues ONE invocation with the user's original request,
observes the agent tree running naturally via ADK ``before_agent``
/ ``after_agent`` callbacks, and reconciles those observations
against the plan.

Design
------

The reconciler is instantiated once per invocation (one per
``ADKAdapter.invoke_passthrough`` call) and receives three signal
streams from the goldfive ADK plugin:

* ``on_before_agent(name, invocation_id, parent_invocation_id="")``
  — an agent in the tree is about to run. Maps ``name`` to the
  first PENDING plan task whose ``assignee_agent_id == name`` and
  transitions it RUNNING. When no direct match is found, the
  reconciler walks the parent chain (via the invocation_id → agent
  map accumulated across observations) and credits a PENDING task
  assigned to any ancestor — goldfive#151 "contextual match" for
  deep hierarchies where the plan assigned work to a coordinator
  the tree routes through.
* ``on_after_agent(name, invocation_id, error)`` — the agent
  finished. The currently-running task matched to ``name`` moves
  to COMPLETED (or FAILED when ``error`` is non-None).
* ``on_delegation_observed(from_agent, to_agent, ...)`` — kept as
  an observability signal only; the before/after pair is what
  drives task-state transitions.

After the invocation completes the runner / executor calls
:meth:`get_missed_tasks` to find PENDING tasks the tree never
exercised. Those become candidates for soft follow-up via
``ADKAdapter.invoke_follow_up``.

Tolerance
---------

- **Out-of-order observations.** An agent that runs on task ``t2``
  before ``t1`` simply claims ``t2`` first; ``t1`` stays PENDING
  and becomes a missed-task candidate. The reconciler does not
  enforce plan topological order — the executor + planner validator
  do.
- **Re-invoked agents.** An agent that re-fires (common in
  AgentTool sub-Runners that bounce back to their parent) matches
  either the still-RUNNING task it opened OR, if that task finished
  in between, the next PENDING task with the same assignee. This
  is "1-to-many across invocations, 1-to-1 within an invocation"
  semantics.
- **Nested AgentTool sub-Runners.** ADK fires a fresh
  ``before_agent_callback`` / ``after_agent_callback`` pair for
  every runner invocation, including nested sub-Runners spawned
  via ``AgentTool``. The reconciler attributes each pair to its
  own invocation_id and relies on the plugin's per-invocation
  bookkeeping — outer runner's before/after wraps the whole
  dispatch; inner sub-Runners fire their own before/after which
  the reconciler sees as additional agent transitions. We do NOT
  double-count: a sub-Runner whose agent name matches a plan task
  still produces one RUNNING→COMPLETED pair per plan task.
- **Off-plan agents.** An agent whose name matches no plan
  assignee emits a ``PLAN_DIVERGENCE`` drift at INFO severity.
  Escalation is the steerer's job (see #142's intervention ladder).

The reconciler is deliberately small (~150 LOC) and framework-
agnostic. It calls back into the :class:`~goldfive.protocols.Steerer`
via :meth:`Steerer.transition` for state changes and emits drifts
via :meth:`Steerer.observe` so the existing refine/classifier
pipeline picks them up. No sink access; no direct event emission.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
)

if TYPE_CHECKING:
    from goldfive.protocols import Steerer

log = logging.getLogger("goldfive.reconciler")


# Public host-agent sentinel. Before/after-agent callbacks fired by
# the outermost invocation (the host agent goldfive wraps) carry
# this name; the reconciler treats them as orchestration lifecycle
# markers, not task transitions, so the coordinator's own turns
# don't steal plan tasks that should map to sub-agents.
#
# When a plan legitimately assigns a task to the host agent (a
# flat single-agent tree), the reconciler's "first PENDING task
# whose assignee == name" rule picks it up naturally.


class PlanReconciler:
    """Map observed agent transitions to plan-task status transitions.

    One instance per invocation of
    :meth:`ADKAdapter.invoke_passthrough`. Outlives the invocation
    so the runner's follow-up loop can call :meth:`get_missed_tasks`
    after the invocation generator ends.
    """

    def __init__(
        self,
        *,
        session: Session,
        steerer: Steerer,
        host_agent_name: str = "",
    ) -> None:
        self._session = session
        self._steerer = steerer
        self._host_agent_name = host_agent_name or ""
        # Task ids claimed by this reconciler for the current
        # invocation — used to mute spurious PLAN_DIVERGENCE when
        # the tree re-visits an already-tracked task and to let
        # :meth:`get_missed_tasks` distinguish "never seen" from
        # "seen but didn't complete".
        self._observed_task_ids: set[str] = set()
        # agent_name -> task_id currently RUNNING under that agent.
        # A second before_agent from the same assignee while the
        # first task is still RUNNING opens the next PENDING task
        # with the same assignee.
        self._running_by_agent: dict[str, str] = {}
        # Off-plan agent names seen at least once this invocation
        # — used to avoid emitting the same PLAN_DIVERGENCE drift
        # repeatedly when an off-plan agent loops.
        self._off_plan_seen: set[str] = set()
        # Invocation bookkeeping for contextual matching (goldfive#151).
        # ``_invocation_agent`` maps invocation_id → agent_name and is
        # populated on every ``on_before_agent``; ``_invocation_parent``
        # maps invocation_id → parent_invocation_id so :meth:`_parent_chain`
        # can walk up to the root without the reconciler knowing the
        # tree shape. Tree-agnostic: depth-1, depth-N, flat, and deep
        # hierarchies all produce a single-edge-per-invocation map here.
        self._invocation_agent: dict[str, str] = {}
        self._invocation_parent: dict[str, str] = {}
        # Public counters for tests and observability.
        self.observed_agents: list[str] = []
        self.divergence_events: list[str] = []

    # ------------------------------------------------------------------
    # Plugin-facing hooks
    # ------------------------------------------------------------------

    async def on_before_agent(
        self,
        *,
        agent_name: str,
        invocation_id: str = "",
        parent_invocation_id: str = "",
    ) -> None:
        """An agent is about to run. Claim the matching plan task.

        ``parent_invocation_id`` (goldfive#151) is optional and only
        consulted for contextual fallback matching: when the observed
        agent has no direct plan task and also no intermediate-turn
        escape (see below), the reconciler walks the invocation chain
        using the ``_invocation_agent`` map and tries to match an
        ancestor's name against the pending plan. This handles deep
        hierarchies where the plan assigned work to an ancestor
        coordinator but the tree ran the leaf directly via
        ``transfer_to_agent`` / ``AgentTool``.
        """
        if not agent_name:
            return
        self.observed_agents.append(agent_name)
        # Record the invocation → agent mapping so subsequent
        # observations can resolve parent chains. Tree-agnostic: the
        # reconciler has no notion of depth — it just stores what
        # the plugin tells it.
        if invocation_id:
            self._invocation_agent[invocation_id] = agent_name
            if parent_invocation_id:
                self._invocation_parent[invocation_id] = parent_invocation_id
        if self._is_host_agent_turn(agent_name, invocation_id):
            # Top-level host-agent before/after wraps the whole
            # dispatch; plan tasks never map to the host itself
            # unless a user explicitly assigned one, in which case
            # the matching-rule below will still find it. Don't
            # opportunistically claim a task just because the
            # coordinator's own turn opened.
            return
        task = self._pick_pending_for_agent(agent_name)
        if task is None:
            # Contextual fallback (goldfive#151): when the observed
            # agent is a leaf invoked via an ancestor coordinator
            # delegation, the ancestor's name may carry the plan
            # task. Walk the parent chain and try to claim the first
            # ancestor that has a pending plan task. This is the
            # inverse of the usual leaf-match case and handles plans
            # that assigned to a coordinator the tree routes through.
            ancestor_task = self._pick_pending_via_parent_chain(invocation_id)
            if ancestor_task is not None:
                self._running_by_agent[agent_name] = ancestor_task.id
                self._observed_task_ids.add(ancestor_task.id)
                try:
                    await self._steerer.transition(
                        ancestor_task.id,
                        TaskStatus.RUNNING,
                        detail=(
                            f"observed: {agent_name} started (contextual match via "
                            f"ancestor {ancestor_task.assignee_agent_id!r})"
                        ),
                        session=self._session,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PlanReconciler.on_before_agent: contextual transition(RUNNING) raised: %s",
                        exc,
                    )
                return
            # No PENDING task matches this agent — either an
            # off-plan agent or a plan task that already finished.
            # Emit PLAN_DIVERGENCE once per off-plan agent and
            # let the steerer decide escalation via #142's ladder.
            if agent_name in self._off_plan_seen:
                return
            # Skip if the agent's name matches ANY task (including
            # already-terminal ones) — those are "re-visits" not
            # divergence. A coordinator that delegates to the
            # research agent twice is normal.
            if self._agent_has_any_plan_task(agent_name):
                return
            # Contextual-plumbing suppression (goldfive#151): if any
            # descendant in the parent chain is itself plan-attached,
            # treat the current agent as intermediate plumbing rather
            # than divergence. Tree-agnostic — the check is over
            # invocation bookkeeping, not tree structure.
            if self._invocation_chain_contains_plan_attached_descendant(invocation_id):
                return
            # Symmetric suppression: when the agent is a descendant of
            # a plan-attached ancestor (i.e. an ancestor coordinator
            # carries a plan task), the current invocation is the
            # leaf-side of a coordinator delegation. The ancestor
            # already claimed its task; the leaf is plumbing here, not
            # divergence.
            chain = self._parent_chain(invocation_id)
            if any(self._agent_has_any_plan_task(name) for name in chain):
                return
            self._off_plan_seen.add(agent_name)
            await self._emit_divergence(
                agent_name=agent_name,
                invocation_id=invocation_id,
            )
            return
        # Claim the task: mark RUNNING and remember.
        self._running_by_agent[agent_name] = task.id
        self._observed_task_ids.add(task.id)
        try:
            await self._steerer.transition(
                task.id,
                TaskStatus.RUNNING,
                detail=f"observed: {agent_name} started",
                session=self._session,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "PlanReconciler.on_before_agent: transition(RUNNING) raised: %s",
                exc,
            )

    async def on_after_agent(
        self,
        *,
        agent_name: str,
        invocation_id: str = "",
        error: BaseException | None = None,
        summary: str = "",
        parent_invocation_id: str = "",
    ) -> None:
        """An agent finished. Close out its matched task."""
        if not agent_name:
            return
        # Keep bookkeeping up to date even on close so parent-chain
        # lookups from late observations stay resolvable.
        if invocation_id and parent_invocation_id:
            self._invocation_parent.setdefault(invocation_id, parent_invocation_id)
        if self._is_host_agent_turn(agent_name, invocation_id):
            return
        task_id = self._running_by_agent.pop(agent_name, "")
        if not task_id:
            # No matching before_agent pair — either we skipped
            # the claim (host agent) or the tree started the
            # after-agent on a branch the reconciler didn't track.
            return
        task = self._find_task(task_id)
        if task is None or task.status in TERMINAL_TASK_STATUSES:
            return
        if error is not None:
            try:
                await self._steerer.transition(
                    task_id,
                    TaskStatus.FAILED,
                    detail=f"observed: {agent_name} raised {type(error).__name__}: {error}",
                    session=self._session,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PlanReconciler.on_after_agent: transition(FAILED) raised: %s",
                    exc,
                )
            return
        try:
            await self._steerer.transition(
                task_id,
                TaskStatus.COMPLETED,
                detail=summary or f"observed: {agent_name} completed",
                session=self._session,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "PlanReconciler.on_after_agent: transition(COMPLETED) raised: %s",
                exc,
            )

    async def on_delegation_observed(
        self,
        *,
        from_agent: str,
        to_agent: str,
        invocation_id: str = "",
    ) -> None:
        """Observability-only hook. No plan-state mutation.

        Kept as an explicit method so the plugin can forward
        AgentTool delegations through one stable API. The real
        task-state work happens on before/after_agent for the
        delegated sub-agent; AgentTool spawns its own sub-Runner
        which fires its own before/after_agent pair that we pick
        up there.
        """
        # Intentionally a no-op beyond the signal that a delegation
        # happened. The intervention ladder (#142) may later
        # escalate on specific delegation patterns.
        _ = from_agent, to_agent, invocation_id

    # ------------------------------------------------------------------
    # Missed-task accounting (called by the runner / executor)
    # ------------------------------------------------------------------

    def reset_for_new_plan(self, new_plan: Plan | None) -> None:
        """Clear per-plan observation state so tasks in ``new_plan`` map fresh.

        Used when the steerer installs a revised plan mid-run
        (``USER_STEER``, ``PLAN_DIVERGENCE`` refine, etc.). The
        reconciler's internal mapping of agent invocations to plan
        tasks is plan-scoped; after the plan changes, existing
        mappings point at task ids that may no longer exist (or have
        different semantics) in the revised plan, so before/after
        agent pairs fired by the re-invoked tree would be attributed
        incorrectly.

        Clears task-to-observation bookkeeping
        (``_observed_task_ids``, ``_running_by_agent``,
        ``_off_plan_seen``) so the re-invoked tree sees a clean
        slate. Preserves cumulative ``observed_agents`` /
        ``divergence_events`` — those are historical records the
        caller may want for replay / introspection (see
        goldfive#144), not plan-scoped claim state.

        The ``new_plan`` argument is accepted for clarity at the call
        site (so it's obvious the caller is installing a new plan);
        this method does not read any tasks from it — the session's
        live plan remains the source of truth via ``get_missed_tasks``.
        """
        _ = new_plan  # reserved: may populate hints from new plan in future
        self._observed_task_ids.clear()
        self._running_by_agent.clear()
        self._off_plan_seen.clear()

    def get_missed_tasks(self, plan: Plan | None = None) -> list[Task]:
        """Return PENDING tasks the tree never exercised.

        Called by the executor's overlay-mode loop after the single
        invocation ends. Each returned task is a candidate for a
        soft follow-up via :meth:`ADKAdapter.invoke_follow_up`.

        If ``plan`` is ``None`` the reconciler reads the session's
        live plan — which may differ from the plan at the start of
        the invocation if the steerer swapped in a revision mid-
        run. That's correct: missed-task detection should always
        operate on the latest plan shape.
        """
        target = plan if plan is not None else self._session.plan
        if target is None:
            return []
        missed: list[Task] = []
        for task in target.tasks:
            if task.status is not TaskStatus.PENDING:
                continue
            # Deduplicate: a task we saw go RUNNING but didn't see
            # complete (e.g. because the invocation was cancelled)
            # should not be counted as "never exercised".
            if task.id in self._observed_task_ids:
                continue
            missed.append(task)
        return missed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_host_agent_turn(self, agent_name: str, invocation_id: str) -> bool:
        """Return True when ``agent_name`` is the host agent we wrap.

        Used to skip plan-task claims for the outermost invocation's
        before/after pair so the coordinator's own turn doesn't
        accidentally consume a task assigned to a sub-agent.

        When the plan explicitly assigns a task to the host agent
        (flat single-agent trees), this returns False so the regular
        claiming logic still matches — we only return True when
        the agent name matches the host AND no plan task is assigned
        to it.
        """
        _ = invocation_id  # reserved for future nesting heuristics
        if not self._host_agent_name:
            return False
        if agent_name != self._host_agent_name:
            return False
        # If the plan assigns any task to this agent name, treat
        # this as a legitimate task turn and let the claim logic
        # run.
        return not self._agent_has_any_plan_task(agent_name)

    def _pick_pending_for_agent(self, agent_name: str) -> Task | None:
        """Return the first PENDING task whose assignee matches."""
        plan = self._session.plan
        if plan is None:
            return None
        for task in plan.tasks:
            if task.status is not TaskStatus.PENDING:
                continue
            if not task.assignee_agent_id:
                continue
            if task.assignee_agent_id == agent_name:
                return task
        return None

    def _parent_chain(self, invocation_id: str) -> list[str]:
        """Return the ordered list of ancestor agent names for ``invocation_id``.

        Walks ``_invocation_parent`` up to the root. Stops on cycles
        (shouldn't happen, but defensive) and on missing edges. Returns
        ``[]`` when no chain is known.

        Tree-agnostic: the reconciler does not read any tree metadata;
        it composes the chain purely from observations it has already
        received.
        """
        if not invocation_id:
            return []
        names: list[str] = []
        seen: set[str] = set()
        cur = self._invocation_parent.get(invocation_id, "")
        while cur and cur not in seen:
            seen.add(cur)
            parent_name = self._invocation_agent.get(cur, "")
            if parent_name:
                names.append(parent_name)
            cur = self._invocation_parent.get(cur, "")
        return names

    def _pick_pending_via_parent_chain(self, invocation_id: str) -> Task | None:
        """Return a PENDING task whose assignee is in the parent chain.

        Contextual match (goldfive#151): the leaf observation has no
        directly-assigned plan task, but an ancestor coordinator on its
        invocation chain does. Credit the leaf observation to the first
        pending ancestor task we find walking up the chain.
        """
        chain = self._parent_chain(invocation_id)
        if not chain:
            return None
        for ancestor in chain:
            # Skip the host agent — its task semantics are handled by
            # the direct-match rules.
            task = self._pick_pending_for_agent(ancestor)
            if task is not None:
                return task
        return None

    def _invocation_chain_contains_plan_attached_descendant(
        self,
        invocation_id: str,
    ) -> bool:
        """Return True when any descendant observation already claimed a task.

        Used to suppress PLAN_DIVERGENCE for intermediate coordinators
        that the planner did not assign tasks to but which merely
        route through to task-attached leaves. Kept as a small check
        over the observation history — the reconciler never looks at
        tree metadata directly (tree-agnostic invariant).
        """
        if not invocation_id:
            return False
        # Any observed invocation whose parent chain passes through
        # ``invocation_id`` and whose agent_name has a plan task is a
        # task-attached descendant.
        for inv_id, parent in self._invocation_parent.items():
            cur = parent
            seen: set[str] = set()
            while cur and cur not in seen:
                seen.add(cur)
                if cur == invocation_id:
                    desc_name = self._invocation_agent.get(inv_id, "")
                    if desc_name and self._agent_has_any_plan_task(desc_name):
                        return True
                    break
                cur = self._invocation_parent.get(cur, "")
        return False

    def _agent_has_any_plan_task(self, agent_name: str) -> bool:
        plan = self._session.plan
        if plan is None:
            return False
        return any(t.assignee_agent_id == agent_name for t in plan.tasks)

    def _find_task(self, task_id: str) -> Task | None:
        plan = self._session.plan
        if plan is None or not task_id:
            return None
        for t in plan.tasks:
            if t.id == task_id:
                return t
        return None

    async def _emit_divergence(
        self,
        *,
        agent_name: str,
        invocation_id: str,
    ) -> None:
        """Emit a PLAN_DIVERGENCE drift via the steerer.

        INFO severity by default. The steerer's drift pipeline
        (and, once #142 lands, the intervention ladder) decides
        whether to escalate. We deliberately do not refine here:
        an off-plan agent observation is informational until the
        steerer has enough context to decide.
        """
        detail = (
            f"observed off-plan agent {agent_name!r} "
            f"(invocation={invocation_id or '?'}); no plan task "
            f"assigned to this agent"
        )
        self.divergence_events.append(detail)
        drift = DriftEvent(
            kind=DriftKind.PLAN_DIVERGENCE,
            severity=DriftSeverity.INFO,
            detail=detail,
            current_task_id=self._session.current_task_id or "",
            current_agent_id=agent_name,
        )
        # Prefer the steerer's internal _handle_drift (pre-built
        # DriftEvent → DriftDetected emission + refine policy). The
        # steerer's ``observe`` path expects a raw event to classify,
        # not a DriftEvent, so passing the pre-built drift through
        # ``observe`` would round-trip through detect_drift and
        # almost certainly return None. ``_handle_drift`` is the
        # same entry point other plugin-side drifts (CONFABULATION_RISK,
        # RUNAWAY_DELEGATION) already use.
        handle = getattr(self._steerer, "_handle_drift", None)
        if callable(handle):
            try:
                await handle(drift, self._session)
                return
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "PlanReconciler._emit_divergence: _handle_drift raised: %s",
                    exc,
                )
        # Fallback for steerer stubs without _handle_drift: feed
        # the DriftEvent through observe() and let custom steerers
        # decide. Tests that stub the steerer lean on this path.
        try:
            observe = getattr(self._steerer, "observe", None)
            if observe is not None:
                await observe(drift, self._session)
        except Exception as exc:  # noqa: BLE001
            log.debug("PlanReconciler._emit_divergence: observe raised: %s", exc)


__all__ = ["PlanReconciler"]

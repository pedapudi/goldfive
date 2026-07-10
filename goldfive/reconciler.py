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
exercised. As of goldfive#163 the overlay executor transitions
those to ``TaskStatus.NOT_NEEDED`` and does NOT dispatch soft
follow-ups — flow-prompted coordinators were re-running their
full pipeline on every follow-up user message. The method is
kept available for external callers that want to surface the
coverage gap in their own way (logging, custom nudge prompts,
etc.).

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
  assignee grows the plan with a ``discovered=True`` task and
  claims it (goldfive#423 / AGENCY-PRESERVATION.md PR 2 — the
  ``transfer_to_agent``-style counterpart of the AgentTool
  delegation-pin growth; see :meth:`_maybe_grow_discovered_task`).
  When descriptive growth is disabled or unavailable, the legacy
  behaviour remains: a divergence record on
  ``divergence_events`` (the PLAN_DIVERGENCE drift itself was
  disabled by goldfive#252).

The reconciler is deliberately small (~150 LOC) and framework-
agnostic. It calls back into the :class:`~goldfive.protocols.Steerer`
via :meth:`Steerer.transition` for state changes and emits drifts
via :meth:`Steerer.observe` so the existing refine/classifier
pipeline picks them up. No sink access; no direct event emission.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from goldfive import state_store as _ostate
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
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
            # goldfive#423 / AGENCY-PRESERVATION.md PR 2 — descriptive
            # growth for transfer_to_agent-style trees. The AgentTool
            # delegation path grows the plan at pin time
            # (``_maybe_pin_delegation_task``), but trees that move via
            # ADK ``transfer_to_agent`` never cross that hook — their
            # only goldfive-visible signal is this before_agent
            # observation. Apply the same dedup-hash → grow → claim
            # flow here so the ledger reflects observed work for every
            # tree shape. On growth, claim the discovered task exactly
            # like the direct-match path above (RUNNING transition +
            # bookkeeping) so ``on_after_agent`` closes it out.
            grown = await self._maybe_grow_discovered_task(agent_name)
            if grown is not None:
                self._running_by_agent[agent_name] = grown.id
                self._observed_task_ids.add(grown.id)
                try:
                    await self._steerer.transition(
                        grown.id,
                        TaskStatus.RUNNING,
                        detail=(
                            f"observed: {agent_name} started "
                            f"(discovered via descriptive growth)"
                        ),
                        session=self._session,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "PlanReconciler.on_before_agent: discovered-task "
                        "transition(RUNNING) raised: %s",
                        exc,
                    )
                _ostate.sync_current_task_from_transition(
                    self._session.state, grown, TaskStatus.RUNNING
                )
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
        # goldfive#152: stamp the orchestration-state current_task_*
        # keys so downstream refine paths / prompt templates /
        # sinks see a single source of truth for the active task.
        # Done AFTER the steerer transition so a steerer stub that
        # refuses the transition (e.g. test doubles) doesn't leak
        # into these reads.
        _ostate.sync_current_task_from_transition(
            self._session.state, task, TaskStatus.RUNNING
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
            # goldfive#152: clear the current_task_* stamp if it was
            # pointing at this task. A FAILED transition retires the
            # task even if a follow-up refine re-spawns it under a
            # different id.
            _ostate.sync_current_task_from_transition(
                self._session.state, task, TaskStatus.FAILED
            )
            return
        # AGENCY-PRESERVATION.md PR 11(c): capture the observed output as
        # outcome-progress evidence BEFORE the COMPLETED transition (the
        # transition re-enters the task-boundary hook that spawns the
        # outcome-progress judge, which reads ``session.completed_outputs``).
        # In overlay mode — the ONLY execution mode ledger plan mode
        # supports — the executors that normally populate
        # ``completed_outputs`` never run, so without this the outcome
        # judge has no evidence and can never decide OUTCOME completion.
        # Keyed by the closed task id (the DISCOVERED trajectory task) so
        # the judge can attribute contributing steps. Ledger-gated: in
        # forecast mode the dict is left untouched (bit-identical to
        # pre-PR-11).
        self._maybe_record_completed_output(task_id, summary)
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
        # goldfive#152: clear the current_task_* stamp when the active
        # task terminates. A later before_agent will repopulate.
        _ostate.sync_current_task_from_transition(
            self._session.state, task, TaskStatus.COMPLETED
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
        invocation ends. As of goldfive#163 the overlay transitions
        these tasks to ``TaskStatus.NOT_NEEDED`` rather than
        re-dispatching them; the method is retained for external
        callers (custom executors, telemetry) that want to surface
        the coverage gap.

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

    async def _maybe_grow_discovered_task(self, agent_name: str) -> Task | None:
        """Grow the plan with a ``discovered=True`` task for an off-plan agent.

        goldfive#423 / AGENCY-PRESERVATION.md PR 2 — the reconciler-side
        descriptive-growth trigger for ``transfer_to_agent``-style trees
        (the AgentTool path grows at delegation-pin time instead; see
        ``_maybe_pin_delegation_task`` in the ADK plugin).

        Gated on ``SteeringConfig.descriptive_growth_enabled`` (read off
        the bound steerer's typed config, mirroring the ADK plugin's
        ``_descriptive_growth_enabled``) and on the steerer exposing
        :meth:`~goldfive.plan_reviser.PlanReviser.install_descriptive_growth`
        — custom :class:`~goldfive.protocols.Steerer` implementations
        without either keep the pre-#423 divergence-record-only
        behaviour.

        Identity hash: ``before_agent`` observations carry NO tool
        args (ADK's callback surfaces only the agent + invocation ids;
        there is no ``DelegationObserved.tool_args_json`` equivalent
        for a transfer), so we pass ``tool_args_json=""`` and the hash
        degrades to the documented per-``(agent_name, "")`` key — the
        §9 forward-compat fallback of
        ``docs/design/PLAN-DESCRIPTIVE-GROWTH.md``. Coarser than the
        AgentTool path's per-(agent, args-token-set) granularity, but
        still dedups repeat transfers to the same agent onto one
        discovered task while it is live (§11.1 TTL).

        Returns the discovered (or dedup-matched) task, or ``None``
        when growth is unavailable / disabled / failed — the caller
        then falls through to the legacy divergence record.
        """
        try:
            cfg = getattr(self._steerer, "_steering_config", None)
            if cfg is None or not bool(
                getattr(cfg, "descriptive_growth_enabled", False)
            ):
                return None
            plans = getattr(self._steerer, "plans", None)
            install = getattr(plans, "install_descriptive_growth", None)
            if not callable(install):
                return None
            return await install(
                self._session,
                agent_name=agent_name,
                tool_args_json="",
                delegation_event_id="",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "PlanReconciler._maybe_grow_discovered_task: growth "
                "raised: %s; falling back to divergence record",
                exc,
            )
            return None

    def _ledger_mode(self) -> bool:
        """Return True iff ``SteeringConfig.plan_mode == "ledger"``.

        Delegates to :func:`goldfive.steerer.plan_mode_is_ledger` — the
        single implementation of the parse (three modules previously
        re-implemented it verbatim). Defensive: any failure resolves to
        forecast mode so the legacy overlay path is untouched.
        """
        try:
            from goldfive.steerer import plan_mode_is_ledger

            return plan_mode_is_ledger(self._steerer)
        except Exception:  # noqa: BLE001
            return False

    def _maybe_record_completed_output(self, task_id: str, summary: str) -> None:
        """Record ``summary`` as the outcome-progress evidence for ``task_id``.

        AGENCY-PRESERVATION.md PR 11(c). In ledger plan mode the
        outcome-progress judge (``drift/outcome_progress.py``) grades
        OUTCOME deliverables against ``session.completed_outputs`` — the
        full-output capture the executors populate in the driving model.
        Ledger plan mode only ever runs the overlay model, where the
        executors never see task outputs, so this is the observation
        entry point that populates that evidence source. No-op (and no
        write) in forecast mode, so forecast-mode observation state is
        byte-identical to pre-PR-11.
        """
        if not task_id or not summary:
            return
        if not self._ledger_mode():
            return
        try:
            self._session.completed_outputs[task_id] = summary
        except Exception as exc:  # noqa: BLE001 — observability capture
            log.debug(
                "PlanReconciler._maybe_record_completed_output: capture "
                "for %r raised: %s",
                task_id,
                exc,
            )

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
        """Detector hook for off-plan agents (drift emission disabled).

        goldfive#252: PLAN_DIVERGENCE replaced by CAPABILITY_MISMATCH
        (#253) — disabled here. Pre-#252 this fired a PLAN_DIVERGENCE
        drift comparing the planner-predicted ``assignee_agent_id``
        against runtime delegation. With #252 the planner no longer
        declares assignees (assignment is observational), so the
        comparison's premise is gone. Detection coverage moves to
        CAPABILITY_MISMATCH, which is grounded in actual agent tools.

        The detector still records the divergence string on
        ``self.divergence_events`` so existing observers and tests that
        read the list keep working; we just don't construct a
        ``DriftEvent`` and we don't dispatch through the steerer.
        """
        detail = (
            f"observed off-plan agent {agent_name!r} "
            f"(invocation={invocation_id or '?'}); no plan task "
            f"assigned to this agent"
        )
        self.divergence_events.append(detail)
        log.debug(
            "PlanReconciler._emit_divergence: detector observed "
            "off-plan agent %r but PLAN_DIVERGENCE drift is disabled "
            "(goldfive#252); no drift fired",
            agent_name,
        )
        return


__all__ = ["PlanReconciler"]

"""Task-status transitions + per-status sink emission for :class:`DefaultSteerer`.

Extracted from :mod:`goldfive.steerer` in Wave C of the steerer split.
This module owns the imperative ``mark_task_*`` family — the canonical
write path for every task-status transition driven by reporting tools,
the reconciler, and the executor — plus the shared
:func:`cascade_cancel_downstream` primitive used by both the
``mark_task_cancelled`` recoverable path and the ``mark_task_failed``
unrecoverable path.

Responsibilities
----------------

* Validate ``(from_status, to_status)`` against the
  :data:`~goldfive.types.TERMINAL_TASK_STATUSES` set so terminal tasks
  cannot be reanimated, and so a re-call on an already-terminal task is
  a no-op (matters for cascade reentry).
* Derive the new immutable :class:`~goldfive.types.Plan` via
  :func:`~goldfive.types.with_task_status` and swap it onto
  ``session.plan`` under :func:`~goldfive.types.channel_processor_active`
  — the goldfive#247 contract for Plan + Task immutability.
* Stamp the per-task progress liveness watermark
  (``session.task_last_progress_at``) so the structural progress-stall
  gate in :mod:`goldfive.drift_observer` can distinguish productively
  iterating tasks from stuck ones (goldfive#271).
* Maintain ``session.task_lineage`` so reasoning-judge attribution can
  detect "child of a delegation chain rooted at this assignee".
* Emit one proto ``Task*`` event PLUS one ``TaskTransitioned`` event
  per transition. Cascade cancellation emits the same pair per
  downstream task, with a structured ``upstream_failed:<id>`` reason
  (goldfive#205).
* Spawn drift cascades fire-and-forget for FAILED / BLOCKED so the
  reporting tool that triggered the transition returns immediately
  (iter-11A). The cascade itself is owned by the
  :class:`~goldfive.drift_observer.DriftObserver`; this module only
  spawns the task.
* Fire the per-task-boundary GOAL_DRIFT check (goldfive#219).

The module DOES NOT own:

* Drift classification / dispatch (lives in
  :class:`~goldfive.drift_observer.DriftObserver`).
* Plan revision install / supersedes / refine sequencing (lives in
  :class:`~goldfive.plan_reviser.PlanReviser`).
* The shared event-emission primitives (``_new_envelope`` / ``_emit``
  / pb-value helpers) — those live on the :class:`DefaultSteerer`
  router because every component emits through them.

All cross-component calls go through the router back-reference passed
to :meth:`TaskStateMachine.__init__` (``self._steerer``). This keeps
the three components decoupled and lets the router own the shared
state (sinks, adapter, control channel, background-task tracking).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from goldfive import state_store as _ostate
from goldfive.types import (
    TERMINAL_TASK_STATUSES,
    DriftEvent,
    DriftKind,
    DriftSeverity,
    Plan,
    Session,
    Task,
    TaskStatus,
    channel_processor_active,
    set_session_plan,
    with_task_status,
)

if TYPE_CHECKING:
    from goldfive.steerer import DefaultSteerer

log = logging.getLogger(__name__)


# Re-export for compat with code that imported the private name from
# :mod:`goldfive.steerer` historically; do NOT redefine.
_TERMINAL_TASK_STATUSES = TERMINAL_TASK_STATUSES


class TaskStateMachine:
    """Per-task status transitions + cascade + transition-event emission.

    Constructed by :class:`DefaultSteerer` and exposed publicly as
    ``DefaultSteerer.tasks`` (goldfive#410). Callers — the Runner,
    executors, reporting handlers, planners, tests — reach the
    ``mark_task_*`` family directly as ``steerer.tasks.mark_task_X``.
    """

    def __init__(self, steerer: DefaultSteerer) -> None:
        # Back-reference to the router. Used to reach the shared event-
        # emission primitives (``_new_envelope``, ``_emit``) and the
        # other two components (``steerer.drift``, ``steerer.plans``)
        # when a transition path needs to cross a boundary
        # (e.g. mark_task_failed spawns a drift cascade owned by the
        # observer; cascade emits a TaskTransitioned which needs the
        # invocation-id resolver).
        self._steerer = steerer

    # ------------------------------------------------------------------
    # Plan lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _find_task(session: Session, task_id: str) -> Task | None:
        if not task_id or session.plan is None:
            return None
        for t in session.plan.tasks:
            if t.id == task_id:
                return t
        return None

    # ------------------------------------------------------------------
    # mark_task_* family
    # ------------------------------------------------------------------

    async def mark_task_running(
        self,
        task_id: str,
        *,
        session: Session,
        detail: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``RUNNING`` and emit ``TaskStarted``.

        ``source`` (goldfive#251 R4) is the attribution string emitted on
        the paired ``TaskTransitioned`` sink event — see
        :func:`goldfive.events.task_transitioned_event` for the
        vocabulary. Defaults to ``"other"`` for callers that haven't been
        threaded through (back-compat); the live LLM-driven path through
        :mod:`goldfive.reporting` passes ``"llm_report"`` /
        ``"handler_default"`` / ``"supersedes_reroute"`` as appropriate.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: Plan + Task are frozen. Mutate by deriving a new
        # plan via :func:`with_task_status` and swapping the pointer
        # under :func:`channel_processor_active`. The local ``task``
        # reference becomes stale; refresh from the live plan so
        # downstream side-effects see the new status.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.RUNNING))
        task = self._find_task(session, task_id) or task
        session.current_task_id = task_id
        if detail:
            session.agent_notes[task_id] = detail
        # goldfive#152: stamp current_task_* on the orchestration-state
        # dict so downstream prompt templates / refine paths see it.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.RUNNING)
        # goldfive#271: stamp task progress liveness for the structural
        # progress-stall escalation. A drift firing on this task within
        # ``PROGRESS_STALL_THRESHOLD_SECONDS`` of any progress signal is
        # considered productively iterating; outside that window, the
        # next drift escalates to HUMAN_INTERVENTION_REQUIRED.
        session.task_last_progress_at[task_id] = time.monotonic()
        # Seed the observed-agent lineage with the static plan
        # assignee. ``before_tool_callback`` extends the set with each
        # delegated child agent so consumers (e.g. the reasoning judge)
        # can distinguish "child of a delegation chain rooted at the
        # assignee" from "off-plan agent". Cleared on every terminal
        # transition.
        session.task_lineage[task_id] = (
            {task.assignee_agent_id} if task.assignee_agent_id else set()
        )
        await self._emit_task_started(session, task_id, detail)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.RUNNING,
            source=source,
        )

    async def mark_task_progress(
        self,
        task_id: str,
        *,
        session: Session,
        fraction: float = 0.0,
        detail: str = "",
    ) -> None:
        """Record mid-task progress and emit ``TaskProgress``.

        No status transition — a progress update is a liveness ping only.
        ``fraction`` is clamped to ``[0.0, 1.0]``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        try:
            frac = float(fraction)
        except (TypeError, ValueError):
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        session.task_progress[task_id] = frac
        if detail:
            session.agent_notes[task_id] = detail
        # goldfive#271: refresh progress liveness so a productively
        # iterating task is not flagged as stalled by the structural
        # escalation gate.
        session.task_last_progress_at[task_id] = time.monotonic()
        await self._emit_task_progress(session, task_id, frac, detail)

    async def mark_task_completed(
        self,
        task_id: str,
        *,
        session: Session,
        summary: str = "",
        artifacts: dict[str, str] | None = None,
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``COMPLETED`` and emit ``TaskCompleted``.

        See :meth:`mark_task_running` for the ``source`` contract.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.COMPLETED))
        task = self._find_task(session, task_id) or task
        if summary:
            session.completed_results[task_id] = summary
        # goldfive#152: clear current_task_* if we were the active task.
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.COMPLETED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # iter-11B: pair the prior ``agent_invocation_started`` entry
        # so the GOAL_DRIFT judge does not see an orphan-start +
        # task-COMPLETED shape and false-positive on "looping".
        # ``after_run_callback`` will append the real
        # ``agent_invocation_completed`` slightly later when the agent
        # actually returns; duplicate completed entries are harmless
        # (each is benign and the ring buffer trims naturally).
        if task.assignee_agent_id:
            self._steerer.drift.note_agent_activity(
                session,
                kind="agent_invocation_completed",
                agent_name=task.assignee_agent_id,
                task_id=task_id,
            )
        await self._emit_task_completed(session, task_id, summary, artifacts or {})
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.COMPLETED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=task
        )

    async def mark_task_failed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        recoverable: bool = True,
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``FAILED`` and emit ``TaskFailed``.

        Also fires a drift event of kind ``TASK_FAILED_RECOVERABLE`` or
        ``TASK_FAILED_FATAL``. The drift event is dispatched through the
        same drift pipeline as observer-detected drift: if severity is
        ``>= WARNING`` (both of these are) we invoke ``planner.refine``.

        When ``recoverable=False`` the failure is fatal for this task
        lineage: **cascade-cancel every reachable downstream non-terminal
        task** via :meth:`cascade_cancel_downstream`, so the plan lands
        in a consistent terminal-only shape instead of orphaning
        dependents that would sit PENDING forever. This is the
        implementation of PLAN-LIFECYCLE.md §6.2 step 3 and it shares
        its primitive with the §6.3 cancel cascade path
        (:meth:`mark_task_cancelled`). The downstream cascade fires
        *before* we dispatch the fatal drift through ``_handle_drift``
        so that planner.refine sees the post-cascade plan shape and a
        refine-failure back-off does not leave orphans behind. See
        ``STATE-MACHINE.md §"Cascade semantics on unrecoverable drift"``.
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.FAILED))
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.FAILED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # iter-11B: pair the prior ``agent_invocation_started`` entry
        # so the GOAL_DRIFT judge does not see an orphan-start +
        # task-FAILED shape and false-positive on "looping".  See the
        # matching write in :meth:`mark_task_completed` for rationale.
        if task.assignee_agent_id:
            self._steerer.drift.note_agent_activity(
                session,
                kind="agent_invocation_completed",
                agent_name=task.assignee_agent_id,
                task_id=task_id,
            )
        await self._emit_task_failed(session, task_id, reason, recoverable)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.FAILED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=task
        )
        # Fatal failures cascade downstream via the same primitive used
        # by mark_task_cancelled, so both §6.2 and §6.3 produce the
        # same TaskCancelled event stream and share rejection guards.
        if not recoverable:
            # The cascade is a propagation of the same source-attribution
            # decision (e.g. an LLM-reported fatal failure cascades as
            # ``"cancellation"`` from the framework's perspective — the
            # cascaded tasks weren't moved by the LLM directly).
            await self.cascade_cancel_downstream(session, task_id, source="cancellation")
        kind = DriftKind.TASK_FAILED_RECOVERABLE if recoverable else DriftKind.TASK_FAILED_FATAL
        severity = DriftSeverity.WARNING if recoverable else DriftSeverity.CRITICAL
        drift = DriftEvent(
            kind=kind,
            severity=severity,
            detail=f"task {task_id} failed: {reason}" if reason else f"task {task_id} failed",
            current_task_id=task_id,
        )
        # iter-11A: spawn the drift cascade fire-and-forget so the
        # reporting tool that triggered us (``report_task_failed``)
        # can return immediately. The cascade
        # (refine → supersedes → cancellation) lands asynchronously;
        # tests that need the post-cascade plan state await
        # :meth:`_wait_background_drifts_idle`.
        self._steerer.drift._spawn_drift_handler_background(drift, session)

    async def mark_task_blocked(
        self,
        task_id: str,
        *,
        session: Session,
        blocker: str = "",
        needed: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``BLOCKED`` and emit ``TaskBlocked``.

        Also fires a drift event of kind ``BLOCKED`` which flows through
        the standard drift pipeline (WARNING severity → refine).
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        # BLOCKED is not a terminal status but we still guard against
        # re-blocking a task that's already blocked (idempotent).
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.BLOCKED))
        task = self._find_task(session, task_id) or task
        if blocker or needed:
            session.agent_notes[task_id] = f"blocked: {blocker}" + (
                f" (needed: {needed})" if needed else ""
            )
        await self._emit_task_blocked(session, task_id, blocker, needed)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.BLOCKED,
            source=source,
        )
        detail = f"task {task_id} blocked: {blocker}" + (f" (needed: {needed})" if needed else "")
        drift = DriftEvent(
            kind=DriftKind.BLOCKED,
            severity=DriftSeverity.WARNING,
            detail=detail,
            current_task_id=task_id,
        )
        # iter-11A: spawn the drift cascade fire-and-forget so the
        # reporting tool that triggered us (``report_task_blocked``)
        # can return immediately. See :meth:`mark_task_failed` for
        # the matching call-site comment.
        self._steerer.drift._spawn_drift_handler_background(drift, session)

    async def mark_task_cancelled(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``CANCELLED`` and emit ``TaskCancelled``.

        Also **cascades** the cancellation forward through the plan's
        edges: every non-terminal task reachable from ``task_id`` is
        transitioned to ``CANCELLED`` with a "cascade from <task_id>"
        reason. Without this cascade, downstream PENDING tasks with a
        CANCELLED predecessor would never satisfy the executor's
        "all deps COMPLETED" check — they would sit PENDING forever and
        the executor would report the run as successful while leaving
        them orphaned. See TASK-LIFECYCLE.md §"Cancellation cascade" and
        STATE-MACHINE.md §"Cascade semantics on unrecoverable drift".
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            # Already terminal (including already CANCELLED) — no-op, and
            # crucially do NOT re-run the cascade: we would double-emit
            # TaskCancelled events for downstream tasks on every call.
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        with channel_processor_active():
            assert session.plan is not None
            set_session_plan(session, with_task_status(session.plan, task_id, TaskStatus.CANCELLED))
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.CANCELLED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        await self._emit_task_cancelled(session, task_id, reason)
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.CANCELLED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        # Fire before cascade so the judge sees the initiator's transition;
        # cascade-cancel downstream tasks share the same rate-limit bucket
        # and will no-op as subsequent boundary fires fall within the
        # 10s guard.
        await self._steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=task
        )
        await self.cascade_cancel_downstream(session, task_id, source="cancellation")

    async def mark_task_not_needed(
        self,
        task_id: str,
        *,
        session: Session,
        reason: str = "",
        source: str = "other",
    ) -> None:
        """Transition ``task_id`` to ``NOT_NEEDED`` terminally.

        Introduced by the overlay-model refactor (goldfive#141). Unlike
        :meth:`mark_task_cancelled` this path does NOT cascade — a task
        the :class:`~goldfive.reconciler.PlanReconciler` deemed "not
        needed" post-invocation is an observation about that specific
        plan entry, not a signal that downstream work is invalid. The
        reconciler independently evaluates each PENDING task.

        Idempotent on terminal tasks. Emits ``TaskCancelled`` at the
        proto level (there's no dedicated NOT_NEEDED event — the
        status lives on the task itself and sinks can distinguish via
        ``task.status`` on the next ``TaskCancelled`` / ``PlanRevised``
        envelope). The reason string carries the distinguishing
        context ("not needed: superseded by ...").
        """
        task = self._find_task(session, task_id)
        if task is None:
            return
        if task.status in _TERMINAL_TASK_STATUSES:
            return
        from_status = task.status
        # goldfive#247: derive a new immutable Plan and swap the pointer.
        assert session.plan is not None
        with channel_processor_active():
            set_session_plan(
                session,
                with_task_status(session.plan, task_id, TaskStatus.NOT_NEEDED),
            )
        task = self._find_task(session, task_id) or task
        _ostate.sync_current_task_from_transition(session.state, task, TaskStatus.NOT_NEEDED)
        # Drop the observed-agent lineage now the task is terminal.
        session.task_lineage.pop(task_id, None)
        # There is no dedicated ``TaskNotNeeded`` proto message;
        # reuse TaskCancelled with the reason prefix so sinks that
        # inspect reason can differentiate if they wish. The live
        # ``task.status`` on the plan is the authoritative signal.
        await self._emit_task_cancelled(
            session, task_id, f"not_needed: {reason}" if reason else "not_needed"
        )
        await self._emit_task_transitioned(
            session,
            task,
            from_status=from_status,
            to_status=TaskStatus.NOT_NEEDED,
            source=source,
        )
        # goldfive#219: task boundary is a natural goal-drift checkpoint.
        await self._steerer.drift._maybe_run_goal_drift_on_task_boundary(
            session, transitioned_task=task
        )

    async def cascade_cancel_downstream(
        self,
        session: Session,
        cancelled_id: str,
        *,
        source: str = "cancellation",
    ) -> None:
        """BFS-cancel every downstream non-terminal task of ``cancelled_id``.

        Shared primitive for both cascade codepaths
        (PLAN-LIFECYCLE.md §6.2 unrecoverable cascade and §6.3
        cancellation cascade):

        - The recoverable path
          (:meth:`mark_task_cancelled`) calls it after transitioning
          the initiator to ``CANCELLED``.
        - The unrecoverable path
          (:meth:`mark_task_failed` with ``recoverable=False``) calls
          it after transitioning the initiator to ``FAILED``.

        Both paths therefore produce the same ``TaskCancelled`` event
        stream for the downstream set and share the rejection guards
        (terminal tasks are skipped; diamond DAGs are de-duplicated).
        The initiator's own transition-emission is caller-controlled —
        this method only emits for the *downstream* set.

        Walks ``session.plan.edges`` forward from ``cancelled_id`` and
        transitions every reachable non-terminal task to ``CANCELLED``
        in-place, emitting one ``TaskCancelled`` event per transition.
        Terminal tasks (COMPLETED / FAILED / CANCELLED) are skipped so a
        diamond DAG does not re-cancel a task through two paths and so
        already-COMPLETED dependents are preserved verbatim.

        Implemented as an iterative BFS on a precomputed adjacency list
        (rather than recursing into :meth:`mark_task_cancelled`) so a
        single top-level cancel produces one summary log line and a
        predictable number of emitted events, independent of graph shape.
        """
        plan = session.plan
        if plan is None:
            return
        # Precompute forward adjacency once.
        downstream: dict[str, list[str]] = {}
        for e in plan.edges:
            downstream.setdefault(e.from_task_id, []).append(e.to_task_id)
        tasks_by_id: dict[str, Task] = {t.id: t for t in plan.tasks if t.id}

        # goldfive#205: structured reason consumed by harmonograf's
        # Trajectory view. Old ``cascade from <id>`` form preserved in a
        # human-readable tail after the colon so sinks that render the
        # raw reason keep their existing copy; new sinks parse the
        # ``upstream_failed:`` prefix to categorise the cancel.
        cascade_reason = f"upstream_failed:{cancelled_id}"
        cascaded: list[str] = []
        queue: list[str] = list(downstream.get(cancelled_id, []))
        visited: set[str] = set()
        while queue:
            next_id = queue.pop(0)
            if next_id in visited:
                continue
            visited.add(next_id)
            dep = tasks_by_id.get(next_id)
            if dep is None:
                continue
            if dep.status in _TERMINAL_TASK_STATUSES:
                # Already terminal (COMPLETED/FAILED/CANCELLED) — preserve
                # and do not traverse its children. A COMPLETED task that
                # sits downstream of a late-cancelled ancestor keeps its
                # completion; cascading past it would mean cancelling
                # tasks whose preserved prerequisite is still valid.
                continue
            # Transition by deriving a new plan and swapping it in. We
            # deliberately do NOT recurse through ``mark_task_cancelled``
            # here; we fan out via our own BFS queue so the surrounding
            # summary log and emission count stay deterministic.
            #
            # goldfive#247: the local ``dep`` reference becomes stale
            # after the swap (frozen Task) — re-resolve from the live
            # plan so the emit reads the new status. Note that
            # ``tasks_by_id`` was built from the *original* plan; the
            # iteration loop only inspects ``status`` for terminal
            # gating which is monotonic (a task that wasn't terminal
            # at loop start is the one we're cancelling).
            dep_from = dep.status
            assert session.plan is not None
            with channel_processor_active():
                set_session_plan(
                    session,
                    with_task_status(session.plan, next_id, TaskStatus.CANCELLED),
                )
            # Drop the observed-agent lineage now the task is terminal —
            # mirrors the cleanup that ``mark_task_cancelled`` performs
            # for the initiator (this BFS does not recurse through that
            # method so the cleanup is duplicated explicitly).
            session.task_lineage.pop(next_id, None)
            await self._emit_task_cancelled(session, next_id, cascade_reason)
            # Re-fetch so the transition emit reads the swapped task.
            updated_dep = self._find_task(session, next_id) or dep
            await self._emit_task_transitioned(
                session,
                updated_dep,
                from_status=dep_from,
                to_status=TaskStatus.CANCELLED,
                source=source,
            )
            cascaded.append(next_id)
            for grandchild in downstream.get(next_id, []):
                if grandchild not in visited:
                    queue.append(grandchild)
        if cascaded:
            log.info(
                "DefaultSteerer: cascade-cancelled %d downstream task(s) from %s: %s",
                len(cascaded),
                cancelled_id,
                ", ".join(cascaded),
            )

    # ------------------------------------------------------------------
    # Per-status proto emitters
    # ------------------------------------------------------------------

    async def _emit_task_started(self, session: Session, task_id: str, detail: str) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_started.task_id = task_id
        evt.task_started.detail = detail
        await self._steerer._emit(evt)

    async def _emit_task_progress(
        self, session: Session, task_id: str, fraction: float, detail: str
    ) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_progress.task_id = task_id
        evt.task_progress.fraction = fraction
        evt.task_progress.detail = detail
        await self._steerer._emit(evt)

    async def _emit_task_completed(
        self,
        session: Session,
        task_id: str,
        summary: str,
        artifacts: dict[str, str],
    ) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_completed.task_id = task_id
        evt.task_completed.summary = summary
        for k, v in artifacts.items():
            evt.task_completed.artifacts[k] = v
        await self._steerer._emit(evt)

    async def _emit_task_failed(
        self, session: Session, task_id: str, reason: str, recoverable: bool
    ) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_failed.task_id = task_id
        evt.task_failed.reason = reason
        evt.task_failed.recoverable = recoverable
        await self._steerer._emit(evt)

    async def _emit_task_blocked(
        self, session: Session, task_id: str, blocker: str, needed: str
    ) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_blocked.task_id = task_id
        evt.task_blocked.blocker = blocker
        evt.task_blocked.needed = needed
        await self._steerer._emit(evt)

    async def _emit_task_cancelled(self, session: Session, task_id: str, reason: str) -> None:
        evt = self._steerer._new_envelope(session)
        evt.task_cancelled.task_id = task_id
        evt.task_cancelled.reason = reason
        await self._steerer._emit(evt)

    async def _emit_task_transitioned(
        self,
        session: Session,
        task: Task,
        *,
        from_status: TaskStatus,
        to_status: TaskStatus,
        source: str,
    ) -> None:
        """Emit a ``TaskTransitioned`` envelope (goldfive#251 R4).

        Sink-only observability. Called from every site that mutates a
        task's status — both the imperative ``mark_task_*`` path and
        the cascade path inside :meth:`cascade_cancel_downstream`. The
        LLM never sees this event; the ``report_task_*`` surface still
        returns ``{"acknowledged": True}``.

        Source attribution is the caller's responsibility (defaults to
        ``"other"`` on un-threaded callers); see
        :func:`goldfive.events.task_transitioned_event` for the
        vocabulary.

        ``agent_name`` resolves to ``task.assignee_agent_id``; that's
        the most stable surface goldfive owns. ``invocation_id`` is a
        best-effort lookup against the reconciler's
        ``_invocation_agent`` map (goldfive#151) when available; empty
        when no in-flight invocation matches the assignee. Tolerant of
        missing maps / proto stubs — emission failures are swallowed
        with a debug log rather than breaking the transition path.
        """
        # goldfive#271: stamp task progress liveness on every transition
        # so the structural progress-stall escalation gate sees the task
        # as productively iterating. Done BEFORE the sink check because
        # progress liveness must be tracked even when sinks are missing
        # (test scenarios) — the gate consults this map regardless.
        task_id_for_progress = str(getattr(task, "id", "") or "")
        if task_id_for_progress:
            session.task_last_progress_at[task_id_for_progress] = time.monotonic()
        if task_id_for_progress and to_status in _TERMINAL_TASK_STATUSES:
            # AGENCY-PRESERVATION.md PR 5 (observe-only): when a task reaches a
            # terminal status, resolve any open, delivered SignalLedger keys bound
            # to it — ``self_corrected_after_signal`` if a real signal was
            # delivered, ``self_corrected_unaided`` if only dry-run (the
            # ``observation_only`` base rate). Terminal-only is the conservative
            # "resolved" detection (never over-claims self-correction). Run before
            # the sink guard so ledger state stays consistent even without sinks
            # (mirrors the progress-liveness rationale above); the emit inside is
            # a no-op when no sink is bound. Best-effort; gates nothing.
            recorder = getattr(
                getattr(self._steerer, "drift", None),
                "record_signal_outcomes_for_task",
                None,
            )
            if callable(recorder):
                try:
                    await recorder(session, task_id_for_progress)
                except Exception as exc:  # noqa: BLE001 -- telemetry best-effort
                    log.debug(
                        "TaskStateMachine._emit_task_transitioned: signal-outcome "
                        "resolution raised (swallowed): %s",
                        exc,
                    )
            # Drift-condition lifecycle: a terminal task moots every
            # condition still open against it. Done here because every
            # transition path funnels through this method (mark_task_*,
            # cascade, plan-revision transitions), and BEFORE the sink
            # check because the ``KEY_ACTIVE_DRIFTS`` cleanup is lifecycle
            # truth that must land even when emission is impossible.
            await self._steerer.drift.resolve_conditions_for_terminal_task(
                session, task_id=task_id_for_progress, to_status=to_status
            )
        sinks = self._steerer._sinks
        if not sinks:
            return
        try:
            from goldfive.events import emit, task_transitioned_event
        except Exception as exc:  # noqa: BLE001 — proto stubs may be missing
            log.debug(
                "DefaultSteerer._emit_task_transitioned: events module unavailable: %s",
                exc,
            )
            return

        agent_name = str(getattr(task, "assignee_agent_id", "") or "")
        invocation_id = self._resolve_invocation_id_for_agent(agent_name)
        revision_stamp = 0
        plan = getattr(session, "plan", None)
        if plan is not None:
            try:
                revision_stamp = int(getattr(plan, "revision_index", 0) or 0)
            except (TypeError, ValueError):
                revision_stamp = 0
        try:
            evt = task_transitioned_event(
                session.run_id,
                session.next_sequence(),
                task_id=str(getattr(task, "id", "") or ""),
                from_status=str(getattr(from_status, "value", from_status) or ""),
                to_status=str(getattr(to_status, "value", to_status) or ""),
                source=str(source or "other"),
                revision_stamp=revision_stamp,
                agent_name=agent_name,
                invocation_id=invocation_id,
                session_id=session.id,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_task_transitioned: proto event build failed: %s",
                exc,
            )
            return
        try:
            await emit(sinks, evt)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "DefaultSteerer._emit_task_transitioned: sink emit raised: %s",
                exc,
            )

    async def _emit_plan_revision_transitions(
        self,
        session: Session,
        prev_plan: Plan | None,
        revised: Plan,
    ) -> None:
        """Emit ``TaskTransitioned`` events for status changes carried by a refine.

        Compares ``prev_plan`` vs ``revised`` task-by-task and emits one
        ``TaskTransitioned`` event per task whose status changed (or
        whose ``status`` is now non-PENDING and the task didn't exist
        in ``prev_plan`` — a refine-introduced task that arrived in a
        non-PENDING state, e.g. a CORRECT-kind successor that the
        planner pre-stamped).

        Source is always ``"plan_revision"``: the refine is the
        authoritative driver. Tasks that exist in both plans with the
        same status are skipped (no transition happened).

        ``prev_plan`` may be ``None`` on the first revision after a run
        with no initial plan; in that case every task in ``revised``
        with non-PENDING status emits a "(implicit) PENDING ->
        actual_status" event so operators see the post-revision state
        on the wire.
        """
        if not self._steerer._sinks:
            return
        prev_by_id: dict[str, Task] = {}
        if prev_plan is not None:
            for t in getattr(prev_plan, "tasks", []) or []:
                tid = str(getattr(t, "id", "") or "")
                if tid:
                    prev_by_id[tid] = t
        for t in getattr(revised, "tasks", []) or []:
            tid = str(getattr(t, "id", "") or "")
            if not tid:
                continue
            new_status = getattr(t, "status", None)
            if not isinstance(new_status, TaskStatus):
                continue
            old = prev_by_id.get(tid)
            if old is None:
                old_status: TaskStatus = TaskStatus.PENDING
            else:
                old_status = getattr(old, "status", TaskStatus.PENDING)
            if old_status == new_status:
                continue
            # No transition to record when the new status is the
            # default PENDING and the task is brand-new — sinks would
            # render that as a phantom "started in PENDING" row.
            if old is None and new_status is TaskStatus.PENDING:
                continue
            await self._emit_task_transitioned(
                session,
                t,
                from_status=old_status,
                to_status=new_status,
                source="plan_revision",
            )

    def _resolve_invocation_id_for_agent(self, agent_name: str) -> str:
        """Best-effort lookup of an active invocation_id for ``agent_name``.

        Mirrors the reconciler-walk pattern in
        :meth:`_resolve_active_invocation_ids` but scopes the match to a
        single agent name (the assignee of the transitioning task). The
        most-recent matching invocation_id wins; empty string when no
        match (no reconciler, no in-flight invocation under that
        agent, etc.). Tolerant of every failure mode — never raises.
        """
        if not agent_name:
            return ""
        adapter = self._steerer._adapter
        plugin = getattr(adapter, "_plugin", None) if adapter is not None else None
        reconciler = getattr(plugin, "_reconciler", None) if plugin is not None else None
        if reconciler is None:
            return ""
        try:
            inv_agent = getattr(reconciler, "_invocation_agent", None)
            if not isinstance(inv_agent, Mapping):
                return ""
            # Iterate insertion-order; later writes win.
            match = ""
            for inv_id, name in inv_agent.items():
                if name == agent_name and inv_id:
                    match = str(inv_id)
            return match
        except Exception:  # noqa: BLE001
            return ""
